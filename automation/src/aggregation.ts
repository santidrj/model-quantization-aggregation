import type { Page } from "playwright";
import {
  clickUpdateEvidenceAggregation,
  clickViewAggregatedModel,
  executeAdd,
  executeHumanDecision,
  executeJoin,
  executeRemove,
  readAggregatorSnapshot,
} from "./actions.js";
import { pickNextDecision, type AggregationPolicy } from "./aggregationRules.js";
import { config as defaultConfig, type AppConfig } from "./config.js";
import { ensureAuthenticated, isAuthenticationRequired } from "./auth.js";
import type { BrowserContext } from "playwright";
import { NetworkMonitor } from "./network.js";
import { defaultSemanticMatcher } from "./semanticMatching.js";
import { waitForServerIdle, type SyncContext } from "./synchronization.js";
import type { AggregatorSnapshot } from "./types.js";

async function withRetries(maxActionRetries: number, label: string, fn: () => Promise<void>): Promise<void> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxActionRetries; attempt++) {
    try {
      await fn();
      return;
    } catch (error) {
      lastError = error;
      console.warn(`Retry ${attempt}/${maxActionRetries} for ${label}:`, error);
    }
  }
  throw lastError;
}

function snapshotKey(snapshot: AggregatorSnapshot): string {
  return [
    snapshot.evidence1Id,
    snapshot.evidence2Id,
    snapshot.redirectVisible,
    snapshot.eligible.map((e) => `${e.termIds.join(".")}:${e.origin}`).join("|"),
  ].join("::");
}

/**
 * Main aggregation workflow for one named aggregation group.
 */
export async function runAggregation(
  page: Page,
  context: BrowserContext,
  options: {
    allowAnonymous?: boolean;
    humanDefault?: "add" | "remove" | "abort";
    config?: AppConfig;
  } = {},
): Promise<void> {
  const allowAnonymous = options.allowAnonymous ?? false;
  const humanDefault = options.humanDefault;
  const config = options.config ?? defaultConfig;
  const policy: AggregationPolicy = {
    keepModelQuantizationAspects: config.keepModelQuantizationAspects,
  };
  const monitor = new NetworkMonitor(page);
  const ctx: SyncContext = { page, monitor };
  const matcher = defaultSemanticMatcher;
  const selectedParam = `selectedAggregationId=${config.aggregationGroupId}`;

  try {
    console.log(
      `Opening aggregation overview (${config.targetDisplayName} / ${config.targetSlug}, group ${config.aggregationGroupId})…`,
    );
    if (config.keepModelQuantizationAspects) {
      console.log("Policy: keep Model-quantization contextual aspects via Add.");
    }
    await page.goto(config.overviewUrl, { waitUntil: "domcontentloaded" });
    if (!allowAnonymous) {
      await ensureAuthenticated(page, context);
    }
    if (!page.url().includes(selectedParam)) {
      await page.goto(config.overviewUrl, { waitUntil: "domcontentloaded" });
    }

    if (!allowAnonymous && (await isAuthenticationRequired(page))) {
      throw new Error("Authentication expired or missing after navigation.");
    }

    console.log("Clicking Update evidence aggregation (navigates to matching editor)…");
    await clickUpdateEvidenceAggregation(ctx);
    await waitForServerIdle(ctx);

    // Initial POST /evidenceAggregator runs on document ready.
    await waitForServerIdle(ctx);

    let idleRounds = 0;
    let lastKey = "";

    while (true) {
      if (!allowAnonymous && (await isAuthenticationRequired(page))) {
        console.log("Session expired mid-run.");
        await ensureAuthenticated(page, context);
        await page.goto(config.aggregatorUrl, { waitUntil: "domcontentloaded" });
        await waitForServerIdle(ctx);
      }

      await waitForServerIdle(ctx);
      const snapshot = await readAggregatorSnapshot(page);
      const key = snapshotKey(snapshot);

      console.log(
        `Pair ${snapshot.evidence1Id}↔${snapshot.evidence2Id} | red=${snapshot.eligible.length} | redirect=${snapshot.redirectVisible}`,
      );

      if (snapshot.redirectVisible && snapshot.eligible.length === 0) {
        console.log("All differences resolved. Clicking view aggregated model…");
        await withRetries(config.maxActionRetries, "view aggregated model", async () => {
          if (/\/evidenceEditor\/displayAggregationResult/.test(page.url())) {
            return;
          }
          await clickViewAggregatedModel(ctx);
        });
        console.log("Aggregation submit completed.");
        if (process.env.EF_CAPTURE_RESULT === "1") {
          try {
            await page.screenshot({ path: "/tmp/ef-golden-result.png", fullPage: true });
            const body = (await page.locator("body").innerText()).slice(0, 6000);
            console.log("--- result tree text ---\n" + body + "\n--- end result tree ---");
          } catch (error) {
            console.warn("Could not capture result screenshot/text:", error);
          }
        }
        return;
      }

      if (snapshot.eligible.length === 0) {
        idleRounds += 1;
        if (idleRounds > 40) {
          throw new Error("No eligible red elements and view aggregated model never appeared.");
        }
        // Wait for possible in-flight pair advance without fixed long sleeps.
        await waitForServerIdle(ctx);
        continue;
      }
      idleRounds = 0;

      const phases = ["remove", "join", "add", "residual"] as const;
      let acted = false;

      for (const phase of phases) {
        const next = pickNextDecision(snapshot, matcher, phase, policy);
        if (!next) continue;

        const { element, decision } = next;
        const label = `${phase}:${decision.type}:${element.label}(${element.origin})`;
        console.log(`Action → ${label} @ ${element.pathNames.join(" > ")}`);

        await withRetries(config.maxActionRetries, label, async () => {
          if (decision.type === "remove") {
            await executeRemove(ctx, element);
          } else if (decision.type === "join") {
            await executeJoin(ctx, element, decision.partnerLabel);
          } else if (decision.type === "add") {
            await executeAdd(ctx, element);
          } else {
            const result = await executeHumanDecision(ctx, element, decision.reason, {
              defaultAction: humanDefault,
            });
            if (result === "abort") throw new Error("Aborted by reviewer.");
          }
        });

        acted = true;
        break;
      }

      if (!acted) {
        throw new Error("Eligible elements remain but no policy decision was produced.");
      }

      // Stale-guard: if nothing changed, surface a failure instead of spinning forever.
      const after = await readAggregatorSnapshot(page);
      const afterKey = snapshotKey(after);
      if (afterKey === key && afterKey === lastKey) {
        console.warn("Snapshot unchanged after action; continuing with caution.");
      }
      lastKey = afterKey;
    }
  } finally {
    monitor.dispose();
  }
}
