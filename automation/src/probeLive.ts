/**
 * Headed probe: open aggregator, dump snapshot, optionally try one Remove.
 * Skips interactive auth so we can validate DOM selectors against the live UI.
 *
 *   npx tsx src/probeLive.ts           # dump only
 *   npx tsx src/probeLive.ts --remove  # dump + one Remove click
 */
import { chromium } from "playwright";
import { readAggregatorSnapshot, executeRemove } from "./actions.js";
import { config } from "./config.js";
import { NetworkMonitor, installNetworkIdleHooks } from "./network.js";
import { waitForServerIdle, type SyncContext } from "./synchronization.js";
import { pickNextDecision } from "./aggregationRules.js";
import { defaultSemanticMatcher } from "./semanticMatching.js";
import { sleep } from "./utils.js";

async function main(): Promise<void> {
  const doRemove = process.argv.includes("--remove");
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();
  await installNetworkIdleHooks(page);

  const monitor = new NetworkMonitor(page);
  const ctx: SyncContext = { page, monitor };

  try {
    console.log("Goto aggregator…");
    await page.goto(config.aggregatorUrl, { waitUntil: "domcontentloaded" });
    await waitForServerIdle(ctx);

    const snapshot = await readAggregatorSnapshot(page);
    console.log(
      `Pair ${snapshot.evidence1Id}↔${snapshot.evidence2Id} | red=${snapshot.eligible.length} | redirect=${snapshot.redirectVisible}`,
    );
    for (const e of snapshot.eligible) {
      console.log(`  [${e.treeIndex}] (${e.origin}) ${e.kind} | ${e.pathNames.join(" > ")}`);
    }

    const next = pickNextDecision(snapshot, defaultSemanticMatcher, "remove");
    console.log("next remove decision:", next);

    if (doRemove && next?.decision.type === "remove") {
      console.log("Executing Remove…");
      await executeRemove(ctx, next.element);
      const after = await readAggregatorSnapshot(page);
      console.log(
        `After Remove: Pair ${after.evidence1Id}↔${after.evidence2Id} | red=${after.eligible.length}`,
      );
      for (const e of after.eligible) {
        console.log(`  [${e.treeIndex}] (${e.origin}) ${e.kind} | ${e.pathNames.join(" > ")}`);
      }
      await page.screenshot({ path: "/tmp/ef-after-remove.png", fullPage: true });
      console.log("Screenshot /tmp/ef-after-remove.png");
    } else {
      await page.screenshot({ path: "/tmp/ef-probe.png", fullPage: true });
      console.log("Screenshot /tmp/ef-probe.png");
    }

    await sleep(1500);
  } finally {
    monitor.dispose();
    await browser.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
