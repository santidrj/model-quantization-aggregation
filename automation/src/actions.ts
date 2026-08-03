import type { Page } from "playwright";
import { FIND_RED_SPAN, READ_AGGREGATOR_SNAPSHOT } from "./browser/evaluateScripts.js";
import { config } from "./config.js";
import { submitResultPredicate } from "./network.js";
import type { SyncContext } from "./synchronization.js";
import { clickAndWait } from "./synchronization.js";
import type { AggregatorSnapshot, EligibleElement, ResolutionAction } from "./types.js";
import { promptLine, stripOriginSuffix } from "./utils.js";

/**
 * Read eligible red conflicts from the live `difference` object in page JS.
 */
export async function readAggregatorSnapshot(page: Page): Promise<AggregatorSnapshot> {
  // String IIFE avoids tsx/esbuild injecting __name into page.evaluate payloads.
  return page.evaluate(READ_AGGREGATOR_SNAPSHOT) as Promise<AggregatorSnapshot>;
}

async function mousedownRedElement(page: Page, el: EligibleElement): Promise<void> {
  const termIdsJson = JSON.stringify(el.termIds);
  const handle = await page.evaluateHandle(FIND_RED_SPAN, { termIdsJson });

  const element = handle.asElement() as import("playwright").ElementHandle | null;
  if (!element) {
    // Fallback by visible text
    const suffix = ` (${el.origin})`;
    const locator = page.locator("#differenceDiv span", {
      hasText: el.label,
    });
    const count = await locator.count();
    let clicked = false;
    for (let i = 0; i < count; i++) {
      const candidate = locator.nth(i);
      const text = (await candidate.innerText()).trim();
      const style = (await candidate.getAttribute("style")) || "";
      if (!/color:\s*red/i.test(style)) continue;
      if (stripOriginSuffix(text) !== el.label) continue;
      if (!text.includes(suffix) && count > 1) continue;
      await candidate.dispatchEvent("mousedown");
      clicked = true;
      break;
    }
    if (!clicked) throw new Error(`Could not locate red element for ${el.label} (${el.origin}) path=${el.pathNames.join(" > ")}`);
    return;
  }

  await element.dispatchEvent("mousedown");
}

async function clickResolutionMenuItem(page: Page, action: ResolutionAction): Promise<void> {
  const li = page.locator(`#differenceResolutionMenuItems li[data-resolution="${action}"]`);
  await li.waitFor({ state: "visible", timeout: 10_000 });
  // Menu items use onmousedown handlers
  await li.dispatchEvent("mousedown");
}

async function clickJoinPartner(page: Page, partnerLabel: string): Promise<void> {
  const menu = page.locator("#additionalOptionsMenu");
  await menu.waitFor({ state: "visible", timeout: 10_000 });
  const li = menu.locator("li").filter({ hasText: partnerLabel }).first();
  await li.waitFor({ state: "visible", timeout: 10_000 });
  await li.dispatchEvent("mousedown");
}

export async function executeRemove(ctx: SyncContext, el: EligibleElement): Promise<void> {
  await clickAndWait(ctx, async () => {
    await mousedownRedElement(ctx.page, el);
    await clickResolutionMenuItem(ctx.page, "REMOVE_CONCEPT");
  });
}

export async function executeAdd(ctx: SyncContext, el: EligibleElement): Promise<void> {
  await clickAndWait(ctx, async () => {
    await mousedownRedElement(ctx.page, el);
    await clickResolutionMenuItem(ctx.page, "ADD_CONCEPT");
  });
}

export async function executeJoin(ctx: SyncContext, el: EligibleElement, partnerLabel: string): Promise<void> {
  await clickAndWait(ctx, async () => {
    await mousedownRedElement(ctx.page, el);
    await clickResolutionMenuItem(ctx.page, "JOIN_CONCEPTS");
    await clickJoinPartner(ctx.page, partnerLabel);
  });
}

export async function executeHumanDecision(
  ctx: SyncContext,
  el: EligibleElement,
  reason: string,
  options: { defaultAction?: "add" | "remove" | "abort" } = {},
): Promise<"abort" | void> {
  console.log("\n—— Human decision point ——");
  console.log(`Reason: ${reason}`);
  console.log(`Element: ${el.label} (${el.origin}) [${el.kind}]`);
  console.log(`Path: ${el.pathNames.join(" > ")}`);
  console.log("Browser is focused on this conflict; choose an action.");

  await mousedownRedElement(ctx.page, el);

  let answer: string;
  if (options.defaultAction) {
    answer = options.defaultAction;
    console.log(`Using --human-default=${answer}`);
  } else {
    answer = (await promptLine("Action? [add / remove / join / abort]: ")).toLowerCase();
  }

  if (answer === "abort") return "abort";

  if (answer === "add") {
    await clickAndWait(ctx, async () => {
      await clickResolutionMenuItem(ctx.page, "ADD_CONCEPT");
    });
    return;
  }
  if (answer === "remove") {
    await clickAndWait(ctx, async () => {
      await clickResolutionMenuItem(ctx.page, "REMOVE_CONCEPT");
    });
    return;
  }
  if (answer === "join") {
    const partner = await promptLine("Join partner label (exact menu text): ");
    await clickAndWait(ctx, async () => {
      await clickResolutionMenuItem(ctx.page, "JOIN_CONCEPTS");
      await clickJoinPartner(ctx.page, partner);
    });
    return;
  }

  throw new Error(`Unrecognized human action: ${answer}`);
}

export async function clickUpdateEvidenceAggregation(ctx: SyncContext): Promise<void> {
  const button = ctx.page.locator("#btnAggregateEvidence");
  await button.waitFor({ state: "visible", timeout: 30_000 });
  await Promise.all([
    ctx.page.waitForURL(/\/evidenceAggregator\/\d+/, { timeout: 120_000 }),
    button.click(),
  ]);
}

export async function clickViewAggregatedModel(ctx: SyncContext): Promise<void> {
  if (/\/evidenceEditor\/displayAggregationResult/.test(ctx.page.url())) {
    return;
  }

  const button = ctx.page.locator("#redirectButton");
  await button.waitFor({ state: "visible", timeout: 30_000 });

  // Prefer navigation as the success signal — submitResult redirects and destroys the
  // aggregator execution context, so post-click idle waits are unreliable here.
  const navigated = ctx.page.waitForURL(/\/evidenceEditor\/displayAggregationResult/, {
    timeout: config.requestTimeoutMs,
  });
  const submitted = ctx.page
    .waitForResponse(submitResultPredicate, { timeout: config.requestTimeoutMs })
    .catch(() => null);

  await button.click();
  const response = await submitted;
  if (response && !response.ok()) {
    throw new Error(`submitResult failed: HTTP ${response.status()}`);
  }
  await navigated;
}
