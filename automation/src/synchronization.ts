import type { Page, Response } from "playwright";
import { config } from "./config.js";
import {
  NetworkMonitor,
  aggregatorResponsePredicate,
  readInPagePending,
  submitResultPredicate,
} from "./network.js";
import { sleep } from "./utils.js";

export type SyncContext = {
  page: Page;
  monitor: NetworkMonitor;
};

/**
 * Wait until tracked aggregator traffic and in-page XHR/fetch counters are idle.
 * Never uses fixed "hope it finished" sleeps as the sole signal — only a short
 * settle window after observed idle to catch request chains (ADR-0001).
 */
export async function waitForServerIdle(
  ctx: SyncContext,
  options: { timeoutMs?: number; settleMs?: number } = {},
): Promise<void> {
  const timeoutMs = options.timeoutMs ?? config.requestTimeoutMs;
  const settleMs = options.settleMs ?? config.settleMs;
  const deadline = Date.now() + timeoutMs;
  let idleSince: number | null = null;

  while (Date.now() < deadline) {
    const inPageRaw = await readInPagePending(ctx.page);
    const inPage = typeof inPageRaw === "number" && Number.isFinite(inPageRaw) ? inPageRaw : 0;
    const pending = ctx.monitor.pendingCount + inPage;
    if (pending === 0) {
      if (idleSince == null) idleSince = Date.now();
      if (Date.now() - idleSince >= settleMs) {
        if (ctx.monitor.recentFailures.length > 0) {
          const failures = ctx.monitor.recentFailures.join("; ");
          ctx.monitor.clearFailures();
          throw new Error(`Server request failed while waiting for idle: ${failures}`);
        }
        return;
      }
    } else {
      idleSince = null;
    }
    await sleep(50);
  }

  throw new Error(
    `Timed out after ${timeoutMs}ms waiting for server idle (pending monitor=${ctx.monitor.pendingCount})`,
  );
}

/**
 * Perform an action that is expected to trigger POST /evidenceAggregator,
 * then wait for that response and full idle.
 */
export async function performAction(
  ctx: SyncContext,
  action: () => Promise<void>,
  options: { expectAggregatorPost?: boolean; expectSubmitResult?: boolean } = {},
): Promise<Response | null> {
  const expectAggregator = options.expectAggregatorPost ?? true;
  const expectSubmit = options.expectSubmitResult ?? false;

  await waitForServerIdle(ctx);

  let responsePromise: Promise<Response> | null = null;
  if (expectSubmit) {
    responsePromise = ctx.page.waitForResponse(submitResultPredicate, { timeout: config.requestTimeoutMs });
  } else if (expectAggregator) {
    responsePromise = ctx.page.waitForResponse(aggregatorResponsePredicate, {
      timeout: config.requestTimeoutMs,
    });
  }

  ctx.monitor.clearFailures();
  await action();

  let response: Response | null = null;
  if (responsePromise) {
    response = await responsePromise;
    if (!response.ok()) {
      throw new Error(`Aggregator request failed: HTTP ${response.status()} ${response.url()}`);
    }
  }

  await waitForServerIdle(ctx);
  return response;
}

export async function clickAndWait(
  ctx: SyncContext,
  click: () => Promise<void>,
  options?: { expectAggregatorPost?: boolean; expectSubmitResult?: boolean },
): Promise<Response | null> {
  return performAction(ctx, click, options);
}
