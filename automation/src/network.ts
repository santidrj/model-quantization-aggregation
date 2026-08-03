import type { Page, Request, Response } from "playwright";
import { fileURLToPath } from "node:url";
import { READ_IN_PAGE_PENDING } from "./browser/evaluateScripts.js";

const TRACKED_URL_RE = /\/evidenceAggregator(\/submitResult)?\/?(\?|$)/;
const IDLE_HOOKS_PATH = fileURLToPath(new URL("./browser/idleHooks.js", import.meta.url));

export type PendingRequest = {
  id: number;
  url: string;
  method: string;
  startedAt: number;
};

/**
 * Tracks Playwright-visible network activity for EvidenceFactory aggregator calls.
 * Complements in-page instrumentation (see installNetworkIdleHooks).
 */
export class NetworkMonitor {
  private pending = new Map<number, PendingRequest>();
  private nextId = 1;
  private failures: string[] = [];

  constructor(private readonly page: Page) {
    this.page.on("request", this.onRequest);
    this.page.on("requestfinished", this.onFinished);
    this.page.on("requestfailed", this.onFailed);
    this.page.on("response", this.onResponse);
  }

  dispose(): void {
    this.page.off("request", this.onRequest);
    this.page.off("requestfinished", this.onFinished);
    this.page.off("requestfailed", this.onFailed);
    this.page.off("response", this.onResponse);
  }

  get pendingCount(): number {
    return this.pending.size;
  }

  get recentFailures(): string[] {
    return [...this.failures];
  }

  clearFailures(): void {
    this.failures = [];
  }

  isTracked(url: string, method?: string): boolean {
    if (!TRACKED_URL_RE.test(url)) return false;
    if (method && method.toUpperCase() !== "POST") return false;
    return true;
  }

  private onRequest = (request: Request): void => {
    if (!this.isTracked(request.url(), request.method())) return;
    const id = this.nextId++;
    (request as Request & { __monitorId?: number }).__monitorId = id;
    this.pending.set(id, {
      id,
      url: request.url(),
      method: request.method(),
      startedAt: Date.now(),
    });
  };

  private onFinished = (request: Request): void => {
    const id = (request as Request & { __monitorId?: number }).__monitorId;
    if (id != null) this.pending.delete(id);
  };

  private onFailed = (request: Request): void => {
    const id = (request as Request & { __monitorId?: number }).__monitorId;
    if (id != null) this.pending.delete(id);
    if (this.isTracked(request.url(), request.method())) {
      this.failures.push(`${request.method()} ${request.url()} failed: ${request.failure()?.errorText ?? "unknown"}`);
    }
  };

  private onResponse = (response: Response): void => {
    if (!this.isTracked(response.url(), response.request().method())) return;
    if (response.status() >= 400) {
      this.failures.push(`${response.request().method()} ${response.url()} → HTTP ${response.status()}`);
    }
  };
}

/**
 * Inject hooks so the page exposes pending fetch/XHR counts (ADR-0001).
 * Safe to call on every navigation via addInitScript.
 */
export async function installNetworkIdleHooks(page: Page): Promise<void> {
  await page.addInitScript({ path: IDLE_HOOKS_PATH });
}

export async function readInPagePending(page: Page): Promise<number> {
  return page.evaluate(READ_IN_PAGE_PENDING) as Promise<number>;
}

export function aggregatorResponsePredicate(response: Response): boolean {
  return (
    response.request().method() === "POST" &&
    TRACKED_URL_RE.test(response.url()) &&
    !response.url().includes("submitResult")
  );
}

export function submitResultPredicate(response: Response): boolean {
  return response.request().method() === "POST" && /\/evidenceAggregator\/submitResult/.test(response.url());
}

export { TRACKED_URL_RE };
