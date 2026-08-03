import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser, type BrowserContext, type Page } from "playwright";
import { config } from "./config.js";
import { installNetworkIdleHooks } from "./network.js";
import { isLoginUrl, promptLine, sleep } from "./utils.js";

function storageStateFilePath(): string {
  return fileURLToPath(config.storageStatePath);
}

export async function ensureAuthDir(): Promise<void> {
  await mkdir(dirname(storageStateFilePath()), { recursive: true });
}

export async function launchAuthenticatedContext(): Promise<{
  browser: Browser;
  context: BrowserContext;
  page: Page;
}> {
  await ensureAuthDir();
  const browser = await chromium.launch({ headless: false });
  const storagePath = storageStateFilePath();

  let context: BrowserContext;
  try {
    context = await browser.newContext({ storageState: storagePath });
  } catch {
    context = await browser.newContext();
  }

  const page = await context.newPage();
  await installNetworkIdleHooks(page);
  return { browser, context, page };
}

export async function saveStorageState(context: BrowserContext): Promise<void> {
  await ensureAuthDir();
  await context.storageState({ path: storageStateFilePath() });
  console.log(`Saved authenticated storage state to ${storageStateFilePath()}`);
}

/** True when the chrome shows Login (session missing/expired). */
export async function isAuthenticationRequired(page: Page): Promise<boolean> {
  if (isLoginUrl(page.url())) return true;
  const logout = page.locator('a[href*="logout"], a:has-text("(logout)")');
  if ((await logout.count()) > 0) return false;
  const loginLink = page.locator('a[href*="/user/login"]');
  return (await loginLink.count()) > 0;
}

/**
 * Headed manual login. Does not automate credentials.
 */
export async function interactiveLogin(page: Page, context: BrowserContext): Promise<void> {
  console.log("Authentication required. Opening login page…");
  console.log("Log in manually in the browser window, then return here.");
  await page.goto(config.loginUrl, { waitUntil: "domcontentloaded" });
  await promptLine("Press Enter after login has succeeded… ");

  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (!(await isAuthenticationRequired(page))) break;
    await sleep(250);
  }
  if (await isAuthenticationRequired(page)) {
    throw new Error("Still appears logged out after Enter. Aborting without saving storage state.");
  }

  await saveStorageState(context);
}

export async function ensureAuthenticated(page: Page, context: BrowserContext): Promise<void> {
  if (await isAuthenticationRequired(page)) {
    await interactiveLogin(page, context);
  }
}
