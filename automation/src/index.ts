import { ensureAuthenticated, interactiveLogin, launchAuthenticatedContext, saveStorageState } from "./auth.js";
import { runAggregation } from "./aggregation.js";
import { config } from "./config.js";
import { sleep } from "./utils.js";

function parseHumanDefault(): "add" | "remove" | "abort" | undefined {
  const idx = process.argv.indexOf("--human-default");
  if (idx >= 0) {
    const value = process.argv[idx + 1]?.toLowerCase();
    if (value === "add" || value === "remove" || value === "abort") return value;
    throw new Error(`Invalid --human-default value: ${value ?? "(missing)"}`);
  }
  const eq = process.argv.find((a) => a.startsWith("--human-default="));
  if (eq) {
    const value = eq.split("=", 2)[1]?.toLowerCase();
    if (value === "add" || value === "remove" || value === "abort") return value;
    throw new Error(`Invalid --human-default value: ${value ?? "(missing)"}`);
  }
  return undefined;
}

async function main(): Promise<void> {
  const loginOnly = process.argv.includes("--login-only");
  const allowAnonymous = process.argv.includes("--allow-anonymous");
  const humanDefault = parseHumanDefault();
  const { browser, context, page } = await launchAuthenticatedContext();

  try {
    if (loginOnly) {
      await page.goto(config.loginUrl, { waitUntil: "domcontentloaded" });
      await interactiveLogin(page, context);
      console.log("Login storage state saved. You can run npm start next.");
      return;
    }

    await page.goto(config.baseUrl, { waitUntil: "domcontentloaded" });
    if (!allowAnonymous) {
      await ensureAuthenticated(page, context);
      await saveStorageState(context);
    } else {
      console.log("Running with --allow-anonymous (skipping interactive login).");
    }
    if (humanDefault) {
      console.log(`Human decision default: ${humanDefault}`);
    }

    await runAggregation(page, context, { allowAnonymous, humanDefault });
    if (!allowAnonymous) {
      await saveStorageState(context);
    }
    console.log("Done.");
  } finally {
    console.log("Closing browser in 2s…");
    await sleep(2000);
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
