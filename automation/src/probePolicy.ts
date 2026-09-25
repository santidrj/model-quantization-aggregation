/**
 * Offline policy probe against a dumped AggregatorSnapshot JSON.
 * Usage:
 *   npx tsx src/probePolicy.ts /tmp/ef-snapshot.json
 *   npx tsx src/probePolicy.ts /tmp/ef-snapshot.json --target ptq-fp32-w8a8
 */
import { readFileSync } from "node:fs";
import { pickNextDecision } from "./aggregationRules.js";
import { createConfig, parseTargetFlag, parseTargetSlug, policyFromConfig } from "./config.js";
import { defaultSemanticMatcher } from "./semanticMatching.js";
import type { AggregatorSnapshot } from "./types.js";

const pathArg = process.argv.slice(2).find((a, i, arr) => {
  if (a.startsWith("-")) return false;
  if (i > 0 && arr[i - 1] === "--target") return false;
  return true;
});
const path = pathArg ?? "/tmp/ef-snapshot.json";
const snapshot = JSON.parse(readFileSync(path, "utf8")) as AggregatorSnapshot;
const matcher = defaultSemanticMatcher;
const appConfig = createConfig(parseTargetSlug(parseTargetFlag() ?? process.env.EF_TARGET));
const policy = policyFromConfig(appConfig);

console.log(
  `Target ${appConfig.targetSlug} | Pair ${snapshot.evidence1Id}↔${snapshot.evidence2Id} | red=${snapshot.eligible.length} | redirect=${snapshot.redirectVisible}`,
);
for (const e of snapshot.eligible) {
  console.log(`  [${e.treeIndex}] (${e.origin}) ${e.kind.padEnd(18)} | ${e.pathNames.join(" > ")}`);
}

const phases = ["remove", "join", "add", "residual"] as const;
for (const phase of phases) {
  const next = pickNextDecision(snapshot, matcher, phase, policy);
  if (!next) {
    console.log(`phase=${phase}: (none)`);
    continue;
  }
  console.log(
    `phase=${phase}: ${next.decision.type} → ${next.element.label} (${next.element.origin}) @ ${next.element.pathNames.join(" > ")}`,
    next.decision,
  );
}
