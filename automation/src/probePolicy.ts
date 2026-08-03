/**
 * Offline policy probe against a dumped AggregatorSnapshot JSON.
 * Usage: npx tsx src/probePolicy.ts /tmp/ef-snapshot.json
 */
import { readFileSync } from "node:fs";
import { pickNextDecision } from "./aggregationRules.js";
import { defaultSemanticMatcher } from "./semanticMatching.js";
import type { AggregatorSnapshot } from "./types.js";

const path = process.argv[2] ?? "/tmp/ef-snapshot.json";
const snapshot = JSON.parse(readFileSync(path, "utf8")) as AggregatorSnapshot;
const matcher = defaultSemanticMatcher;

console.log(
  `Pair ${snapshot.evidence1Id}↔${snapshot.evidence2Id} | red=${snapshot.eligible.length} | redirect=${snapshot.redirectVisible}`,
);
for (const e of snapshot.eligible) {
  console.log(`  [${e.treeIndex}] (${e.origin}) ${e.kind.padEnd(18)} | ${e.pathNames.join(" > ")}`);
}

const phases = ["remove", "join", "add", "residual"] as const;
for (const phase of phases) {
  const next = pickNextDecision(snapshot, matcher, phase);
  if (!next) {
    console.log(`phase=${phase}: (none)`);
    continue;
  }
  console.log(
    `phase=${phase}: ${next.decision.type} → ${next.element.label} (${next.element.origin}) @ ${next.element.pathNames.join(" > ")}`,
    next.decision,
  );
}
