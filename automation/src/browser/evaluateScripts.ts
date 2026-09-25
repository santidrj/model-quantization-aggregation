/**
 * Browser-side helpers as string sources for page.evaluate.
 * Kept as plain JS text so tsx/esbuild keepNames (__name) cannot break Playwright serialization.
 *
 * No-arg scripts must be IIFEs (Playwright evaluates the string as an expression; a bare
 * `() => …` returns a non-serializable function → undefined). Arg-taking scripts stay as
 * function expressions — Playwright invokes them when an argument is passed.
 */

/** @returns {string} IIFE expression → number */
export const READ_IN_PAGE_PENDING = `(() => {
  const w = window;
  const hooked = w.__efPending ?? 0;
  const jq = typeof w.jQuery?.active === "number" ? w.jQuery.active : 0;
  return Math.max(hooked, jq);
})()`;

/**
 * Reads AggregatorSnapshot from live page state.
 * @returns {string} IIFE expression → AggregatorSnapshot
 */
export const READ_AGGREGATOR_SNAPSHOT = `(() => {
  const eligible = [];
  const matchedEffectLabels = [];
  let treeIndex = 0;

  const collectMatchedEffects = (nodes) => {
    if (!nodes) return;
    for (const n of nodes) {
      if (n.typeName === "effect") matchedEffectLabels.push(n.term.name);
      collectMatchedEffects(n.matchingNextNodes);
    }
  };

  const walk = (nodes, pathNames, pathIds, mode) => {
    if (!nodes) return;
    for (const n of nodes) {
      const names = pathNames.concat([n.term.name]);
      const ids = pathIds.concat([n.term.id]);
      if (mode === "different") {
        const origin = n.presentOnEvidence1 ? 1 : 2;
        eligible.push({
          termIds: ids,
          pathNames: names,
          label: n.term.name,
          kind: n.typeName,
          origin: origin,
          depth: ids.length,
          treeIndex: treeIndex++,
        });
      }
      if (mode === "matching") {
        walk(n.matchingNextNodes, names, ids, "matching");
        walk(n.differentNextNodes, names, ids, "different");
      } else {
        walk(n.matchingNextNodes, names, ids, "undefined");
        walk(n.differentNextNodes, names, ids, "undefined");
      }
    }
  };

  const difference = window.difference;
  if (difference) {
    collectMatchedEffects(difference.matchingStartingNodes);
    walk(difference.matchingStartingNodes, [], [], "matching");
    walk(difference.differentStartingNodes, [], [], "different");
  }

  const redirect = document.getElementById("redirectButton");
  const redirectVisible = !!(redirect && getComputedStyle(redirect).display !== "none");

  return {
    evidence1Id: difference?.evidence1Id ?? window.currentEvidence1Id ?? null,
    evidence2Id: difference?.evidence2Id ?? window.currentEvidence2Id ?? null,
    numberOfDifferentNodes: eligible.length,
    redirectVisible,
    eligible,
    matchedEffectLabels,
  };
})()`;

/**
 * Find red span for termIds JSON string.
 * @returns {string} function expression: ({ termIdsJson }) => Element | null
 */
export const FIND_RED_SPAN = `({ termIdsJson }) => {
  const spans = [...document.querySelectorAll("#differenceDiv span")];
  for (const span of spans) {
    const attr = span.getAttribute("onmousedown") || "";
    if (attr.includes(termIdsJson) && /color:\\s*red/i.test(span.getAttribute("style") || "")) {
      return span;
    }
  }
  return null;
}`;
