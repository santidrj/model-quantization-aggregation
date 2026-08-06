import type { SemanticMatcher } from "./semanticMatching.js";
import type { AggregatorSnapshot, EligibleElement, PolicyDecision } from "./types.js";
import { normalizeLabel } from "./utils.js";

const MODEL_QUANTIZATION = normalizeLabel("Model quantization");
const SYSTEM = normalizeLabel("System");
const DL_MODEL = normalizeLabel("DL model");

/** Per-named-aggregation-group policy toggles. */
export type AggregationPolicy = {
  /** When true (PTQ from FP32 to w-int8, a-int8), keep Model quantization cause aspects via Add. */
  keepModelQuantizationAspects: boolean;
};

export const DEFAULT_AGGREGATION_POLICY: AggregationPolicy = {
  keepModelQuantizationAspects: false,
};

function pathHasModelQuantization(el: EligibleElement): boolean {
  return el.pathNames.some((n) => normalizeLabel(n) === MODEL_QUANTIZATION);
}

function isSystemFirstLevelAspect(el: EligibleElement): boolean {
  if (el.kind !== "contextual aspect") return false;
  if (el.pathNames.length < 2) return false;
  if (normalizeLabel(el.pathNames[0]!) !== SYSTEM) return false;
  // First-level under System: System > Aspect
  return el.pathNames.length === 2;
}

function isProtectedDlModel(el: EligibleElement): boolean {
  return isSystemFirstLevelAspect(el) && normalizeLabel(el.label) === DL_MODEL;
}

/** Contextual aspects under Model quantization that PTQ keeps via Add. */
export function isModelQuantizationAspect(el: EligibleElement): boolean {
  return el.kind === "contextual aspect" && pathHasModelQuantization(el);
}

/** Mandatory cleanup Removes. */
export function shouldAutoRemove(
  el: EligibleElement,
  policy: AggregationPolicy = DEFAULT_AGGREGATION_POLICY,
): boolean {
  if (el.kind !== "contextual aspect") return false;
  if (isProtectedDlModel(el)) return false;
  if (pathHasModelQuantization(el)) {
    if (policy.keepModelQuantizationAspects) return false;
    return true;
  }
  if (isSystemFirstLevelAspect(el)) return true;
  return false;
}

/**
 * Find a Join partner among eligible elements: opposite origin, same kind,
 * labels linked by the semantic map (alias ↔ canonical).
 */
export function findJoinPartner(
  el: EligibleElement,
  eligible: EligibleElement[],
  matcher: SemanticMatcher,
): EligibleElement | null {
  const mapped = matcher.mappedCanonical(el.label);
  if (!mapped) return null;

  for (const other of eligible) {
    if (other.treeIndex === el.treeIndex) continue;
    if (other.kind !== el.kind) continue;
    if (other.origin === el.origin) continue;
    if (!matcher.areEquivalent(el.label, other.label)) continue;
    // Prefer partner whose label is the canonical term when possible
    return other;
  }
  return null;
}

export function joinPartnerLabel(el: EligibleElement, partner: EligibleElement, matcher: SemanticMatcher): string {
  const canon = matcher.canonicalOf(el.label);
  // additionalInformation must be the other concept's display name as shown in the UI
  if (normalizeLabel(partner.label) === normalizeLabel(canon)) return partner.label;
  if (normalizeLabel(el.label) === normalizeLabel(canon)) return partner.label;
  // Default: pass partner's rendered label (EvidenceFactory expects the other term name)
  return partner.label;
}

/**
 * Decide the next automatic or human action for one eligible element,
 * given the full snapshot (for Join pairs and Add duplicate checks).
 */
export function decideForElement(
  el: EligibleElement,
  snapshot: AggregatorSnapshot,
  matcher: SemanticMatcher,
  phase: "remove" | "join" | "add" | "residual",
  policy: AggregationPolicy = DEFAULT_AGGREGATION_POLICY,
): PolicyDecision | null {
  if (phase === "remove") {
    if (!shouldAutoRemove(el, policy)) return null;
    // Prefer Join when a legal map partner is also eligible (e.g. LLM ↔ DL model).
    if (findJoinPartner(el, snapshot.eligible, matcher)) return null;
    return { type: "remove" };
  }

  if (phase === "join") {
    const partner = findJoinPartner(el, snapshot.eligible, matcher);
    if (!partner) return null;
    // Only act from the alias side when labels differ, so Join keeps canonical via partner name.
    const canon = matcher.canonicalOf(el.label);
    if (normalizeLabel(el.label) === normalizeLabel(canon) && normalizeLabel(partner.label) !== normalizeLabel(canon)) {
      // el is already canonical — let the alias side drive the Join
      return null;
    }
    return { type: "join", partnerLabel: joinPartnerLabel(el, partner, matcher) };
  }

  if (phase === "add") {
    if (findJoinPartner(el, snapshot.eligible, matcher)) return null;

    // PTQ from FP32 to w-int8, a-int8: keep Model quantization cause contextual aspects via Add.
    if (policy.keepModelQuantizationAspects && isModelQuantizationAspect(el)) {
      return { type: "add" };
    }

    if (el.kind !== "effect") return null;
    if (shouldAutoRemove(el, policy)) return null;

    const orphanAlias =
      matcher.isAlias(el.label) &&
      !snapshot.eligible.some(
        (o) =>
          o.treeIndex !== el.treeIndex &&
          o.kind === el.kind &&
          matcher.areEquivalent(el.label, o.label),
      ) &&
      snapshot.matchedEffectLabels.some((m) => matcher.areEquivalent(el.label, m));

    if (orphanAlias) {
      return { type: "human", reason: `Orphan alias Effect "${el.label}" — canonical already present; needs human decision` };
    }

    const duplicate = snapshot.matchedEffectLabels.some((m) => matcher.areEquivalent(el.label, m));
    if (duplicate) {
      return {
        type: "human",
        reason: `Effect "${el.label}" looks equivalent to an already-matched Effect; needs human decision`,
      };
    }

    // Auto-Add either origin when no equivalent exists in the matched model
    return { type: "add" };
  }

  // residual: everything left is human
  return { type: "human", reason: `Residual eligible ${el.kind} "${el.label}" (${el.origin})` };
}

/**
 * Pick the next work item in tree order for the current phase.
 */
export function pickNextDecision(
  snapshot: AggregatorSnapshot,
  matcher: SemanticMatcher,
  phase: "remove" | "join" | "add" | "residual",
  policy: AggregationPolicy = DEFAULT_AGGREGATION_POLICY,
): { element: EligibleElement; decision: PolicyDecision } | null {
  const ordered = [...snapshot.eligible].sort((a, b) => a.treeIndex - b.treeIndex);
  for (const el of ordered) {
    const decision = decideForElement(el, snapshot, matcher, phase, policy);
    if (decision) return { element: el, decision };
  }
  return null;
}
