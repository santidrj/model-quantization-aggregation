export type ElementKind = "archetype" | "cause" | "contextual aspect" | "effect" | string;

export type ModelOrigin = 1 | 2;

export type ResolutionAction = "ADD_CONCEPT" | "REMOVE_CONCEPT" | "JOIN_CONCEPTS";

export type PolicyDecision =
  | { type: "remove" }
  | { type: "join"; partnerLabel: string }
  | { type: "add" }
  | { type: "human"; reason: string };

/** One red (actionable) conflict in the matching editor. */
export interface EligibleElement {
  /** Term-id path used by showDifferenceResolutionMenu. */
  termIds: number[];
  /** Human-readable ancestor path including this node. */
  pathNames: string[];
  label: string;
  kind: ElementKind;
  origin: ModelOrigin;
  /** Depth in the tree (root = 1). */
  depth: number;
  /** Document order among red nodes. */
  treeIndex: number;
}

export interface AggregatorSnapshot {
  evidence1Id: number | null;
  evidence2Id: number | null;
  numberOfDifferentNodes: number | null;
  redirectVisible: boolean;
  eligible: EligibleElement[];
  /** Black/matching effect labels currently visible (for duplicate detection). */
  matchedEffectLabels: string[];
}
