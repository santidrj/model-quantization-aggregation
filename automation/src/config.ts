/**
 * Runtime configuration for the EvidenceFactory aggregation automation.
 * Keep secrets out of this file — auth lives in .auth/storage-state.json (gitignored).
 *
 * Select a named aggregation group:
 *   EF_TARGET=full-v2 npm start
 *   npm start -- --target sensitivity-n6
 */
const baseUrl = "https://evidencefactory.lens-ese.cos.ufrj.br";

/** Named aggregation groups selectable by slug. */
export const NAMED_AGGREGATION_GROUPS = {
  "full-v2": {
    displayName: "Full aggregation v2",
    aggregationGroupId: 282042,
    keepModelQuantizationAspects: false,
  },
  "sensitivity-n6": {
    displayName: "Sensitivity analysis (n<6)",
    aggregationGroupId: 327146,
    keepModelQuantizationAspects: false,
  },
  "ptq-fp32-w8a8": {
    displayName: "PTQ from FP32 to w-int8, a-int8",
    aggregationGroupId: 281366,
    keepModelQuantizationAspects: true,
  },
  test: {
    displayName: "Test aggregation",
    aggregationGroupId: 304080,
    keepModelQuantizationAspects: false,
  },
} as const;

export type AggregationGroupSlug = keyof typeof NAMED_AGGREGATION_GROUPS;

export const DEFAULT_AGGREGATION_GROUP_SLUG: AggregationGroupSlug = "full-v2";

export const AGGREGATION_GROUP_SLUGS = Object.keys(NAMED_AGGREGATION_GROUPS) as AggregationGroupSlug[];

export function isAggregationGroupSlug(value: string): value is AggregationGroupSlug {
  return Object.prototype.hasOwnProperty.call(NAMED_AGGREGATION_GROUPS, value);
}

export function parseTargetSlug(raw: string | undefined | null): AggregationGroupSlug {
  if (raw == null || raw.trim() === "") return DEFAULT_AGGREGATION_GROUP_SLUG;
  const slug = raw.trim();
  if (!isAggregationGroupSlug(slug)) {
    throw new Error(
      `Unknown aggregation target "${slug}". Expected one of: ${AGGREGATION_GROUP_SLUGS.join(", ")}`,
    );
  }
  return slug;
}

const shared = {
  baseUrl,
  loginUrl: `${baseUrl}/user/login`,
  storageStatePath: new URL("../.auth/storage-state.json", import.meta.url),
  /** Alias → canonical (more generic) term kept on Join. */
  semanticEquivalence: {
    LLM: "DL model",
    "Large language model": "DL model",
    "Large Language Model": "DL model",
    "Clock cycle": "Inference latency",
    "Clock cycles": "Inference latency",
    "Clock Cycle": "Inference latency",
    Accuracy: "Classification accuracy",
    accuracy: "Classification accuracy",
  } as Record<string, string>,
  requestTimeoutMs: 120_000,
  settleMs: 150,
  maxActionRetries: 3,
} as const;

export type AppConfig = typeof shared & {
  synthesisId: number;
  targetSlug: AggregationGroupSlug;
  targetDisplayName: string;
  aggregationGroupId: number;
  /** PTQ-only: auto-Add contextual aspects under Model quantization instead of Remove. */
  keepModelQuantizationAspects: boolean;
  overviewUrl: string;
  aggregatorUrl: string;
};

export function createConfig(slug: AggregationGroupSlug = DEFAULT_AGGREGATION_GROUP_SLUG): AppConfig {
  const synthesisId = Number(process.env.EF_SYNTHESIS_ID ?? 244422);
  const group = NAMED_AGGREGATION_GROUPS[slug];
  return {
    ...shared,
    synthesisId,
    targetSlug: slug,
    targetDisplayName: group.displayName,
    aggregationGroupId: group.aggregationGroupId,
    keepModelQuantizationAspects: group.keepModelQuantizationAspects,
    overviewUrl: `${baseUrl}/evidenceAggregation/synthesisAggregation/${synthesisId}?selectedAggregationId=${group.aggregationGroupId}`,
    aggregatorUrl: `${baseUrl}/evidenceAggregator/${group.aggregationGroupId}`,
  };
}

/** Default config (Full aggregation v2, or `EF_TARGET` when set at process start). */
export const config = createConfig(parseTargetSlug(process.env.EF_TARGET));
