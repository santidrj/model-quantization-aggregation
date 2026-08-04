/**
 * Runtime configuration for the EvidenceFactory aggregation automation.
 * Keep secrets out of this file — auth lives in .auth/storage-state.json (gitignored).
 *
 * Override the aggregation target without editing this file:
 *   EF_AGGREGATION_GROUP_ID=282042 EF_SYNTHESIS_ID=244422 npm start
 */
const synthesisId = Number(process.env.EF_SYNTHESIS_ID ?? 244422);
const aggregationGroupId = Number(process.env.EF_AGGREGATION_GROUP_ID ?? 282042);
const baseUrl = "https://evidencefactory.lens-ese.cos.ufrj.br";

export const config = {
  baseUrl,
  /** Aggregation-group overview. */
  overviewUrl: `${baseUrl}/evidenceAggregation/synthesisAggregation/${synthesisId}?selectedAggregationId=${aggregationGroupId}`,
  /** Matching editor reached after Update. */
  aggregatorUrl: `${baseUrl}/evidenceAggregator/${aggregationGroupId}`,
  aggregationGroupId,
  synthesisId,
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

export type AppConfig = typeof config;
