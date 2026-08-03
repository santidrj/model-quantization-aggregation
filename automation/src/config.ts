/**
 * Runtime configuration for the EvidenceFactory aggregation automation.
 * Keep secrets out of this file — auth lives in .auth/storage-state.json (gitignored).
 */
export const config = {
  baseUrl: "https://evidencefactory.lens-ese.cos.ufrj.br",
  /** Aggregation-group overview (Test group). */
  overviewUrl:
    "https://evidencefactory.lens-ese.cos.ufrj.br/evidenceAggregation/synthesisAggregation/244422?selectedAggregationId=304080",
  /** Matching editor reached after Update. */
  aggregatorUrl: "https://evidencefactory.lens-ese.cos.ufrj.br/evidenceAggregator/304080",
  aggregationGroupId: 304080,
  loginUrl: "https://evidencefactory.lens-ese.cos.ufrj.br/user/login",
  storageStatePath: new URL("../.auth/storage-state.json", import.meta.url),
  /** Alias → canonical (more generic) term kept on Join. */
  semanticEquivalence: {
    LLM: "DL model",
    "Large language model": "DL model",
    "Large Language Model": "DL model",
    "Clock cycle": "Latency",
    "Clock cycles": "Latency",
    "Clock Cycle": "Latency",
  } as Record<string, string>,
  requestTimeoutMs: 120_000,
  settleMs: 150,
  maxActionRetries: 3,
} as const;

export type AppConfig = typeof config;
