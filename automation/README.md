# EvidenceFactory aggregation automation

Playwright + TypeScript automation for EvidenceFactory evidence aggregation (Add / Remove / Join).

Default target is the **final** aggregation group (`selectedAggregationId=282042`). Override with env vars for the disposable Test group or other IDs.

## Setup

```bash
cd automation
npm install
npx playwright install chromium
```

## Auth (manual login, no credentials in code)

```bash
npm run auth
```

Log in in the headed browser (session is detected automatically, or press Enter). Session is stored in `.auth/storage-state.json` (gitignored).

## Run aggregation (logged in)

```bash
npm start
```

Flow: overview → **Update evidence aggregation** → Remove → Join → Add (with human prompts for residuals) → **view aggregated model**.

### Target selection

```bash
# Final aggregation (default)
npm start

# Disposable Test group
EF_AGGREGATION_GROUP_ID=304080 npm start

# Anonymous Test-only shortcut (not for the final aggregation)
EF_AGGREGATION_GROUP_ID=304080 npm run start:anon
```

`--human-default add|remove|abort` answers residual human decision points without a terminal prompt. Set `EF_CAPTURE_RESULT=1` to dump the result-page tree text and a screenshot to `/tmp/ef-golden-result.png` after submit.

## Config

- URLs and semantic map: `src/config.ts` (`EF_AGGREGATION_GROUP_ID`, `EF_SYNTHESIS_ID`)
- Domain language: `CONTEXT.md`
- Phase 1 network notes: `docs/phase1-network-analysis.md`
