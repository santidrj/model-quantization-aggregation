# EvidenceFactory aggregation automation

Playwright + TypeScript automation for EvidenceFactory evidence aggregation (Add / Remove / Join).

Default target is **Full aggregation v2** (`full-v2` / `selectedAggregationId=282042`). Select another named aggregation group with `--target` or `EF_TARGET`.

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

Flow: overview → **Update evidence aggregation** (or **Aggregate evidence** on first run) → Remove → Join → Add (with human prompts for residuals) → **view aggregated model**.

### Target selection

Named aggregation groups (slug → group):

| Slug | Group | ID |
| --- | --- | --- |
| `full-v2` (default) | Full aggregation v2 | 282042 |
| `sensitivity-n6` | Sensitivity analysis (n&lt;6) | 327146 |
| `ptq-fp32-w8a8` | PTQ from FP32 to w-int8, a-int8 | 281366 |
| `test` | Test aggregation (sandbox) | 304080 |

```bash
# Full aggregation v2 (default)
npm start

# Sensitivity analysis
npm start -- --target sensitivity-n6

# PTQ subgroup (keeps Model-quantization contextual aspects via Add)
EF_TARGET=ptq-fp32-w8a8 npm start

# Disposable Test group
npm start -- --target test

# Anonymous Test-only shortcut
npm run start:anon
```

`--human-default add|remove|abort` answers residual human decision points without a terminal prompt. Set `EF_CAPTURE_RESULT=1` to dump the result-page tree text and a screenshot to `/tmp/ef-golden-result.png` after submit.

## Config

- Named targets and semantic map: `src/config.ts` (`EF_TARGET`, `EF_SYNTHESIS_ID`)
- Domain language: `CONTEXT.md`
- Phase 1 network notes: `docs/phase1-network-analysis.md`
