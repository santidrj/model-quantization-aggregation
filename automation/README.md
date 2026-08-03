# EvidenceFactory aggregation automation

Playwright + TypeScript automation for the disposable **Test** aggregation group (`selectedAggregationId=304080`).

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

Log in in the headed browser, press Enter in the terminal. Session is stored in `.auth/storage-state.json` (gitignored).

## Run aggregation

```bash
npm start
```

Flow: overview → **Update evidence aggregation** → Remove → Join → Add (with human prompts for residuals) → **view aggregated model**.

For the disposable Test group without logging in (mutations work anonymously there):

```bash
npm run start:anon
# equivalent: npx tsx src/index.ts --allow-anonymous --human-default add
```

`--human-default add|remove|abort` answers residual human decision points without a terminal prompt (useful for unattended runs). Set `EF_CAPTURE_RESULT=1` to dump the result-page tree text and a screenshot to `/tmp/ef-golden-result.png` after submit.

## Config

- URLs and semantic map: `src/config.ts`
- Domain language: `CONTEXT.md`
- Phase 1 network notes: `docs/phase1-network-analysis.md`
