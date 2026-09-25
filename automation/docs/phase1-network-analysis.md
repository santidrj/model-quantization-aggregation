# Phase 1 — EvidenceFactory communication analysis

Target: Test aggregation group `304080` (safe disposable aggregation).

## Screens and entry flow

| Step | URL / control | What happens |
|------|----------------|--------------|
| Overview | `/evidenceAggregation/synthesisAggregation/244422?selectedAggregationId=304080` | Lists evidence in the Test group; shows **Update evidence aggreagation** (`#btnAggregateEvidence`) |
| Start matching | Link navigates to `/evidenceAggregator/304080` | Full page navigation (not an in-place AJAX refresh) |
| Matching editor | `/evidenceAggregator/304080` | Theoretical structure matching; red/gray/black tree |
| Finish | `#redirectButton` **view aggregated model** | POST submit, then redirect to `/evidenceEditor/displayAggregationResult` |

The overview and matching editor are **different routes**. The shared query URL opens the overview; Update switches to the aggregator route.

## Transport stack

| Mechanism | Used for app workflow? | Evidence |
|-----------|------------------------|----------|
| **jQuery `$.ajax` → XMLHttpRequest** | **Yes — primary** | Inline aggregator script; `performance` shows `initiatorType: "xmlhttprequest"` for `/evidenceAggregator` |
| `fetch()` | Not used by app logic (not observed) | No fetch calls in aggregator/overview scripts |
| REST-ish JSON over POST | **Yes** | JSON request/response bodies, `Accept`/`Content-Type: application/json` |
| GraphQL | No | No GraphQL endpoints or payloads |
| WebSockets | No | No WS resources; constructors exist in browser only |
| Server-Sent Events | No | Not used by app scripts |

Framework: server-rendered HTML + **jQuery 2.2.4** + Bootstrap. Not React/Angular/Vue.

## Core API calls

### 1. Load / apply resolutions — `POST /evidenceAggregator`

Fired on page ready and after **every** Add / Remove / Join.

Request body shape:

```json
{
  "evidenceAggregationGroupId": 304080,
  "evidenceIds": [319396, 320740, 263314, 319452, 248092],
  "instructions": [
    { "resolutions": [ /* cumulative resolution steps */ ] }
  ]
}
```

Each resolution:

```json
{
  "termIds": [155, 243297, 319368],
  "actionName": "ADD_CONCEPT | REMOVE_CONCEPT | JOIN_CONCEPTS",
  "additionalInformation": null
}
```

For **Join**, `additionalInformation` is the **partner concept’s display name** (chosen from the secondary menu).

Response (observed fields):

- `evidenceAggregationGroupId`
- `difference` with `evidence1Id`, `evidence2Id`, `matchingStartingNodes`, `differentStartingNodes`
- `numberOfDifferentNodes` (when `0`, show **view aggregated model**)

Nodes carry `term.id`, `term.name`, `typeName` (`archetype` | `cause` | `contextual aspect` | `effect`), `presentOnEvidence1` / `presentOnEvidence2`, and child lists `matchingNextNodes` / `differentNextNodes`.

### 2. Finalize — `POST /evidenceAggregator/submitResult`

Same JSON envelope as above. Success navigates to `/evidenceEditor/displayAggregationResult`. Button label while waiting: `processing aggregation ...`.

### 3. Overview-only AJAX (group management)

Not part of the matching turn, but present on the overview page:

- `POST /evidenceAggregation/{aggregationId}/{add|remove}/evidence/{evidenceId}`
- `POST /evidenceAggregation/add/synthesisAggregation/244422`
- `POST /evidenceAggregation/{groupId}/remove/`

Tab switches are full navigations back to the overview URL with `?selectedAggregationId=…`.

## UI semantics (from client script + live DOM)

| Color | Meaning | Actionable? |
|-------|---------|-------------|
| Red | Conflict (`different` node under a matching parent, or different starting node) | Yes — `onmousedown` opens resolution menu |
| Gray | Unanalyzed child under an unresolved difference | No |
| Black | Matching / resolved | No |

Origin suffixes `(1)` / `(2)` come from `presentOnEvidence1` / else `(2)`.

Kind badges are `node.typeName` in a Bootstrap `badge`.

### Menu rules encoded in the client

Action names in the DOM: `ADD_CONCEPT`, `REMOVE_CONCEPT`, `JOIN_CONCEPTS`.

- **Cause**: Add and Remove hidden; Join only if a same-kind sibling exists on the other evidence.
- **Root archetype**: Join hidden; Add allowed; Remove mostly constrained.
- **Join**: only if same-level, opposite-origin, same `typeName` siblings exist; then a second menu lists partner names.
- Conflicts are resolved **one level at a time** (gray children become actionable only after parents are resolved).
- Left sidebar highlight auto-updates when the server advances `evidence1Id` / `evidence2Id` after a pair is cleared.

## How completion / idle is indicated

1. **Primary:** `jQuery.active === 0` after each `$.ajax` (aggregator has no dedicated spinner).
2. **Secondary:** DOM redraw of `#differenceDiv` / `#evidenceDiv` in the AJAX `success` handler.
3. **Turn/pair complete for generate:** `#redirectButton` becomes visible when `numberOfDifferentNodes === 0`.
4. **Generate in flight:** button disabled + text `processing aggregation ...`, then full navigation.

No WebSocket/SSE “idle” signal. Fixed sleeps are unnecessary if automation waits on XHR/`jQuery.active` (and optionally UI predicates above).

## Loading indicators

- Overview create-group: bootstrap-growl “Loading…”.
- Aggregator resolutions: **no** growl/spinner; only silent XHR then redraw.
- Final submit: button label change on `#redirectButton`.

## Auth observation (this Phase 1 session)

The IDE browser session was **not logged in** (`Login` / `Register` visible), yet `POST /evidenceAggregator` still returned a full difference tree for the Test group. Automation should still implement manual login + `storageState` as specified (mutations / other environments may require auth; expiry detection remains mandatory).

## Implications for the sync framework

- Instrument or wait on **XHR** (jQuery.ajax); `fetch` patching is optional defense-in-depth.
- Prefer `waitForResponse` / pending-request counter for `POST /evidenceAggregator` and `POST /evidenceAggregator/submitResult`.
- After each resolution, re-scan **red** spans only (`style*="color:red"` or equivalent), in tree/document order.
- Join is two UI steps: open menu → choose partner name.
- Selectors of note: `#btnAggregateEvidence`, `#redirectButton`, `#differenceDiv`, `#differenceResolutionMenu`, `li[data-resolution]`.

## First pair snapshot (live Test group)

Current pair: evidence `319396` (1) vs `320740` (2). Example red conflicts:

- Under Model quantization (cause): `Post-training quantization with re-training (1)`, `Quantization-aware training (2)` — both contextual aspects → Remove policy.
- Under System: `Machine Transaltion System (1)`, `Speech Recognition System (2)` — first-level contextual aspects ≠ DL model → Remove policy.
