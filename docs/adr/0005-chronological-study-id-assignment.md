# Chronological Study ID assignment

Included primary studies are labeled with consecutive **Study IDs** `S1`…`Sn` (no gaps), assigned by publication year ascending then lead-author citation name ascending within a year. Study IDs are stamped as explicit constants on each paper entity and checked by test against that rule; they are distinct from the durable **paper key** used for paths. User-facing order is numeric on the integer after `S`, not lexicographic string sort. After a renumber, evidence extraction is regenerated so processed outputs match the stamps. The `Papers` enum declaration order follows Study ID order, but call sites that present ordered lists still sort explicitly by numeric Study ID.

## Considered Options

- Close gaps while preserving prior relative `S#` order — rejected; leaves snowball studies out of chronological sequence
- Derive Study IDs at runtime from the current paper set — rejected; metadata edits must not silently reshuffle public manuscript labels
- Zero-padded labels (`S01`…) so naive string sort works — rejected; keep familiar `S1`…`Sn` and fix sorts to be numeric
- Rename paper keys / data folders to match Study IDs — rejected; paper key remains the stable filesystem identity
