# Mixed precision carries study-reported average bit width

Uniform `intN` labels were swallowing mixed-precision runs (e.g. Xu) when averages were cast to integers, so mixed and uniform setups looked identical. We encode a study-declared mixed quantization with a known average bit width as the numeric format `mixed-<avg>` (trailing `.0` dropped), keep bare `mixed` only when no average is known, and never round that average into a uniform `intN` identity. Eligibility is the study’s mixed-vs-uniform distinction, not whether the average is fractional. Metadata and by-precision keys list each distinct `mixed-<avg>`; analysis order sorts by the numeric average alongside uniform widths, with uniform before `mixed-N` on a tie.

## Considered Options

- Round / cast average bit width to `intN` / `w-intN, a-intN` — rejected; misreads mixed as uniform
- Bare `mixed` only, average elsewhere or discarded — rejected; loses a reported identity dimension and collapses distinct averages
- Side-column for average bit width outside precision configuration — rejected; by-precision identity would still omit the average unless every consumer joined it
