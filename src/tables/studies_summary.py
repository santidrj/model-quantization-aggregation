"""Generate manuscript studies-summary LaTeX tabular fragments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from src.config import TABLES_DIR, processed_paper_path
from src.data.papers.entities import Paper, Papers
from src.data.papers.study_id import study_id_number

QUASI_EXPERIMENTS = "quasi-experiments"
OBSERVATIONAL = "observational"

QUASI_EXPERIMENTS_FILENAME = "studies-quasi-experiments.tex"
OBSERVATIONAL_FILENAME = "studies-observational.tex"

_MEASURED_METHODS = frozenset({"hardware-based", "software-based"})
_ESTIMATED_METHODS = frozenset({"analytical", "model-based"})

_PAPER_KEY_AUTHOR = re.compile(r"^(?P<author>[a-z]+)")

_TIMING_LABELS = {
    "qat": "QAT",
    "ptq": "PTQ",
    "ptq-retrain": "PTQ-R",
}

_FULL_BASELINE = re.compile(r"^full-(?P<fmt>.+)$", re.IGNORECASE)
_COMPONENT = re.compile(r"^(?P<comp>[a-z]+)-(?P<fmt>.+)$", re.IGNORECASE)
_INT_FMT = re.compile(r"^int(?P<n>\d+)$", re.IGNORECASE)
_FP_FMT = re.compile(r"^fp(?P<n>\d+)$", re.IGNORECASE)
_Q0_FMT = re.compile(r"^q0\.(?P<n>\d+)$", re.IGNORECASE)
_LOG_FMT = re.compile(r"^log(?P<n>\d+)$", re.IGNORECASE)
_MIXED_AVG = re.compile(r"^mixed-(?P<avg>\d+(?:\.\d+)?)$", re.IGNORECASE)

_PATTERN_ORDER = ("W", "A", "B", "WA", "WAB", "F")


@dataclass(frozen=True)
class StudySummaryRow:
    paper_key: str
    study_id: str
    belief: float
    domain: str
    instrumentation: str
    models_data: str
    source: str
    target: str
    timing: str
    ts: int
    energy_evidence_group: str = QUASI_EXPERIMENTS


def energy_evidence_group(methods: list[str] | None) -> str:
    """Map energy measurement methods to a manuscript energy evidence group.

    Null / empty methods (study did not measure energy) belong with quasi-experiments.
    """
    normalized = list(methods or [])
    if not normalized:
        return QUASI_EXPERIMENTS

    unknown = set(normalized) - _MEASURED_METHODS - _ESTIMATED_METHODS
    if unknown:
        raise ValueError(f"Unknown energy measurement method(s) for energy evidence group: {sorted(unknown)}")

    has_measured = bool(set(normalized) & _MEASURED_METHODS)
    has_estimated = bool(set(normalized) & _ESTIMATED_METHODS)
    if has_measured and has_estimated:
        raise ValueError(
            f"Cannot assign energy evidence group when methods mix measured and estimated tokens: {normalized}"
        )
    if has_estimated:
        return OBSERVATIONAL
    return QUASI_EXPERIMENTS


def paper_key_author_macro(paper_key: str) -> str:
    """Leading lowercase author segment of a paper key (citation macro name)."""
    match = _PAPER_KEY_AUTHOR.match(paper_key)
    if not match:
        raise ValueError(f"Cannot derive author macro from paper key: {paper_key!r}")
    return match.group("author")


def study_id_latex(paper_key: str) -> str:
    """Manuscript ID cell: author macro plus cite of the paper key."""
    macro = paper_key_author_macro(paper_key)
    return rf"\{macro} \cite{{{paper_key}}}"


def format_belief_percent(belief: float) -> str:
    """Study belief as a nearest-integer percent for LaTeX."""
    return f"{round(belief * 100)}\\%"


def format_timing(methods: list[str]) -> str:
    """Explicit quantization-method labels for the Timing column."""
    labels: list[str] = []
    for method in methods:
        try:
            labels.append(_TIMING_LABELS[method])
        except KeyError as exc:
            raise ValueError(f"Unknown quantization method for Timing: {method!r}") from exc
    return ", ".join(labels)


def format_models_data(models: list[str], datasets: list[str]) -> str:
    """Models/Data cell from metadata list lengths."""
    return f"{len(models)} / {len(datasets)}"


def format_instrumentation(methods: list[str] | None, software_tools: list[str] | None) -> str:  # noqa: PLR0911
    """Instrumentation cell derived from energy measurement method and tools."""
    normalized = list(methods or [])
    tools = list(software_tools or [])
    if not normalized:
        return "N/A"
    if normalized == ["analytical"]:
        return "Analytical"
    if normalized == ["model-based"]:
        return "Model-based"
    if set(normalized) <= _MEASURED_METHODS:
        if tools and "hardware-based" in normalized and "software-based" in normalized:
            return f"Hardware + {', '.join(tools)}"
        if tools:
            return ", ".join(tools)
        if "software-based" in normalized and "hardware-based" not in normalized:
            return "Software-based"
        return "Hardware-based"
    raise ValueError(f"Cannot format instrumentation for methods={normalized!r} tools={tools!r}")


def format_source_precision(baseline: str) -> str:
    """Compact Source cell from a baseline precision configuration."""
    match = _FULL_BASELINE.fullmatch(baseline.strip())
    if not match:
        raise ValueError(f"Unsupported baseline precision configuration: {baseline!r}")
    return _display_numeric_format(match.group("fmt"))


def format_target_precision(configs: list[str], baseline: str) -> str:
    """Compact Target cell from precision configurations and baseline."""
    baseline_fmt = None
    baseline_match = _FULL_BASELINE.fullmatch(baseline.strip())
    if baseline_match:
        baseline_fmt = baseline_match.group("fmt").lower()

    patterns: set[str] = set()
    formats: list[tuple[str, float | None, str]] = []
    seen_formats: set[str] = set()

    for config in configs:
        pattern, config_formats = _parse_precision_configuration(config, baseline_fmt=baseline_fmt)
        patterns.add(pattern)
        for raw_fmt in config_formats:
            if baseline_fmt is not None and raw_fmt.lower() == baseline_fmt:
                continue
            display = _display_numeric_format(raw_fmt)
            if display in seen_formats:
                continue
            seen_formats.add(display)
            formats.append((display, _format_sort_width(raw_fmt), raw_fmt.lower()))

    if not formats and configs:
        for config in configs:
            _, config_formats = _parse_precision_configuration(config, baseline_fmt=None)
            for raw_fmt in config_formats:
                display = _display_numeric_format(raw_fmt)
                if display in seen_formats:
                    continue
                seen_formats.add(display)
                formats.append((display, _format_sort_width(raw_fmt), raw_fmt.lower()))

    formats.sort(key=lambda item: (item[1] is None, item[1] if item[1] is not None else 0.0, item[0]))
    format_text = _compress_format_list([item[0] for item in formats], [item[2] for item in formats])
    pattern_text = ", ".join(p for p in _PATTERN_ORDER if p in patterns)
    extra = sorted(patterns - set(_PATTERN_ORDER))
    if extra:
        pattern_text = ", ".join(filter(None, [pattern_text, *extra]))
    return f"{format_text} ({pattern_text})"


def render_tabularx_body(rows: list[StudySummaryRow]) -> str:
    """Render a studies-summary ``tabularx`` body (no float chrome)."""
    sorted_rows = sorted(
        rows,
        key=lambda row: (-round(row.belief * 100), study_id_number(row.study_id)),
    )
    lines = [
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xl>{\raggedright\arraybackslash}lclllcc}",
        r"\toprule",
        r"\rowcolor{gray!30}",
        (
            r" & \multicolumn{2}{c}{\textbf{Contextual Factors}} & \multicolumn{1}{c}{\textbf{Exp. Units}}"
            r" & \multicolumn{3}{c}{\textbf{Quantization Design}} & \multicolumn{2}{c}{\textbf{SSM Metrics}} \\"
        ),
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-4} \cmidrule(lr){5-7} \cmidrule(lr){8-9}",
        r"\rowcolor{gray!30}",
        (
            r"ID & Domain & Instrumentation & Models/Data & Source & Target\footnotemark[1]"
            r" & Timing & Belief & TS\footnotemark[2] \\"
        ),
        r"\midrule",
    ]
    for row in sorted_rows:
        lines.append(
            " & ".join(
                [
                    study_id_latex(row.paper_key),
                    row.domain,
                    row.instrumentation,
                    row.models_data,
                    row.source,
                    row.target,
                    row.timing,
                    format_belief_percent(row.belief),
                    str(row.ts),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    return "\n".join(lines)


def write_studies_summary_tables(
    rows: list[StudySummaryRow],
    *,
    output_dir: Path,
) -> list[Path]:
    """Write quasi-experiment and observational tabular fragments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        (QUASI_EXPERIMENTS, QUASI_EXPERIMENTS_FILENAME),
        (OBSERVATIONAL, OBSERVATIONAL_FILENAME),
    )
    written: list[Path] = []
    for group, filename in targets:
        path = output_dir / filename
        group_rows = [row for row in rows if row.energy_evidence_group == group]
        path.write_text(render_tabularx_body(group_rows), encoding="utf-8")
        written.append(path)
    return written


def build_study_summary_rows(papers: list[Paper]) -> list[StudySummaryRow]:
    """Build manuscript summary rows from included papers and processed artifacts."""
    rows: list[StudySummaryRow] = []
    for paper in papers:
        metadata = json.loads(processed_paper_path(paper.KEY, "metadata.json").read_text(encoding="utf-8"))
        domain = metadata.get("domain")
        if not domain:
            raise ValueError(f"Missing required domain for paper {paper.KEY!r}")

        energy = metadata.get("energy_measurement") or {}
        methods = _as_str_list(energy.get("measurement_method"))
        tools = _as_str_list(energy.get("software_tools"))

        schema = metadata["quantization_schema"]
        baseline = schema["baseline_precision_configuration"]
        precision_configs = list(schema.get("precision_configurations") or [])
        quantization_methods = list(schema.get("quantization_method") or [])

        effects = json.loads(processed_paper_path(paper.KEY, "effects_by_precision.json").read_text(encoding="utf-8"))
        rows.append(
            StudySummaryRow(
                paper_key=paper.KEY,
                study_id=paper.ID,
                belief=paper.BELIEF,
                domain=domain,
                instrumentation=format_instrumentation(methods, tools),
                models_data=format_models_data(
                    list(metadata.get("models") or []),
                    list(metadata.get("datasets") or []),
                ),
                source=format_source_precision(baseline),
                target=format_target_precision(precision_configs, baseline),
                timing=format_timing(quantization_methods),
                ts=len(effects),
                energy_evidence_group=energy_evidence_group(methods),
            )
        )
    return rows


def generate_studies_summary_tables(*, output_dir: Path | None = None) -> list[Path]:
    """Generate both studies-summary tabular fragments for all included papers."""
    papers = [paper.value for paper in Papers]
    rows = build_study_summary_rows(papers)
    return write_studies_summary_tables(rows, output_dir=output_dir or TABLES_DIR)


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]  # type: ignore[arg-type]


def _display_numeric_format(fmt: str) -> str:  # noqa: PLR0911
    token = fmt.strip()
    lower = token.lower()
    if lower == "mixed":
        return "Mixed"
    mixed = _MIXED_AVG.fullmatch(lower)
    if mixed:
        return f"Mixed-{mixed.group('avg')}"
    int_match = _INT_FMT.fullmatch(lower)
    if int_match:
        return f"INT{int_match.group('n')}"
    fp_match = _FP_FMT.fullmatch(lower)
    if fp_match:
        return f"FP{fp_match.group('n')}"
    q0_match = _Q0_FMT.fullmatch(lower)
    if q0_match:
        return f"Q0.{q0_match.group('n')}"
    log_match = _LOG_FMT.fullmatch(lower)
    if log_match:
        return f"LOG{log_match.group('n')}"
    return token.upper()


def _format_sort_width(fmt: str) -> float | None:
    lower = fmt.strip().lower()
    for regex in (_INT_FMT, _FP_FMT, _Q0_FMT, _LOG_FMT):
        match = regex.fullmatch(lower)
        if match:
            return float(match.group("n"))
    mixed = _MIXED_AVG.fullmatch(lower)
    if mixed:
        return float(mixed.group("avg"))
    if lower == "mixed":
        return None
    return None


def _parse_precision_configuration(
    config: str,
    *,
    baseline_fmt: str | None = None,
) -> tuple[str, list[str]]:
    raw = config.strip()
    full = _FULL_BASELINE.fullmatch(raw)
    if full:
        return "F", [full.group("fmt")]
    if raw.lower() == "mixed" or _MIXED_AVG.fullmatch(raw.lower()):
        return "WA", [raw]

    components: dict[str, str] = {}
    for part in raw.split(", "):
        match = _COMPONENT.fullmatch(part.strip())
        if not match:
            raise ValueError(f"Unsupported precision configuration: {config!r}")
        components[match.group("comp").lower()] = match.group("fmt")

    pattern_components = {
        comp: fmt for comp, fmt in components.items() if baseline_fmt is None or fmt.lower() != baseline_fmt
    }
    if not pattern_components:
        pattern_components = components

    keys = set(pattern_components)
    if keys == {"w"}:
        pattern = "W"
    elif keys == {"a"}:
        pattern = "A"
    elif keys == {"b"}:
        pattern = "B"
    elif keys == {"w", "a"}:
        pattern = "WA"
    elif keys == {"w", "a", "b"}:
        pattern = "WAB"
    else:
        pattern = "".join(sorted(keys)).upper()
    return pattern, list(components.values())


def _compress_format_list(displays: list[str], raws: list[str]) -> str:
    if len(displays) <= 1:
        return ", ".join(displays)

    families = {_format_family(raw) for raw in raws}
    widths = [_format_sort_width(raw) for raw in raws]
    if len(families) == 1 and None not in widths and _is_compressible_range(widths):
        family = next(iter(families))
        lo, hi = displays[0], displays[-1]
        if family in {"q0", "int", "fp", "log"}:
            return f"{lo}--{hi}"
    return ", ".join(displays)


def _format_family(fmt: str) -> str:
    lower = fmt.strip().lower()
    if _INT_FMT.fullmatch(lower):
        return "int"
    if _FP_FMT.fullmatch(lower):
        return "fp"
    if _Q0_FMT.fullmatch(lower):
        return "q0"
    if _LOG_FMT.fullmatch(lower):
        return "log"
    if lower == "mixed" or _MIXED_AVG.fullmatch(lower):
        return "mixed"
    return "other"


_MIN_RANGE_LEN = 3


def _is_compressible_range(widths: list[float | None]) -> bool:
    values = [int(w) for w in widths if w is not None]
    if len(values) < _MIN_RANGE_LEN:
        return False
    lo, hi = values[0], values[-1]
    if values == list(range(lo, hi + 1)):
        return True
    expected = [values[0]]
    while expected[-1] < values[-1]:
        expected.append(expected[-1] * 2)
    return values == expected
