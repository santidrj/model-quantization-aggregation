"""Canonical precision-configuration and quantization-method nomenclature."""

from __future__ import annotations

import re

COMPONENT_ORDER = {"w": 0, "a": 1, "b": 2}

_METHOD_ALIASES = {
    "qat": "qat",
    "quantization-aware training": "qat",
    "quantization aware training": "qat",
    "ptq": "ptq",
    "post-training quantization": "ptq",
    "post training quantization": "ptq",
    "ptq-retrain": "ptq-retrain",
    "post-training quantization with re-training": "ptq-retrain",
    "post-training quantization with retraining": "ptq-retrain",
}

_COMPACT_WA = re.compile(r"^w(?P<w>\d+)a(?P<a>\d+)$", re.IGNORECASE)
_WA_PREFIX = re.compile(r"^wa-(?P<fmt>.+)$", re.IGNORECASE)
_METHOD_PREFIX = re.compile(r"^(?P<method>qat|ptq)-(?P<rest>.+)$", re.IGNORECASE)
_COMPONENT = re.compile(r"^(?P<comp>[a-z]+)-(?P<fmt>.+)$", re.IGNORECASE)
_BARE_INT = re.compile(r"^int(?P<n>\d+)$", re.IGNORECASE)
_BARE_FP = re.compile(r"^fp(?P<n>\d+)$", re.IGNORECASE)
_BARE_Q0 = re.compile(r"^q0[.,](?P<n>\d+)$", re.IGNORECASE)
_BARE_LOG = re.compile(r"^log(?P<n>\d+)$", re.IGNORECASE)
_FULL = re.compile(r"^full-(?P<fmt>.+)$", re.IGNORECASE)


def normalize_quantization_method(label: str) -> str:
    """Map verbose or short method phrases to canonical tokens."""
    key = " ".join(label.strip().lower().replace("_", " ").split())
    key = key.replace("–", "-").replace("—", "-")
    if key in _METHOD_ALIASES:
        return _METHOD_ALIASES[key]
    collapsed = key.replace("-", " ")
    for alias, token in _METHOD_ALIASES.items():
        if alias.replace("-", " ") == collapsed:
            return token
    raise ValueError(f"Unknown quantization method label: {label!r}")


def _fix_format_typos(fmt: str) -> str:  # noqa: PLR0911
    fmt = fmt.strip()
    fmt = fmt.replace(",", ".")
    fmt = re.sub(r"^q\.0*", "q0.", fmt, count=1, flags=re.IGNORECASE)
    if re.fullmatch(r"q0\.\d+", fmt, flags=re.IGNORECASE):
        return f"q0.{fmt.split('.', 1)[1]}"
    if re.fullmatch(r"int\d+", fmt, flags=re.IGNORECASE):
        return fmt.lower()
    if re.fullmatch(r"fp\d+", fmt, flags=re.IGNORECASE):
        return fmt.lower()
    if re.fullmatch(r"log\d+", fmt, flags=re.IGNORECASE):
        return fmt.lower()
    if re.fullmatch(r"q\d+\.\d+", fmt, flags=re.IGNORECASE):
        whole, frac = fmt[1:].split(".", 1)
        return f"q{int(whole)}.{frac}"
    if fmt.lower() == "mixed":
        return "mixed"
    return fmt.lower()


def _format_components(components: dict[str, str]) -> str:
    ordered = sorted(components.items(), key=lambda item: (COMPONENT_ORDER.get(item[0], 100), item[0]))
    return ", ".join(f"{comp}-{fmt}" for comp, fmt in ordered)


def normalize_precision_configuration(label: str, *, prefer_full_float: bool = False) -> str:  # noqa: PLR0911, PLR0912
    """Rewrite a precision-configuration alias to its canonical form.

    Bare float labels (`fp16`, `fp32`, …) expand to equal-component form by default.
    Pass ``prefer_full_float=True`` when the label is a float reference/baseline so it
    becomes ``full-fpN`` instead.
    """
    raw = label.strip()
    if not raw:
        raise ValueError("Empty precision configuration label")

    method_match = _METHOD_PREFIX.match(raw)
    if method_match:
        raw = method_match.group("rest")

    lowered = raw.lower().replace(" ", "")

    compact = _COMPACT_WA.match(lowered)
    if compact:
        return _format_components({"w": f"int{compact.group('w')}", "a": f"int{compact.group('a')}"})

    wa = _WA_PREFIX.match(raw.replace(" ", ""))
    if wa:
        fmt = _fix_format_typos(wa.group("fmt"))
        if fmt.isdigit():
            fmt = f"int{fmt}"
        return _format_components({"w": fmt, "a": fmt})

    full = _FULL.match(raw.replace(" ", ""))
    if full:
        return f"full-{_fix_format_typos(full.group('fmt'))}"

    if ", " in raw:
        parts = [p.strip() for p in raw.split(", ") if p.strip()]
        components: dict[str, str] = {}
        for part in parts:
            match = _COMPONENT.match(part)
            if not match:
                raise ValueError(f"Unknown precision configuration label: {label!r}")
            components[match.group("comp").lower()] = _fix_format_typos(match.group("fmt"))
        return _format_components(components)

    bare = raw.strip()
    single = _COMPONENT.match(bare)
    if single:
        return _format_components({single.group("comp").lower(): _fix_format_typos(single.group("fmt"))})

    if re.fullmatch(r"q0[.,]\d+", bare, flags=re.IGNORECASE) or re.fullmatch(r"q\.\d+", bare, flags=re.IGNORECASE):
        normalized_q = bare if bare.lower().startswith("q0") else f"q0{bare[1:]}"
        fmt = _fix_format_typos(normalized_q)
        return _format_components({"w": fmt, "a": fmt})

    if _BARE_FP.fullmatch(bare):
        fmt = bare.lower()
        if prefer_full_float:
            return f"full-{fmt}"
        return _format_components({"w": fmt, "a": fmt})

    if _BARE_INT.fullmatch(bare):
        fmt = bare.lower()
        return _format_components({"w": fmt, "a": fmt})

    if _BARE_Q0.fullmatch(bare):
        fmt = _fix_format_typos(bare)
        return _format_components({"w": fmt, "a": fmt})

    if _BARE_LOG.fullmatch(bare):
        return bare.lower()

    if bare.lower() == "mixed":
        return "mixed"

    raise ValueError(f"Unknown precision configuration label: {label!r}")


def parse_precision_label(label: str, *, baseline_precision_configuration: str | None = None) -> tuple[str | None, str]:
    """Split an optional method prefix from a precision label and canonicalize the config."""
    raw = label.strip()
    prefer_full = False
    if baseline_precision_configuration and _BARE_FP.fullmatch(raw):
        prefer_full = baseline_precision_configuration == f"full-{raw.lower()}"

    method_match = _METHOD_PREFIX.match(raw)
    if method_match:
        method = normalize_quantization_method(method_match.group("method"))
        return method, normalize_precision_configuration(method_match.group("rest"), prefer_full_float=prefer_full)
    return None, normalize_precision_configuration(raw, prefer_full_float=prefer_full)
