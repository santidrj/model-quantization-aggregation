import pytest

from src.data.papers.precision_nomenclature import (
    normalize_precision_configuration,
    normalize_quantization_method,
    parse_precision_label,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("int8", "w-int8, a-int8"),
        ("int4", "w-int4, a-int4"),
        ("q0.8", "w-q0.8, a-q0.8"),
        ("fp32", "w-fp32, a-fp32"),
        ("fp16", "w-fp16, a-fp16"),
        ("full-fp32", "full-fp32"),
        ("full-fp16", "full-fp16"),
        ("full-int8", "full-int8"),
        ("w-int8", "w-int8"),
        ("w-int8, a-int4", "w-int8, a-int4"),
        ("a-int8, w-int4", "w-int4, a-int8"),
        ("wa-int8", "w-int8, a-int8"),
        ("w8a8", "w-int8, a-int8"),
        ("log4", "log4"),
        ("w-log4", "w-log4"),
        ("mixed", "mixed"),
        ("q0,32", "w-q0.32, a-q0.32"),
        ("w-q.016", "w-q0.16"),
        ("a-q8.0", "a-q8.0"),
        ("w-int2, a-int8, b-int8", "w-int2, a-int8, b-int8"),
    ],
)
def test_normalize_precision_configuration(raw, expected):
    assert normalize_precision_configuration(raw) == expected


def test_normalize_baseline_float_uses_full_form():
    assert normalize_precision_configuration("fp32", prefer_full_float=True) == "full-fp32"
    assert parse_precision_label("fp32", baseline_precision_configuration="full-fp32") == (
        None,
        "full-fp32",
    )
    assert parse_precision_label("fp16", baseline_precision_configuration="full-fp32") == (
        None,
        "w-fp16, a-fp16",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("quantization-aware training", "qat"),
        ("quantization aware training", "qat"),
        ("qat", "qat"),
        ("post-training quantization", "ptq"),
        ("ptq", "ptq"),
        ("post-training quantization with re-training", "ptq-retrain"),
        ("ptq-retrain", "ptq-retrain"),
    ],
)
def test_normalize_quantization_method(raw, expected):
    assert normalize_quantization_method(raw) == expected


def test_normalize_quantization_method_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown quantization method"):
        normalize_quantization_method("knowledge distillation")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("qat-w8a8", ("qat", "w-int8, a-int8")),
        ("ptq-w8a8", ("ptq", "w-int8, a-int8")),
        ("int8", (None, "w-int8, a-int8")),
        ("w-int8, a-int4", (None, "w-int8, a-int4")),
        ("full-int8", (None, "full-int8")),
    ],
)
def test_parse_precision_label(raw, expected):
    assert parse_precision_label(raw) == expected
