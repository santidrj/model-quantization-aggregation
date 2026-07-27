import pytest

from src.data.papers.precision_nomenclature import (
    format_mixed_numeric_format,
    normalize_precision_configuration,
    normalize_quantization_method,
    parse_precision_label,
    precision_configuration_sort_key,
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
        ("mixed-1.8", "mixed-1.8"),
        ("mixed-2.0", "mixed-2"),
        ("mixed-2", "mixed-2"),
        ("w-mixed-1.8", "w-mixed-1.8"),
        ("w-mixed-2.0, a-int8", "w-mixed-2, a-int8"),
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


@pytest.mark.parametrize(
    ("avg", "expected"),
    [
        (1.8, "mixed-1.8"),
        (2.0, "mixed-2"),
        (2, "mixed-2"),
        ("2.0", "mixed-2"),
        ("1.80", "mixed-1.8"),
    ],
)
def test_format_mixed_numeric_format(avg, expected):
    assert format_mixed_numeric_format(avg) == expected


def test_precision_configuration_sort_key_orders_mixed_with_uniform():
    labels = [
        "mixed-2",
        "w-int2, a-int2",
        "mixed-1.8",
        "w-int1, a-int1",
        "w-int4, a-int4",
    ]
    assert sorted(labels, key=precision_configuration_sort_key) == [
        "w-int1, a-int1",
        "mixed-1.8",
        "w-int2, a-int2",
        "mixed-2",
        "w-int4, a-int4",
    ]
