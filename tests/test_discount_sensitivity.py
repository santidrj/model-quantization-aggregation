from src.belief_assignment import EvidenceModel
from src.discount_sensitivity import (
    DiscountSensitivityRow,
    discount_sensitivity_rows,
    remap_evidence_models,
    render_discount_sensitivity_table,
    render_effective_sample_size_table,
)


def test_remap_uses_cluster_n_eff_for_discounted_belief():
    model = EvidenceModel(
        study_id="S14",
        quantization_method="ptq",
        precision_configuration="w-int4",
        study_belief=0.713,
        evidence_model_count=1,
        effects={"Storage Size": ("strongly positive", 0.713)},
    )
    inputs = {("S14", "ptq", "w-int4", "Storage Size"): (18, 0.0, 75.0)}
    remapped = remap_evidence_models([model], inputs, n0=3, k=0.1, cutoff=4)
    assert remapped[0].effects["Storage Size"][1] == 0.712


def test_render_effective_sample_size_table_includes_quartiles():
    latex = render_effective_sample_size_table([1, 1, 2, 3, 18])
    assert "Q3" in latex
    assert "18" in latex


def test_discount_sensitivity_rows_include_two_three_and_six_with_three_as_reference():
    model = EvidenceModel(
        study_id="S1",
        quantization_method="ptq",
        precision_configuration="w-int8",
        study_belief=0.5,
        evidence_model_count=1,
        effects={"Accuracy": ("indifferent", 0.5)},
    )
    inputs = {("S1", "ptq", "w-int8", "Accuracy"): (2, 0.0, 1.0)}
    rows = discount_sensitivity_rows([model], inputs)
    n0_rows = [row for row in rows if row.label.startswith("$n_0=")]
    assert [row.label for row in n0_rows] == ["$n_0=2$", "$n_0=3$", "$n_0=6$"]
    assert [row.is_reference for row in n0_rows] == [False, True, False]


def test_render_discount_sensitivity_table_marks_reference_row():
    row = DiscountSensitivityRow(
        label="$n_0=3$",
        n0=3,
        k=0.1,
        cutoff=4,
        is_reference=True,
        accuracy_intensity="{IF}",
        accuracy_belief_percent=99,
        alizadeh_accuracy_belief=0.031,
    )
    latex = render_discount_sensitivity_table([row])
    assert r"\textbf{reference}" in latex
    assert "0.031" in latex
