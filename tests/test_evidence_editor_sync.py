import json

from src.belief_assignment import load_evidence_models
from src.evidence_editor_sync import (
    DesiredModel,
    EffectDelta,
    LiveElement,
    WriteBackCatalog,
    apply_deltas_to_evidence_dto,
    apply_write_back,
    desired_effect,
    intensity_control_option,
    live_elements_from_evidence_dto,
    load_write_back_catalog,
    mapping_integrity_faults,
    models_from_precision_records,
    plan_catalog,
    plan_model,
    supporting_statistics_comment,
    write_back_is_allowed,
)


def test_intensity_control_option_is_title_case_of_processed_phrase():
    assert intensity_control_option("indifferent - weakly positive") == "Indifferent - Weakly Positive"
    assert intensity_control_option("strongly positive") == "Strongly Positive"
    assert intensity_control_option("indifferent") == "Indifferent"


def test_supporting_statistics_comment_is_json_without_belief_or_intensity():
    comment = supporting_statistics_comment(
        {
            "improvement": -1.142,
            "std": 0.043,
            "iqr": 0.012,
            "sample_size_discount": 0.865,
            "variability_discount": 0.999,
            "discount_factor": 0.864,
            "p_value": 0.136,
            "belief": 0.34,
            "intensity": "indifferent",
        }
    )
    assert json.loads(comment) == {
        "improvement": -1.142,
        "std": 0.043,
        "iqr": 0.012,
        "sample_size_discount": 0.865,
        "variability_discount": 0.999,
        "discount_factor": 0.864,
        "p_value": 0.136,
    }


_COMPLETE_ACCURACY = {
    "improvement": -1.142,
    "std": 0.043,
    "iqr": 0.012,
    "sample_size_discount": 0.865,
    "variability_discount": 0.999,
    "discount_factor": 0.864,
    "p_value": 0.136,
    "belief": 0.34,
    "intensity": "indifferent",
}


def test_complete_effect_carries_title_case_intensity_discount_residual_and_comment():
    effect = desired_effect("accuracy", _COMPLETE_ACCURACY)
    assert effect.label == "Accuracy"
    assert effect.complete is True
    assert effect.intensity_option == "Indifferent"
    assert effect.p_value == 0.136
    assert json.loads(effect.comment)["p_value"] == 0.136


def test_effect_missing_intensity_or_improvement_is_incomplete():
    incomplete = desired_effect(
        "mIoU",
        {
            "improvement": None,
            "std": None,
            "iqr": None,
            "sample_size_discount": 0.0,
            "variability_discount": 1.0,
            "discount_factor": 0.0,
            "p_value": 1.0,
            "belief": 0.0,
            "intensity": None,
        },
    )
    assert incomplete.label == "mIoU"
    assert incomplete.complete is False
    assert incomplete.intensity_option is None
    assert incomplete.comment is None


def test_duplicate_mapping_id_blocks_write_back_for_the_whole_corpus():
    faults = mapping_integrity_faults(
        mapping_ids_by_study={"S1": [11, 12], "S4": [21, 22, 21]},
        record_counts_by_study={"S1": 2, "S4": 3},
    )
    assert any(fault.study_id == "S4" and fault.kind == "duplicate_id" for fault in faults)
    assert write_back_is_allowed(faults) is False


def test_length_mismatch_blocks_write_back():
    faults = mapping_integrity_faults(
        mapping_ids_by_study={"S1": [11]},
        record_counts_by_study={"S1": 2},
    )
    assert faults[0].kind == "length_mismatch"
    assert write_back_is_allowed(faults) is False


def test_unique_one_to_one_mapping_allows_write_back():
    faults = mapping_integrity_faults(
        mapping_ids_by_study={"S1": [11, 12], "S2": [21]},
        record_counts_by_study={"S1": 2, "S2": 1},
    )
    assert faults == ()
    assert write_back_is_allowed(faults) is True


def _accuracy_model() -> DesiredModel:
    return DesiredModel(
        study_id="S10",
        evidence_factory_id=260079,
        quantization_method="ptq",
        precision_configuration="w-int8, a-int8",
        effects=(desired_effect("accuracy", _COMPLETE_ACCURACY),),
    )


def test_plan_reports_delta_when_live_intensity_or_discount_residual_or_comment_differs():
    desired = _accuracy_model()
    live = (
        LiveElement(
            label="Accuracy",
            kind="Effect",
            intensity="Strongly Positive",
            p_value=0.136,
            comment=desired.effects[0].comment,
        ),
    )
    plan = plan_model(desired, live)
    assert [delta.label for delta in plan.deltas] == ["Accuracy"]
    assert plan.deltas[0].intensity_option == "Indifferent"


def test_plan_fails_effect_when_label_matches_two_nodes():
    desired = _accuracy_model()
    live = (
        LiveElement("Accuracy", "Effect", "Indifferent", 0.136, desired.effects[0].comment),
        LiveElement("Accuracy", "Effect", "Strongly Positive", 0.9, "{}"),
    )
    plan = plan_model(desired, live)
    assert plan.ambiguous_local_effects == ("Accuracy",)
    assert plan.deltas == ()


def test_plan_skips_incomplete_effects_and_warns_on_unmatched_and_extra_effects():
    desired = DesiredModel(
        study_id="S16",
        evidence_factory_id=1,
        quantization_method="qat",
        precision_configuration="w-int2, a-int2",
        effects=(
            desired_effect("accuracy", _COMPLETE_ACCURACY),
            desired_effect(
                "mIoU",
                {
                    "improvement": None,
                    "std": None,
                    "iqr": None,
                    "sample_size_discount": 0.0,
                    "variability_discount": 1.0,
                    "discount_factor": 0.0,
                    "p_value": 1.0,
                    "belief": 0.0,
                    "intensity": None,
                },
            ),
        ),
    )
    live = (
        LiveElement(label="Storage Size", kind="Effect", intensity="Indifferent", p_value=0.1, comment="{}"),
        LiveElement(label="Technology", kind="archetype", intensity=None, p_value=None, comment=None),
    )
    plan = plan_model(desired, live)
    assert plan.incomplete_effects == ("mIoU",)
    assert plan.unmatched_local_effects == ("Accuracy",)
    assert plan.extra_effect_nodes == ("Storage Size",)
    assert plan.deltas == ()


class _FakeEditor:
    def __init__(self, live: dict[int, tuple[LiveElement, ...]], fail_on: tuple[int, str] | None = None):
        self.live = live
        self.fail_on = fail_on
        self.reads: list[int] = []
        self.writes: list[tuple[int, str]] = []

    def read_effects(self, evidence_factory_id: int) -> tuple[LiveElement, ...]:
        self.reads.append(evidence_factory_id)
        return self.live[evidence_factory_id]

    def write_effect(self, evidence_factory_id: int, delta) -> None:
        if self.fail_on == (evidence_factory_id, delta.label):
            raise RuntimeError("write failed")
        self.writes.append((evidence_factory_id, delta.label))


def test_plan_does_not_open_duplicated_ids_but_live_reads_intact_studies():
    catalog = WriteBackCatalog(
        models=(
            DesiredModel("S1", 11, "ptq", "w-int8", (desired_effect("accuracy", _COMPLETE_ACCURACY),)),
            DesiredModel("S4", 21, "ptq", "w-int8", (desired_effect("accuracy", _COMPLETE_ACCURACY),)),
            DesiredModel("S4", 21, "ptq", "w-int4", (desired_effect("accuracy", _COMPLETE_ACCURACY),)),
        ),
        mapping_ids_by_study={"S1": [11], "S4": [21, 21]},
        record_counts_by_study={"S1": 1, "S4": 2},
    )
    live = {
        11: (
            LiveElement("Accuracy", "Effect", "Strongly Positive", 0.5, "{}"),
        )
    }
    plans = plan_catalog(catalog, live)
    assert [plan.evidence_factory_id for plan in plans] == [11]
    assert plans[0].deltas[0].label == "Accuracy"


def test_apply_is_refused_when_mapping_integrity_fails():
    catalog = WriteBackCatalog(
        models=(DesiredModel("S4", 21, "ptq", "w-int8", (desired_effect("accuracy", _COMPLETE_ACCURACY),)),),
        mapping_ids_by_study={"S4": [21, 21]},
        record_counts_by_study={"S4": 2},
    )
    editor = _FakeEditor(live={})
    report = apply_write_back(catalog, editor)
    assert report.refused is True
    assert editor.reads == []
    assert editor.writes == []


def test_apply_stops_on_first_failed_write():
    first = DesiredModel("S1", 11, "ptq", "w-int8", (desired_effect("accuracy", _COMPLETE_ACCURACY),))
    second = DesiredModel(
        "S2",
        22,
        "ptq",
        "w-int8",
        (desired_effect("storage_size", {**_COMPLETE_ACCURACY, "intensity": "strongly positive"}),),
    )
    catalog = WriteBackCatalog(
        models=(first, second),
        mapping_ids_by_study={"S1": [11], "S2": [22]},
        record_counts_by_study={"S1": 1, "S2": 1},
    )
    stale = LiveElement("Accuracy", "Effect", "Strongly Positive", 0.9, "{}")
    stale_storage = LiveElement("Storage Size", "Effect", "Indifferent", 0.9, "{}")
    editor = _FakeEditor(live={11: (stale,), 22: (stale_storage,)}, fail_on=(11, "Accuracy"))
    report = apply_write_back(catalog, editor)
    assert report.refused is False
    assert report.stopped_on_error is not None
    assert editor.writes == []
    assert editor.reads == [11]


def test_models_from_precision_records_assign_factory_ids_by_list_index():
    records = [
        {
            "quantization_method": "ptq",
            "precision_configuration": "w-int8",
            "accuracy": _COMPLETE_ACCURACY,
        }
    ]
    models = models_from_precision_records("S10", records, [260079])
    assert len(models) == 1
    assert models[0].evidence_factory_id == 260079
    assert models[0].effects[0].label == "Accuracy"
    assert models[0].effects[0].intensity_option == "Indifferent"


def test_write_back_catalog_uses_the_same_factory_ids_as_evidence_models():
    catalog = load_write_back_catalog()
    evidence_models = load_evidence_models()
    assert [(model.study_id, model.evidence_factory_id) for model in catalog.models] == [
        (model.study_id, model.evidence_factory_id) for model in evidence_models
    ]


def test_causes_relationships_in_evidence_dto_are_effect_nodes():
    live = live_elements_from_evidence_dto(
        {
            "relationships": [
                {
                    "type": "TYPE_OF",
                    "toTerm": {"name": "System"},
                    "propositionId": None,
                    "propositionOrder": None,
                    "pValue": None,
                    "explanation": None,
                },
                {
                    "type": "CAUSES",
                    "toTerm": {"name": "Accuracy"},
                    "propositionId": 30,
                    "propositionOrder": 0.0,
                    "pValue": 0.368,
                    "explanation": '{"p_value": 0.368}',
                },
            ]
        }
    )
    assert live == (
        LiveElement(
            label="Accuracy",
            kind="Effect",
            intensity="Indifferent",
            p_value=0.368,
            comment='{"p_value": 0.368}',
        ),
    )


def test_apply_deltas_updates_matching_causes_and_leaves_belief_alone():
    dto = {
        "id": 1,
        "relationships": [
            {
                "type": "CAUSES",
                "toTerm": {"name": "Accuracy"},
                "propositionId": 30,
                "propositionOrder": 0.0,
                "pValue": 0.9,
                "explanation": "old",
                "beliefProbability": 0.4,
            }
        ],
    }
    updated = apply_deltas_to_evidence_dto(
        dto,
        (
            EffectDelta(
                label="Accuracy",
                intensity_option="Strongly Positive",
                p_value=0.136,
                comment='{"p_value": 0.136}',
            ),
        ),
    )
    relationship = updated["relationships"][0]
    assert relationship["propositionId"] == 66
    assert relationship["propositionOrder"] == 3.0
    assert relationship["pValue"] == 0.136
    assert relationship["explanation"] == '{"p_value": 0.136}'
    assert relationship["beliefProbability"] == 0.4
    assert dto["relationships"][0]["pValue"] == 0.9


def test_apply_deltas_skips_when_label_matches_two_causes():
    dto = {
        "id": 1,
        "relationships": [
            {
                "type": "CAUSES",
                "toTerm": {"name": "Accuracy"},
                "propositionId": 30,
                "propositionOrder": 0.0,
                "pValue": 0.9,
                "explanation": "old",
                "beliefProbability": 0.4,
            },
            {
                "type": "CAUSES",
                "toTerm": {"name": "Accuracy"},
                "propositionId": 66,
                "propositionOrder": 3.0,
                "pValue": 0.8,
                "explanation": "other",
                "beliefProbability": 0.4,
            },
        ],
    }
    updated = apply_deltas_to_evidence_dto(
        dto,
        (
            EffectDelta(
                label="Accuracy",
                intensity_option="Strongly Positive",
                p_value=0.136,
                comment='{"p_value": 0.136}',
            ),
        ),
    )
    assert updated["relationships"][0]["pValue"] == 0.9
    assert updated["relationships"][1]["pValue"] == 0.8
