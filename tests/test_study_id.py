"""Study ID stamps and numeric ordering (ADR-0005)."""

from src.data.papers.entities import Papers
from src.data.papers.study_id import (
    lead_author_citation_name,
    study_id_number,
    study_id_sort_key,
)

# Independent expected map from chronological assignment (year, then lead-author citation name).
EXPECTED_STUDY_IDS_BY_KEY = {
    "ajiCompressingNeuralMachine2020": "S1",
    "denkingerImpactMemoryVoltage2020": "S2",
    "barnellModelQuantizationSynthetic2021": "S3",
    "dubhirBenchmarkingQuantizationLibraries2021": "S4",
    "vasquezActivationDensityBased2021": "S5",
    "xuMixedPrecisionLowBit2021": "S6",
    "zhanFieldProgrammableGate2021": "S7",
    "flichEfficientInferenceImageBased2022": "S8",
    "paulEnergyEfficientRespiratoryAnomaly2022": "S9",
    "sathishVerifiableEnergyEfficient2022": "S10",
    "taoExperimentalEnergyConsumption2022": "S11",
    "chenImplementingUltralightweightCoinference2023": "S12",
    "gonzalezImpactMLOptimization2025": "S13",
    "alizadehLanguageModelsSoftware2025": "S14",
    "alshammryQYOLOv5mQuantizationbasedApproach2025": "S15",
    "deputterPOQThereParetoOptimal2025": "S16",
    "guerroujQuantizedObjectDetection2025": "S17",
    "khalilEnergyEfficientDeepLearning2025": "S18",
    "koliEdgeAIPoweredSystem2025": "S19",
    "krastevaImplementingDeepNeural2025": "S20",
    "pengEfficientExpirationDate2025": "S21",
}


def test_study_id_stamps_match_chronological_assignment():
    actual = {paper.value.KEY: paper.value.ID for paper in Papers}
    assert actual == EXPECTED_STUDY_IDS_BY_KEY


def test_study_id_stamps_follow_year_then_lead_author_rule():
    ordered = sorted(
        Papers,
        key=lambda paper: (
            paper.value.YEAR,
            lead_author_citation_name(paper.value.AUTHOR).casefold(),
        ),
    )
    assert [paper.value.ID for paper in ordered] == [f"S{n}" for n in range(1, len(ordered) + 1)]


def test_papers_enum_follows_study_id_order():
    ids = [paper.value.ID for paper in Papers]
    assert ids == [f"S{n}" for n in range(1, len(ids) + 1)]


def test_study_ids_are_consecutive_without_gaps():
    numbers = sorted(study_id_number(paper.value.ID) for paper in Papers)
    assert numbers == list(range(1, len(numbers) + 1))


def test_lead_author_citation_name_strips_et_al_and_and():
    assert lead_author_citation_name("De Putter et al.") == "De Putter"
    assert lead_author_citation_name("Gonzalez Alvarez et al.") == "Gonzalez Alvarez"
    assert lead_author_citation_name("Aji and Heafield") == "Aji"


def test_study_id_sort_is_numeric_not_lexicographic():
    labels = ["S10", "S2", "S1", "S21"]
    assert sorted(labels, key=study_id_sort_key) == ["S1", "S2", "S10", "S21"]


def test_study_id_sort_places_non_study_labels_after_study_ids():
    labels = ["Aggregated", "S2", "S10", "Summary"]
    assert sorted(labels, key=study_id_sort_key) == ["S2", "S10", "Aggregated", "Summary"]
