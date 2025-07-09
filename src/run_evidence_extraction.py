import os

from src.config import PROCESSED_DATA_DIR
from src.data.papers.entities import Paper, Papers
from src.data.papers.knowledge_extraction import KnowledgeExtractor


def extract_knowledge_from(paper: Paper):
    data = paper.read_data()

    knowledge_extractor = KnowledgeExtractor(
        data,
        paper=paper,
    )

    knowledge_extractor.extract_knowledge()

    os.makedirs(PROCESSED_DATA_DIR / paper.KEY, exist_ok=True)

    knowledge_extractor.improvement_metrics.write_parquet(
        PROCESSED_DATA_DIR / paper.KEY / "improvement_metrics.parquet"
    )

    statistics = knowledge_extractor.get_improvement_statistics()
    statistics.write_parquet(PROCESSED_DATA_DIR / paper.KEY / "improvement_statistics.parquet")

    statistics_by_precision = knowledge_extractor.get_improvement_statistics(by_quantization_precision=True)
    statistics_by_precision.write_parquet(
        PROCESSED_DATA_DIR / paper.KEY / "improvement_statistics_by_precision.parquet"
    )

    knowledge_extractor.write_json(PROCESSED_DATA_DIR / paper.KEY / "effects.json")


for paper in Papers:
    print(f"Extracting knowledge from {paper.value.AUTHOR}")
    extract_knowledge_from(paper.value)
