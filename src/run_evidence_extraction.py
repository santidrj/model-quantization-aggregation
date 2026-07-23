import os

from src.config import processed_paper_dir, processed_paper_path
from src.data.papers.entities import Paper, Papers
from src.data.papers.knowledge_extraction import KnowledgeExtractor


def extract_knowledge_from(paper: Paper):
    data = paper.read_data()

    knowledge_extractor = KnowledgeExtractor(
        data,
        paper=paper,
    )

    knowledge_extractor.extract_knowledge()

    os.makedirs(processed_paper_dir(paper.KEY), exist_ok=True)

    knowledge_extractor.improvement_metrics.write_parquet(
        processed_paper_path(paper.KEY, "improvement_metrics.parquet")
    )

    statistics = knowledge_extractor.get_improvement_statistics()
    statistics.write_parquet(processed_paper_path(paper.KEY, "improvement_statistics_by_configuration.parquet"))

    statistics_by_precision = knowledge_extractor.get_improvement_statistics(by_precision=True)
    statistics_by_precision.write_parquet(
        processed_paper_path(paper.KEY, "improvement_statistics_by_precision.parquet")
    )

    knowledge_extractor.save_effects_by_configuration(processed_paper_path(paper.KEY, "effects_by_configuration.json"))
    knowledge_extractor.save_effects_by_precision(processed_paper_path(paper.KEY, "effects_by_precision.json"))


def main(paper: Paper = Papers.DUBHIR.value):
    extract_knowledge_from(paper)


if __name__ == "__main__":
    main()
