# Replication Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18401926.svg)](https://doi.org/10.5281/zenodo.18401926)

Replication package for the paper:

"Theory Building from Data Strategy Studies: Aggregating Evidence on Model Quantization in Deep Learning Systems" submitted to the Empirical Software Engineering Journal.

## Contents

This replication package consists of the following components:

1. **Data**:
   - Raw, external, interim, and processed data are stored in the [data](data) directory.

2. **Source Code**:
   - Located in the [src](src) directory, it includes scripts for data processing, analysis, and evidence extraction.
   - Key modules:
     - [data/papers/entities.py](data/papers/entities.py) & [data/papers/knowledge_extraction.py](data/papers/knowledge_extraction.py): Define the structure and data extraction logic for the papers analyzed.
     - [data/download.py](data/download.py): Downloads the list of papers from arXiv and merges them with the Scopus list.
     - [data/selection/llm.py](data/selection/llm.py): Implements logic for selecting studies using Gemini 3.0 Flash.

3. **Jupyter Notebooks**:
   - Located in the [notebooks](notebooks) directory, these notebooks contain the analysis and visualization of the data.
   - Notebooks include:
     - [1.0-llm-promt-refinement.ipynb](notebooks/1.0-llm-promt-refinement.ipynb): Refines the prompt for LLMs and the selection of LLM.
     - [2.0-model-quantization-paper-selection.ipynb](notebooks/2.0-model-quantization-paper-selection.ipynb): Filters the raw list of papers using the selected GEMINI 3.0.
     - [3.0-final-selection-analysis.ipynb](notebooks/3.0-final-selection-analysis.ipynb): Analyzes the final selection of papers.
     - [4.0-paper-metadata-analysis.ipynb](notebooks/4.0-paper-metadata-analysis.ipynb): **Supplementary** — optional audit and characterization of manually extracted paper metadata.
     - [5.0-evidence-analysis.ipynb](notebooks/5.0-evidence-analysis.ipynb): **Core** — reproduces the paper's evidence figures (Fig. 6, Fig. 7, and Fig. 10).

4. **Documentation**:
   - [data/processed/evidence-diagrams-mapping.md](data/processed/evidence-diagrams-mapping.md): Links to evidence diagrams generated during the study.
   - `data/processed/{paperkey}/metadata.json`: Contains metadata for the specific paper.
   - `data/processed/{paperkey}/systematic-studies-quality-evaluation.md`: Contains the filled quality evaluation form for the specific paper.

### Project Structure

The project is organized as follows:

```text
├── data/
│   ├── raw/                                <- Contains the original list of papers retrieved from Scopus
│   ├── external/                           <- Contains the raw data obtained from the selected papers
│   ├── interim/                            <- Contains the interim data used in the analysis
│   └── processed/                          <- Contains the processed data used in the analysis
│       └── evidence-diagrams-mapping.md    <- Contains links to the evidence diagrams
├── notebooks/
│   ├── 1.0-llm-promt-refinement.ipynb
│   ├── 2.0-model-quantization-paper-selection.ipynb
│   ├── 3.0-second-selection-analysis.ipynb
│   ├── 4.0-paper-metadata-analysis.ipynb
│   └── 5.0-evidence-analysis.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── papers/                         <- Contains the logic for extracting and analyzing data from papers
│   │   │   ├── entities.py
│   │   │   └── knowledge_extraction.py
│   │   ├── download.py
│   │   └── selection/                      <- Utility functions for selecting studies using LLMs,
│   │       └── llm.py                         including the prompt
│   ├── forestplot/                         <- Utility functions for generating the forest plot
│   ├── effect_intensity.py                 <- Definition of the effect intensity thresholds
│   ├── run_evidence_extraction.py
│   └── config.py
├── .pre-commit-config.yaml
├── dot-env-template                        <- Template for environment variables
├── requirements.txt                        <- List of Python dependencies
├── uv.lock                                 <- Environment lock file
├── LICENSE
├── pyproject.toml                          <- Project configuration file
└── README.md
```

## Usage Instructions

1. **Setup**:
   - Clone the repository:

     ```bash
     git clone <repository-url>
     cd green-tactics-synthesis
     ```

   - Install dependencies:  
     The project is managed with [uv](https://docs.astral.sh/uv/). To install the dependencies, run:

     ```bash
     uv sync
     ```

     Alternatively, you can use pip to install the dependencies listed in `requirements.txt`:

     ```bash
     pip install -r requirements.txt
     ```

   - **Using Docker** (recommended for reproducibility):  
     A pre-built Docker image is available on Docker Hub:

     ```bash
     docker pull santidr/model-quantization-aggregation
     ```

     Run the container with Jupyter Lab:

     ```bash
     docker run -it -p 8888:8888 santidr/model-quantization-aggregation
     ```

     To use LLM features (paper selection), pass your API key:

     ```bash
     docker run -it -p 8888:8888 \
       -e GEMINI_API_KEY=your_key \
       santidr/model-quantization-aggregation
     ```

     To persist data changes, mount local directories:

     ```bash
     docker run -it -p 8888:8888 \
       -v $(pwd)/data:/app/data \
       -v $(pwd)/reports:/app/reports \
       santidr/model-quantization-aggregation
     ```

2. **Getting the Data**:
   - Run the download script to fetch the list of papers from arXiv and merge it with the Scopus list:

     ```bash
     python src/data/downlad.py
     ```

   - We do not commit external paper data from the selected studies (copyright / size). Most papers expect a local `paper-data.csv` under [data/external](data/external); each paper folder's README explains how to obtain it.
   - Papers with a remote archive descriptor (currently Alizadeh and Gonzalez) **auto-download on first read** when their required files are missing. That needs network access; Gonzalez's first fetch pulls an ~853 MB Zenodo archive, extracts only the required CSV, and discards the rest. Files still are not shipped in git.

3. **Extracting the evidence** (full pipeline validation only):
   - Use [run_evidence_extraction.py](src/run_evidence_extraction.py) to regenerate processed parquets from external study data. This step is **not** required if you only want to reproduce the paper figures from the shipped `data/processed/` artifacts.

## Reproducing the paper

### Path 1 — Reproduce paper figures (default)

1. Complete **Setup** above (`uv sync`).
2. Start Jupyter Lab: `uv run jupyter lab`.
3. Open [5.0-evidence-analysis.ipynb](notebooks/5.0-evidence-analysis.ipynb) and run all sections labeled **Core** in order.
4. Verify outputs in [reports/figures/](reports/figures/):

   | Output file | Paper figure |
   |---|---|
   | `metrics-usage-distribution.pdf` | Fig. 10 |
   | `correctness-forestplot.pdf` | Fig. 6 |
   | `resource-efficiency-forestplot.pdf` | Fig. 7 |

Processed evidence under `data/processed/{paperkey}/` is included in the replication package, so you do not need external study data or evidence extraction for this path.

### Path 2 — Reproduce the full extraction pipeline (validation)

1. Complete **Setup** and **Getting the Data** (place external study data per `data/external/{paperkey}/README.md`).
2. Run `uv run python src/run_evidence_extraction.py`.
3. Run the **Core** sections of [5.0-evidence-analysis.ipynb](notebooks/5.0-evidence-analysis.ipynb) as in Path 1.

### Supplementary notebooks

- [4.0-paper-metadata-analysis.ipynb](notebooks/4.0-paper-metadata-analysis.ipynb) — optional metadata audit and characterization; not required to reproduce paper figures.
- **Supplementary** sections inside [5.0-evidence-analysis.ipynb](notebooks/5.0-evidence-analysis.ipynb) — diagnostics (`distribution-of-sample-size.pdf`) and appendix forest plots (`complete-*-forestplot.pdf`).

Domain terminology is defined in [CONTEXT.md](CONTEXT.md).

## Notes

- Ensure all required data is placed in the appropriate directories.
- For any issues or questions, please contact the authors of the paper.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
