# model-quantization-aggregation

Replication package for the paper:

"Aggregating empirical evidence from data strategies studies: a case on model quantization" submitted to the 19th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM).

## Contents

This replication package contains the following components:

1. **Data**:
   - Raw, external, interim, and processed data are stored in the `data/` directory.

2. **Source Code**:
   - Located in the `src/` directory, it includes scripts for data processing, analysis, and evidence extraction.
   - Key modules:
     - `data/papers/entities.py` & `data/papers/knowledge_extraction.py`: Define the structure and data extraction logic for the papers analyzed.
     - `data/download.py`: Downloads the list of papers from arXiv and merges them with the Scopus list.
     - `data/selection/llm.py`: Implements logic for selecting studies using large language models.

3. **Jupyter Notebooks**:
   - Located in the `notebooks/` directory, these notebooks contain the analysis and visualization of the data.
   - Notebooks include:
     - `1.0-llm-promt-refinement.ipynb`: Refines the prompt for LLMs and the selection of LLM.
     - `2.0-model-quantization-paper-selection.ipynb`: Filters the raw list of papers using the selected GEMINI 2.0.
     - `3.0-final-selection-analysis.ipynb`: Analyzes the final selection of papers.
     - `4.0-paper-metadata-analysis.ipynb`: Analyzes metadata from selected papers.
     - `5.0-evidence-analysis.ipynb`: Analyzes evidence extracted from the papers and generates the forest plot.

4. **Documentation**:
   - `data/processed/evidence-diagrams-mapping.md`: Links to evidence diagrams generated during the study.
   - `data/processed/paperkey/metadata.json`: Contains metadata for the specific paper.
   - `data/processed/paperkey/systematic-studies-quality-evaluation.md`: Contains the filled quality evaluation form for the specific paper.

### Project Structure

The project is organized as follows:
```
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

2. **Getting the Data**:
   - Run the download script to fetch the list of papers from arXiv and merge it with the Scopus list:  
     ```bash
     python src/data/downlad.py
     ```

   - We do not provide the raw data from the selected papers to prevent potential copyright issues. However, we provide instructions on how to obtain the data in each paper's README file. Located in the `data/external/` directory.

3. **Extracting the evidence**:
   - Use the `run_evidence_extraction.py` module to extract the evidence from the selected papers.

4. **Explore the data with Jupyter Notebooks**:
   - Open the Jupyter notebooks in the `notebooks/` directory to explore the data and analysis.

## Notes

- Ensure all required data is placed in the appropriate directories.
- For any issues or questions, please contact the authors of the paper.

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
