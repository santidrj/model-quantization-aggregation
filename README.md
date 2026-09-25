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
     - [src/data/papers/entities.py](src/data/papers/entities.py) & [src/data/papers/knowledge_extraction.py](src/data/papers/knowledge_extraction.py): Define the structure and data extraction logic for the papers analyzed.
     - [src/data/download.py](src/data/download.py): Downloads the list of papers from arXiv and merges them with the Scopus list.
     - [src/data/selection/llm.py](src/data/selection/llm.py): Implements logic for selecting studies using Gemini 3.0 Flash.
     - [dempster_shafer.py](src/dempster_shafer.py) & [belief_assignment.py](src/belief_assignment.py): Combine belief assignments, reproduce Evidence Factory outputs, and emit full audit traces.

3. **Jupyter Notebooks**:
   - Located in the [notebooks](notebooks) directory, these notebooks contain the analysis and visualization of the data.
   - Notebooks include:
     - [1.0-llm-promt-refinement.ipynb](notebooks/1.0-llm-promt-refinement.ipynb): Refines the prompt for LLMs and the selection of LLM.
     - [2.0-model-quantization-paper-selection.ipynb](notebooks/2.0-model-quantization-paper-selection.ipynb): Filters the raw list of papers using the selected GEMINI 3.0.
     - [3.0-final-selection-analysis.ipynb](notebooks/3.0-final-selection-analysis.ipynb): Analyzes the final selection of papers. On the review bar.
     - [4.0-paper-metadata-analysis.ipynb](notebooks/4.0-paper-metadata-analysis.ipynb): Characterizes manually extracted paper metadata. On the review bar.
     - [5.0-evidence-analysis.ipynb](notebooks/5.0-evidence-analysis.ipynb): Reproduces the paper's evidence figures (Fig. 6, Fig. 7, Fig. 8, and Fig. 10). On the review bar.
     - [5.1-subgroup-ptq-w-int8-a-int8.ipynb](notebooks/5.1-subgroup-ptq-w-int8-a-int8.ipynb): Subgroup analysis for `ptq` from `full-fp32` to `w-int8, a-int8`. On the review bar.
     - [6.0-appendix-worked-examples.ipynb](notebooks/6.0-appendix-worked-examples.ipynb): Recomputes manuscript appendix examples. On the review bar.

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
│   ├── 3.0-final-selection-analysis.ipynb
│   ├── 4.0-paper-metadata-analysis.ipynb
│   ├── 5.0-evidence-analysis.ipynb
│   ├── 5.1-subgroup-ptq-w-int8-a-int8.ipynb
│   └── 6.0-appendix-worked-examples.ipynb
├── reports/
│   ├── figures/
│   └── tables/
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
     git clone https://github.com/santidrj/model-quantization-aggregation
     cd model-quantization-aggregation
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

   - **Using Docker** (this version of the replication package):  
     A pre-built image is published on [GitHub Container Registry](https://github.com/users/santidrj/packages/container/model-quantization-aggregation). Pull it (or build locally from the `Dockerfile` if you are changing the package):

     ```bash
     export MQ_IMAGE=ghcr.io/santidrj/model-quantization-aggregation:latest
     docker pull "$MQ_IMAGE"
     ```

     The image for this README revision is also tagged `ghcr.io/santidrj/model-quantization-aggregation:df8ed79` (short git commit). Use that tag when you need a fixed image digest tied to this replication package state.

     Jupyter Lab is the default command and listens on port 8888 with no token:

     ```bash
     docker run -it -p 8888:8888 "$MQ_IMAGE"
     ```

     Notebooks 1.0 and 2.0 call Gemini. Pass a key when you run those notebooks yourself:

     ```bash
     docker run -it -p 8888:8888 \
       -e GEMINI_API_KEY=your_key \
       "$MQ_IMAGE"
     ```

     The review bar needs external paper data mounted at `/app/data/external` and a writable `reports/` directory:

     ```bash
     docker run --rm \
       -v "$(pwd)/data/external:/app/data/external" \
       -v "$(pwd)/reports:/app/reports" \
       "$MQ_IMAGE" \
       mq reproduce review
     ```

     To build the image locally instead of pulling:

     ```bash
     docker build -t ghcr.io/santidrj/model-quantization-aggregation:local .
     export MQ_IMAGE=ghcr.io/santidrj/model-quantization-aggregation:local
     ```

2. **CLI overview**:
   - The project installs an `mq` command via `uv sync`.
   - Use `mq reproduce ...` for deterministic replication workflows.
   - Use `mq papers ...` for paper-maintenance workflows, including live LLM selection.
   - Use `mq extraction ...` for lower-level evidence extraction commands.

3. **Paper data and maintenance**:
   - Refresh the candidate paper catalog from Scopus and arXiv:

     ```bash
     uv run mq papers download
     ```

   - We do not commit external paper data from the selected studies (copyright / size). Most papers expect a local `paper-data.csv` under [data/external](data/external); each paper folder's README explains how to obtain it.
   - Preflight all external study inputs without downloading anything implicitly:

     ```bash
     uv run mq papers ensure-external-data
     ```

   - For papers with a remote archive descriptor (currently Alizadeh and Gonzalez), opt in to fetching missing files:

     ```bash
     uv run mq papers ensure-external-data --download-missing
     ```

   - Run the non-deterministic LLM scoring workflow only with explicit acknowledgement:

     ```bash
     uv run mq papers select --run-llm
     ```

## Reproducing the paper

Pull the published image (see **Using Docker** above), mount external paper data, and mount a writable `reports/` directory:

```bash
export MQ_IMAGE=ghcr.io/santidrj/model-quantization-aggregation:latest
docker pull "$MQ_IMAGE"
docker run --rm \
  -v "$(pwd)/data/external:/app/data/external" \
  -v "$(pwd)/reports:/app/reports" \
  "$MQ_IMAGE" \
  mq reproduce review
```

The same bar on a local checkout, after `uv sync` and the same external paper data:

```bash
uv run mq reproduce review
```

`mq reproduce review` stops if any required external file is missing. It runs these steps, which you can also run one at a time:

1. `uv run mq papers ensure-external-data`
2. `uv run mq extraction run`
3. `uv run mq reproduce figures` — notebook 5.0. Outputs in [reports/figures/](reports/figures/):

   | Output file | Paper figure |
   |---|---|
   | `metrics-usage-distribution.pdf` | Fig. 10 |
   | `correctness-forestplot.pdf` | Fig. 6 |
   | `resource-efficiency-forestplot.pdf` | Fig. 7 |
   | `performance-forestplot.pdf` | Fig. 8 |

4. `uv run mq reproduce tables` — studies-summary fragments in [reports/tables/](reports/tables/) and `data/processed/validated-synthesis.json`.
5. `uv run mq reproduce notebook 3.0`
6. `uv run mq reproduce notebook 4.0`
7. `uv run mq reproduce notebook 5.1`
8. `uv run mq reproduce notebook 6.0`

`mq reproduce notebook` also accepts `5.0`. Notebooks 1.0 and 2.0 stay manual Jupyter runs and need `GEMINI_API_KEY`.

`uv run mq reproduce full-pipeline` still regenerates processed evidence and notebook 5.0 together. Add `--no-notebooks` to stop after processed outputs.

### Other commands

- Run extraction for a subset of papers: `uv run mq extraction run --paper <paper-key> --paper <paper-key>`
- List supported paper keys: `uv run mq papers list`
- Belief-assignment checks: `uv run pytest tests/test_dempster_shafer.py tests/test_belief_assignment.py` (after `uv run mq reproduce tables`)

## Notes

- Ensure all required data is placed in the appropriate directories.
- For any issues or questions, please contact the authors of the paper.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
