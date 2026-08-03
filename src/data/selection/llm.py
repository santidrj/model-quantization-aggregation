from collections.abc import Generator
from enum import IntEnum
import json
import os
from pathlib import Path
import time

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
import polars as pl

from src.config import INTERIM_DATA_DIR


class LikertScale(IntEnum):
    STRONGLY_DISAGREE = 1
    DISAGREE = 2
    SOMEWHAT_DISAGREE = 3
    NEITHER_AGREE_NOR_DISAGREE = 4
    SOMEWHAT_AGREE = 5
    AGREE = 6
    STRONGLY_AGREE = 7


GEMINI_MODEL = "gemini-3-flash-preview"


GEMINI_CONFIG = GenerateContentConfig(
    temperature=1.0,
    top_p=0.1,
    top_k=40,
    thinking_config=ThinkingConfig(thinking_level="high"),
    max_output_tokens=8192,
    response_mime_type="application/json",
)

REQUIRED_QUERY_COLUMNS = ("Title", "Abstract", "Author Keywords")
INCLUSION_CRITERIA_COLUMNS = ("IC1", "IC2", "IC3", "IC4", "IC5")

QUERY_CONTEXT = """**Role:** You are an expert Software Engineering Researcher conducting a Systematic Literature Review (SLR) on "Resource-Efficient Deep Learning via Quantization."

**Objective:** Evaluate research papers to determine if they provide empirical, software-level evidence of the impact of model quantization on resource efficiency (energy, memory, storage) or performance (latency) during inference.

---

### 1. The Evidence-Based Rating Scale (1-7)

For each criterion, assign a score based on the level of evidence found in the Title, Abstract, or Keywords:

* **7 (Strongly Agree):** Evidence is **explicit and numerical** (e.g., "30% energy reduction," "reduced from 120MB to 12MB," "5x speedup").
* **6 (Agree):** Evidence is **explicit but non-numerical** (e.g., "compared to full-precision baseline," "measured on a Jetson Nano").
* **4-5 (Somewhat Agree):** Evidence is **qualitative/vague** (e.g., "energy-efficient approach," "fast inference," "lightweight model") without mentioning specific metrics or baselines.
* **1-3 (Disagree):** The information is **absent, contradictory, or refers to a different domain** (e.g., hardware circuit design, signal processing).

---

### 2. The 6 Evaluation Criteria

1. **[Primary Study]:** Is this an original empirical experiment? (7 = clear experiment; 1 = survey, review, or vision paper).
2. **[DL Quantization]:** Does the study quantize a Deep Learning model (weights/activations/biases)? (7 = explicitly mentions DL model quantization; 1 = signal/MIMO quantization or JPEG/video compression).
3. **[Software-Level Focus]:** Is the primary contribution an algorithm, flow, or software framework?
    * **Rate 6-7:** If implemented in software (PyTorch, TensorFlow, etc.) or as an algorithmic optimization.
    * **Rate 1-3 (REJECT):** If the primary novelty is a **physical hardware component** (e.g., a new 40nm CMOS circuit, SRAM design, ASIC multiplier, or TCI interface).

4. **[Inference Phase]:** Is the study focused on the deployment/inference phase? (7 = mentions "inference," "edge deployment," or "real-time execution").
5. **[Efficiency Metrics]:** Does the study report Energy, Latency, RAM, or Model Size?
    * **Rate 7:** If numerical data/percentages are provided.
    * **Rate 4-5:** If it claims "efficiency" or "speed" but provides no numbers.


6. **[Controlled Comparison]:** Does the study compare the quantized model to a non-quantized baseline?
    * **Rate 6-7:** If it mentions "vs FP32," "vs full precision," or "compared to the original model."
    * **Rate 1-3:** If it only compares one quantized method against a different quantized method.

---

### 3. Strict Red Flags (Forcing Low Scores)

* **Signal Processing:** If the keywords involve "CSI," "MIMO," or "Channel Estimation," **Rate Criterion 2 as 1.**
* **Hardware Architecture:** If the abstract focuses on "Transistors," "SRAM stacking," "Clock frequency," or "Circuit Area," **Rate Criterion 3 as 1-2.**
* **Training-Only:** If the focus is "Faster training" or "Convergence rate" without inference metrics, **Rate Criterion 4 as 1.**

---

### 4. Output Instructions

* **Format:** Provide a single JSON object.
* **Structure:** `{"Paper Title": [C1, C2, C3, C4, C5, C6]}`
* **No extra text:** Do not provide reasoning, introductions, or summaries.

---

### 5. Input Data Structure

Process the following study:

Title: [INSERT TITLE]
Abstract: [INSERT ABSTRACT]
Keywords: [INSERT KEYWORDS]

---

### 6. Examples for Calibration

*(Use these to understand the logic, but do not include the reasoning in your final output)*

**Input:**
Title: A Reconfigurable Approximate Multiplier for Quantized CNN Applications
Abstract: Quantized CNNs, featured with different bit-widths at different layers, have been widely deployed in mobile and embedded applications. The implementation of a quantized CNN may have multiple multipliers at different precisions with limited resource reuse or one multiplier at higher precision than needed causing area overhead. It is then highly desired to design a multiplier by accounting for the characteristics of quantized CNNs to ensure both flexibility and energy efficiency. In this work, we present a reconfigurable approximate multiplier to support multiplications at various precisions, i.e., bit-widths. Moreover, unlike prior works assuming uniform distribution with bit-wise independence, a quantized CNN may have centralized weight distribution and hence follow a Gaussian-like distribution with correlated adjacent bits. Thus, a new block-based approximate adder is also proposed as part of the multiplier to ensure energy efficient operation with awareness of bit-wise correlation. Our experimental results show that the proposed adder significantly reduces the error rate by 76-98% over a state-of-the-art approximate adder for such scenarios. Moreover, with the deployment of the proposed multiplier, which is 17% faster and 22% more power saving than a Xilinx multiplier IP at the same precision, a quantized CNN implemented in FPGA achieves 17% latency reduction and 15% power saving compared with a full precision case. © 2020 IEEE.
Keywords: None

**Output:** `{"A Reconfigurable Approximate Multiplier for Quantized CNN Applications": [7, 6, 7, 5, 3]}`

**Input:**
Title: Energy Adaptive Convolution Neural Network Using Dynamic Partial Reconfiguration
Abstract: Convolutional Neural Network (CNN) is a good candidate for computer vision applications. CNN is well known for its great classification accuracy at image recognition tasks. The cost of CNN is its large power consumption as it needs a lot of multiplication and addition operations. Approximation can reduce the power consumption. CNNs can be implemented by CPUs, GPUs or FPGAs. In this paper, the proposed CNN is implemented on Xilinx XC7Z020 FPGA and is trained to recognize MNIST dataset This CNN is approximated through quantization which reduces the accuracy only by 0.53% while using 7-bits for the implementation. A reduction of 2.7X is achieved in energy consumption compared to the conventional design which uses 16-bits. Dynamic Partial Reconfiguration (DPR) reconfigures the FPGA during the run time with appropriate power consumption design if the battery level decreases. This enables recognition applications to run with low power budget but with sacrificing minor accuracy instead of termination. © 2020 IEEE.
Keywords: Approximate Computing; Convolutional Neural Network; DPR; MNIST; Precision Scaling

**Output:** `{"Energy Adaptive Convolution Neural Network Using Dynamic Partial Reconfiguration": [7, 7, 7, 6, 4]}`

**Input:**
Title: Impact of ML Optimization Tactics on Greener Pre-Trained ML Models
Abstract: Background: Given the fast-paced nature of today's technology, which has surpassed human performance in tasks like image classification, visual reasoning, and English understanding, assessing the impact of Machine Learning (ML) on energy consumption is crucial. Traditionally, ML projects have prioritized accuracy over energy, creating a gap in energy consumption during model inference. Aims: This study aims to (i) analyze image classification datasets and pre-trained models, (ii) improve inference efficiency by comparing optimized and non-optimized models, and (iii) assess the economic impact of the optimizations. Method: We conduct a controlled experiment to evaluate the impact of various PyTorch optimization techniques (dynamic quantization, torch.compile, local pruning, and global pruning) to 42 Hugging Face models for image classification. The metrics examined include GPU utilization, power and energy consumption, accuracy, time, computational complexity, and economic costs. The models are repeatedly evaluated to quantify the effects of these software engineering tactics. Results: Dynamic quantization demonstrates significant reductions in inference time and energy consumption, making it highly suitable for large-scale systems. Additionally, torch.compile balances accuracy and energy. In contrast, local pruning shows no positive impact on performance, and global pruning's longer optimization times significantly impact costs. Conclusions: This study highlights the role of software engineering tactics in achieving greener ML models, offering guidelines for practitioners to make informed decisions on optimization methods that align with sustainability goals.
Keywords: None

**Output:** `{"Impact of ML Optimization Tactics on Greener Pre-Trained ML Models": [7, 6, 7, 7, 7]}`

**Input:**
Title: An Efficient Deep Learning Framework for Low Rate Massive MIMO CSI Reporting
Abstract: Channel state information (CSI) reporting is important for multiple-input multiple-output (MIMO) wireless transceivers to achieve high capacity and energy efficiency in frequency division duplex (FDD) mode. CSI reporting for massive MIMO systems could consume large bandwidth and degrade spectrum efficiency. Deep learning (DL)-based CSI reporting integrated with channel characteristics has demonstrated success in improving CSI compression and recovery. To further improve the encoding efficiency of CSI feedback, we develop an efficient DL-based compression framework CQNet to jointly tackle CSI compression, codeword quantization, and recovery under the bandwidth constraint. CQNet is directly compatible with other DL-based CSI feedback works for further enhancement. We propose a more efficient quantization scheme in the radial coordinate by introducing a novel magnitude-adaptive phase quantization framework. Compared with traditional CSI reporting, CQNet demonstrates superior CSI feedback efficiency and better CSI reconstruction accuracy.  © 2020 IEEE.
Keywords: CSI feedback; deep learning; FDD; Massive MIMO; quantization

**Output:** `{"An Efficient Deep Learning Framework for Low Rate Massive MIMO CSI Reporting": [7, 2, 2, 3, 1]}`

---

### Papers to Process

"""  # noqa: E501


def gemini_query(client: genai.Client, query: str, json_file: str | os.PathLike[str] | None = None) -> dict:
    """
    Query a Gemini model. If the query fails due to rate limiting, the function will wait 60 seconds before retrying.

    When a json file is provided, the results are saved to the file.

    Parameters
    ----------
    client : genai.Client
        The Gemini client.
    client : genai.Client
        The Gemini client.
    query : str
        The query message.
    json_file : str, optional
        The path to a json file to save the results to, by default None.

    Returns
    -------
    dict
        The response from the Gemini model.
    """

    query_completed = False
    while not query_completed:
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=query, config=GEMINI_CONFIG)
            query_completed = True
        except Exception as e:
            if "429" in str(e):
                print("Requests per minute rate limit exceeded, waiting 60 seconds to retry...")
                time.sleep(60)
            else:
                raise e

    json_response = json.loads(response.text)

    if json_file:
        with open(json_file, "w") as f:
            json.dump(json_response, f, indent=2)

    return json_response


def _require_query_columns(papers: pl.DataFrame) -> None:
    missing_columns = set(REQUIRED_QUERY_COLUMNS) - set(papers.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")


def _build_context_message(papers: pl.DataFrame) -> str:
    relevant_data = papers.select([pl.col(column_name) for column_name in REQUIRED_QUERY_COLUMNS])
    return "\n\n".join(create_paper_context_message(paper) for paper in relevant_data.to_dicts())


def _criteria_filter(operator: str) -> pl.Expr:
    threshold = pl.lit(LikertScale.NEITHER_AGREE_NOR_DISAGREE)
    expressions = [getattr(pl.col(column_name), operator)(threshold) for column_name in INCLUSION_CRITERIA_COLUMNS]
    return pl.any_horizontal(*expressions) if operator == "__lt__" else pl.all_horizontal(*expressions)


def gemini_batched_query(client: genai.Client, batch_number: int, query: str) -> pl.DataFrame:
    """
    Query a Gemini model in batches.

    The results are saved to a parquet file in the interim data directory.

    Parameters
    ----------
    client : genai.Client
        The Gemini client.
    batch_number : int
        The batch number.
    query : str
        The query message.

    Returns
    -------
    pl.DataFrame
        The results of the query in a polars DataFrame.
    """
    results = gemini_query(client, query)
    result_df = pl.from_dict(results).transpose(
        include_header=True,
        header_name="Title",
        column_names=["IC1", "IC2", "IC3", "IC4", "IC5"],
    )
    result_df.write_parquet(INTERIM_DATA_DIR / f"{GEMINI_MODEL}-batch-{batch_number}-results.parquet")
    return result_df


def combine_llm_scores(llm_scores: list[pl.DataFrame]) -> pl.DataFrame:
    """
    Combine the scores of the LLMs. The scores are concatenated and then the mean of the inclusion criteria
    is calculated. The final scores are rounded to the nearest integer.

    Parameters
    ----------
    llm_scores : list[pl.DataFrame]
        The scores of the LLMs.

    Returns
    -------
    pl.DataFrame
        The joined scores.
    """
    return pl.concat(llm_scores).group_by("Title").agg(pl.mean(r"^IC\d+$").round().cast(pl.Int8)).sort("Title")


def create_paper_context_message(paper: dict) -> str:
    return f"Title: {paper['Title']}\nAbstract: {paper['Abstract']}\nKeywords: {paper.get('Author Keywords', '')}\n"


def read_llm_output(json_path: str | Path) -> pl.DataFrame:
    """
    Read the output of an LLM in json format.

    The json file should have the following structure:

    {
        "Paper Title": {
            "IC1": 0.0,
            "IC2": 0.0,
            "IC3": 0.0,
            "IC4": 0.0,
            "IC5": 0.0,
            "IC6": 0.0,
        },
        ...
    }

    Parameters
    ----------
    json_path : str
        The path to the LLM output json file.

    Returns
    -------
    pl.DataFrame
        The LLM output.
    """
    with open(json_path) as f:
        results = json.load(f)

    results = {k: v for k, v in results.items()}
    return pl.from_dict(results).transpose(
        include_header=True,
        header_name="Title",
        column_names=["IC1", "IC2", "IC3", "IC4", "IC5", "IC6"],
    )


def get_excluded_papers(paper_scores: pl.DataFrame) -> pl.DataFrame:
    """
    Get the papers that were excluded based on the inclusion criteria.

    Papers with a rating score lower than 4 (neither agree nor disagree)
    in any of the inclusion criteria are excluded.

    Parameters
    ----------
    paper_scores : pl.DataFrame
        The paper scores assigned by the LLM(s).

    Returns
    -------
    pl.DataFrame
        The excluded papers.
    """

    return paper_scores.filter(
        _criteria_filter("__lt__")
    )


def get_included_papers(paper_scores: pl.DataFrame) -> pl.DataFrame:
    """
    Get the papers that were included based on the inclusion criteria.

    Papers with a rating score above 4 (neither agree nor disagree) in all of the inclusion criteria are included.

    Parameters
    ----------
    paper_scores : pl.DataFrame
        The paper scores assigned by the LLM(s).

    Returns
    -------
    pl.DataFrame
        The included papers.
    """

    return paper_scores.filter(
        _criteria_filter("__gt__")
    )


def get_manual_review_papers(
    paper_scores: pl.DataFrame, excluded_papers: pl.DataFrame | None = None, included_papers: pl.DataFrame | None = None
) -> pl.DataFrame:
    """
    Get the papers that require manual review based on the inclusion criteria.

    Parameters
    ----------
    paper_scores : pl.DataFrame
        The paper scores assigned by the LLM(s).
    excluded_papers : pl.DataFrame, optional
        The papers that were excluded based on the inclusion criteria, by default None.
    included_papers : pl.DataFrame, optional
        The papers that were included based on the inclusion criteria, by default None.

    Returns
    -------
    pl.DataFrame
        The papers that require manual review.
    """

    if excluded_papers is None:
        excluded_papers = get_excluded_papers(paper_scores)

    if included_papers is None:
        included_papers = get_included_papers(paper_scores)

    processed_papers = pl.concat([excluded_papers, included_papers])

    processed_titles = processed_papers.select(pl.col("Title").implode()).to_series()
    return paper_scores.filter(~pl.col("Title").is_in(processed_titles))


def assign_inclusion(paper_scores: pl.DataFrame, conservative=True) -> pl.DataFrame:
    """
    Assign the inclusion status to the papers based on the inclusion criteria.

    Parameters
    ----------
    paper_scores : pl.DataFrame
        The paper scores assigned by the LLM(s).
    conservative : bool, optional
        Whether to use a conservative approach to mark papers as included, by default True.
        When True, all papers with a score >3 in all inclusion criteria are marked as included. Otherwise, only papers
        with a score >4 are marked as included.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with the inclusion status assigned in the "Included" column.
    """
    excluded_papers = get_excluded_papers(paper_scores)
    included_papers = get_included_papers(paper_scores)
    manual_review_papers = get_manual_review_papers(paper_scores, excluded_papers, included_papers)

    manual_review_label = "y" if conservative else "n"
    return pl.concat(
        [
            excluded_papers.with_columns(pl.lit("n").alias("Included")),
            included_papers.with_columns(pl.lit("y").alias("Included")),
            manual_review_papers.with_columns(pl.lit(manual_review_label).alias("Included")),
        ]
    )


def build_query(papers: pl.DataFrame) -> str:
    """
    Build the query message for the papers.

    Parameters
    ----------
    papers : pl.DataFrame
        The papers to build the query for.

    Returns
    -------
    str
        The query message.

    Raises
    ------
    ValueError
        If the DataFrame does not have the columns 'Title', 'Abstract', and 'Author Keywords'.
    """

    _require_query_columns(papers)
    return f"{QUERY_CONTEXT}\n\n{_build_context_message(papers)}"


def build_batched_query(papers: pl.DataFrame, batch_size: int) -> Generator[str, None, None]:
    """
    Returns a generator of query messages for the papers.

    Parameters
    ----------
    papers : pl.DataFrame
        The papers to build the query for.
    batch_size : int
        The batch size.

    Yields
    ------
    str
        The query message.
    """
    _require_query_columns(papers)
    relevant_data = papers.select([pl.col(column_name) for column_name in REQUIRED_QUERY_COLUMNS])

    for i in range(0, len(relevant_data), batch_size):
        batch = relevant_data.slice(i, batch_size)
        papers_context_message = "\n\n".join(create_paper_context_message(paper) for paper in batch.to_dicts())
        yield f"{QUERY_CONTEXT}\n\n{papers_context_message}"


def simplify_inclusion_results(inclusion_results: pl.DataFrame) -> pl.DataFrame:
    """
    Simplify the inclusion results DataFrame for evaluation purposes.

    All papers marked as "m" (may be included) by the LLM will have to be reviewed manually, hence we mark them as "y"
    for evaluation purposes.

    Parameters
    ----------
    inclusion_results : pl.DataFrame
        The inclusion results DataFrame.

    Returns
    -------
    pl.DataFrame
        The simplified inclusion results DataFrame.
    """
    return inclusion_results.with_columns(
        pl.when(pl.col("Included") == "m").then(pl.lit("y")).otherwise(pl.col("Included")).alias("Included"),
    ).with_columns(
        (pl.col("Included") == "y").alias("Included"),
        (pl.col("Manually Included") == "y").alias("Manually Included"),
    )
