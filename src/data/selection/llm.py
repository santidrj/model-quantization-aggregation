from collections.abc import Generator
from enum import IntEnum
import json
import os
from pathlib import Path
import time

import anthropic
from google import genai
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


GEMINI_MODEL = "gemini-2.0-flash-exp"


GEMINI_CONFIG = {
    "temperature": 0,
    "top_p": 0.1,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}


QUERY_CONTEXT = """Assume you are a software engineering researcher conducting a systematic literature review (SLR). Consider the title, abstract, and keywords of the following studies.

Follow these steps:
1. Read the title, abstract, and keywords of the study. Consider that the input structure is as follows:

<BOI>
Title: <Title>
Abstract: <Abstract>
Keywords: <Keywords>
<EOI>

2. Use a 1-7 Likert scale (1 - Strongly disagree, 2 - Disagree, 3 - Somewhat disagree, 4 - Neither agree nor disagree, 5 - Somewhat agree, 6 - Agree, and 7 - Strongly agree) to rate your agreement with the following statement (report only the number): The study is a primary study and not a secondary or tertiary study.
3. Use a 1-7 Likert scale (1 - Strongly disagree, 2 - Disagree, 3 - Somewhat disagree, 4 - Neither agree nor disagree, 5 - Somewhat agree, 6 - Agree, and 7 - Strongly agree) to rate your agreement with the following statement (report only the number): The study regards the application of model quantization to optimize a deep learning (DL) model.
4. Use a 1-7 Likert scale (1 - Strongly disagree, 2 - Disagree, 3 - Somewhat disagree, 4 - Neither agree nor disagree, 5 - Somewhat agree, 6 - Agree, and 7 - Strongly agree) to rate your agreement with the following statement (report only the number): The study regards the environmental sustainability and/or energy efficiency of applying model quantization.
5. Use a 1-7 Likert scale (1 - Strongly disagree, 2 - Disagree, 3 - Somewhat disagree, 4 - Neither agree nor disagree, 5 - Somewhat agree, 6 - Agree, and 7 - Strongly agree) to rate your agreement with the following statement (report only the number): The study analyzes the application of model quantization for model inference and not only for model training or any other purpose.
6. Use a 1-7 Likert scale (1 - Strongly disagree, 2 - Disagree, 3 - Somewhat disagree, 4 - Neither agree nor disagree, 5 - Somewhat agree, 6 - Agree, and 7 - Strongly agree) to rate your agreement with the following statement (report only the number): The study regards the application of model quantization at the software level.
7. Format your agreement ratings for each paper as an array under the paper's title in JSON format.

Provide the results in JSON format, where keys are paper titles and values are arrays of agreement ratings. Ensure accuracy in the JSON structure and do not include any additional text or information.

Here is an example of how a paper should be rated:

<BOI>
Title: A Reconfigurable Approximate Multiplier for Quantized CNN Applications
Abstract: Quantized CNNs, featured with different bit-widths at different layers, have been widely deployed in mobile and embedded applications. The implementation of a quantized CNN may have multiple multipliers at different precisions with limited resource reuse or one multiplier at higher precision than needed causing area overhead. It is then highly desired to design a multiplier by accounting for the characteristics of quantized CNNs to ensure both flexibility and energy efficiency. In this work, we present a reconfigurable approximate multiplier to support multiplications at various precisions, i.e., bit-widths. Moreover, unlike prior works assuming uniform distribution with bit-wise independence, a quantized CNN may have centralized weight distribution and hence follow a Gaussian-like distribution with correlated adjacent bits. Thus, a new block-based approximate adder is also proposed as part of the multiplier to ensure energy efficient operation with awareness of bit-wise correlation. Our experimental results show that the proposed adder significantly reduces the error rate by 76-98% over a state-of-the-art approximate adder for such scenarios. Moreover, with the deployment of the proposed multiplier, which is 17% faster and 22% more power saving than a Xilinx multiplier IP at the same precision, a quantized CNN implemented in FPGA achieves 17% latency reduction and 15% power saving compared with a full precision case. © 2020 IEEE.
Keywords: None
<EOI>

Should be rated as: {"A Reconfigurable Approximate Multiplier for Quantized CNN Applications": [7, 6, 7, 5, 3]}

The reasoning behind the ratings is as follows:
1. The study proposes a concrete solution.
2. The study focuses on model quantization, but maybe too much on the arithmetic part.
3. The study reports the energy consumption of quantizing a CNN.
4. The study reports results of deploying a quantized CNN suggesting the use of quantization for inference.
5. It looks like it, but it implements "a new block-based approximate adder" which looks closer to hardware than to software.

Here is a second example:

<BOI>
Title: Energy Adaptive Convolution Neural Network Using Dynamic Partial Reconfiguration
Abstract: Convolutional Neural Network (CNN) is a good candidate for computer vision applications. CNN is well known for its great classification accuracy at image recognition tasks. The cost of CNN is its large power consumption as it needs a lot of multiplication and addition operations. Approximation can reduce the power consumption. CNNs can be implemented by CPUs, GPUs or FPGAs. In this paper, the proposed CNN is implemented on Xilinx XC7Z020 FPGA and is trained to recognize MNIST dataset This CNN is approximated through quantization which reduces the accuracy only by 0.53% while using 7-bits for the implementation. A reduction of 2.7X is achieved in energy consumption compared to the conventional design which uses 16-bits. Dynamic Partial Reconfiguration (DPR) reconfigures the FPGA during the run time with appropriate power consumption design if the battery level decreases. This enables recognition applications to run with low power budget but with sacrificing minor accuracy instead of termination. © 2020 IEEE.
Keywords: Approximate Computing; Convolutional Neural Network; DPR; MNIST; Precision Scaling
<EOI>

Should be rated as: {"Energy Adaptive Convolution Neural Network Using Dynamic Partial Reconfiguration": [7, 7, 7, 6, 4]}

The reasoning behind the ratings is as follows:
1. The study analyzes the effect of Dynamic Partial Reconfiguration on energy consumption when used for CNN applications.
2. The study uses model quantization to approximate a CNN.
3. The study reports reduction in energy consumption when using a quantized CNN.
4. It looks like it uses quantization for inference. It mentions "run time" and "low power budget" rather than inference.
5. The study reports the use of model quantization, but it is unclear how it is applied.

Here is a third example:

<BOI>
Title: Impact of ML Optimization Tactics on Greener Pre-Trained ML Models
Abstract: Background: Given the fast-paced nature of today's technology, which has surpassed human performance in tasks like image classification, visual reasoning, and English understanding, assessing the impact of Machine Learning (ML) on energy consumption is crucial. Traditionally, ML projects have prioritized accuracy over energy, creating a gap in energy consumption during model inference. Aims: This study aims to (i) analyze image classification datasets and pre-trained models, (ii) improve inference efficiency by comparing optimized and non-optimized models, and (iii) assess the economic impact of the optimizations. Method: We conduct a controlled experiment to evaluate the impact of various PyTorch optimization techniques (dynamic quantization, torch.compile, local pruning, and global pruning) to 42 Hugging Face models for image classification. The metrics examined include GPU utilization, power and energy consumption, accuracy, time, computational complexity, and economic costs. The models are repeatedly evaluated to quantify the effects of these software engineering tactics. Results: Dynamic quantization demonstrates significant reductions in inference time and energy consumption, making it highly suitable for large-scale systems. Additionally, torch.compile balances accuracy and energy. In contrast, local pruning shows no positive impact on performance, and global pruning's longer optimization times significantly impact costs. Conclusions: This study highlights the role of software engineering tactics in achieving greener ML models, offering guidelines for practitioners to make informed decisions on optimization methods that align with sustainability goals.
Keywords: None
<EOI>

Should be rated as: {"Impact of ML Optimization Tactics on Greener Pre-Trained ML Models": [7, 6, 7, 7, 7]}

The reasoning behind the ratings is as follows:
1. The study is a controlled expriment.
2. The study analyzes "various PyTorch optimization techniques (dynamic quantization, torch.compile, local pruning, and global pruning)" which includes model quantization.
3. The study reports "Dynamic quantization demonstrates significant reductions in inference time and energy consumption".
4. The study explicitly mentions inference time and efficiency.
5. The study analyzes various optimization techniques implemented in PyTorch.

Here is a fourth example:

<BOI>
Title: An Efficient Deep Learning Framework for Low Rate Massive MIMO CSI Reporting
Abstract: Channel state information (CSI) reporting is important for multiple-input multiple-output (MIMO) wireless transceivers to achieve high capacity and energy efficiency in frequency division duplex (FDD) mode. CSI reporting for massive MIMO systems could consume large bandwidth and degrade spectrum efficiency. Deep learning (DL)-based CSI reporting integrated with channel characteristics has demonstrated success in improving CSI compression and recovery. To further improve the encoding efficiency of CSI feedback, we develop an efficient DL-based compression framework CQNet to jointly tackle CSI compression, codeword quantization, and recovery under the bandwidth constraint. CQNet is directly compatible with other DL-based CSI feedback works for further enhancement. We propose a more efficient quantization scheme in the radial coordinate by introducing a novel magnitude-adaptive phase quantization framework. Compared with traditional CSI reporting, CQNet demonstrates superior CSI feedback efficiency and better CSI reconstruction accuracy.  © 2020 IEEE.
Keywords: CSI feedback; deep learning; FDD; Massive MIMO; quantization
<EOI>

Should be rated as: {"An Efficient Deep Learning Framework for Low Rate Massive MIMO CSI Reporting": [7, 2, 2, 3, 1]}

The reasoning behind the ratings is as follows:
1. The study proposes a new framework.
2. Codeword quanitization seems not to be directly related to ML models (although I'm not sure if the concepts can be linked; but it seems not).
3. It talks about encoding efficiency, which eventually could be linked to environmental sustainability (?).
4. The study doesn't report using model quantization for model inference.
5. It uses hardware-level jargon in all the abstract.

You must provide all five ratings for each paper, do not miss any. Remember that each paper is identified by its title. ENSURE YOU PICK THE PAPER'S TITLE CORRECTLY. Think step by step. THINK CAREFULLY YOUR RATING IN STEPS 2-6 AND CONSIDER EACH PAPER INDIVIDUALLY.

The list of papers you must process starts below. ENSURE YOU PROCESS ALL THE PAPERS. LIMIT TO REPORT THE ASKED INFORMATION ONLY. DO NOT INCLUDE ANY ADDITIONAL TEXT OR INFORMATION.
"""  # noqa: E501


def gemini_query(client: genai.Client, query: str, json_file: str | os.PathLike[str] | None = None) -> dict:
    """
    Query a Gemini model. If the query fails due to rate limiting, the function will wait 60 seconds before retrying.

    When a json file is provided, the results are saved to the file.

    Parameters
    ----------
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


def claude_query(client: anthropic.Anthropic, query: str, json_file: str | os.PathLike[str] | None = None) -> dict:
    """
    Query a Claude model. If the query fails due to rate limiting, the function will wait 60 seconds before retrying.

    When a json file is provided, the results are saved to the file.

    Parameters
    ----------
    client : anthropic.Anthropic
        The Claude client.
    query : str
        The query message.
    json_file : str, optional
        The path to a json file to save the results to, by default None.

    Returns
    -------
    dict
        The response from the Claude model.
    """

    query_completed = False

    while not query_completed:
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query,
                            }
                        ],
                    }
                ],
            )
            json_response = message.to_json()
            response = json.loads(json_response)["content"][0]["text"]
            query_completed = True
        except Exception as e:
            if "429" in str(e):
                print("Requests per minute rate limit exceeded, waiting 60 seconds to retry...")
                time.sleep(60)
            else:
                raise e
    json_response = json.loads(response)

    if json_file:
        with open(json_file, "w") as f:
            json.dump(json_response, f, indent=2)

    return json_response


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
    return (
        f"<BOI>\nTitle: {paper['Title']}\nAbstract: {paper['Abstract']}\n"
        f"Keywords: {paper.get('Author Keywords', '')}\n<EOI>"
    )


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
            "IC5": 0.0
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
        column_names=["IC1", "IC2", "IC3", "IC4", "IC5"],
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
        (pl.col("IC1") < LikertScale.NEITHER_AGREE_NOR_DISAGREE)
        | (pl.col("IC2") < LikertScale.NEITHER_AGREE_NOR_DISAGREE)
        | (pl.col("IC3") < LikertScale.NEITHER_AGREE_NOR_DISAGREE)
        | (pl.col("IC4") < LikertScale.NEITHER_AGREE_NOR_DISAGREE)
        | (pl.col("IC5") < LikertScale.NEITHER_AGREE_NOR_DISAGREE)
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
        (pl.col("IC1") > LikertScale.NEITHER_AGREE_NOR_DISAGREE)
        & (pl.col("IC2") > LikertScale.NEITHER_AGREE_NOR_DISAGREE)
        & (pl.col("IC3") > LikertScale.NEITHER_AGREE_NOR_DISAGREE)
        & (pl.col("IC4") > LikertScale.NEITHER_AGREE_NOR_DISAGREE)
        & (pl.col("IC5") > LikertScale.NEITHER_AGREE_NOR_DISAGREE)
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

    return paper_scores.filter(~pl.col("Title").is_in(processed_papers.get_column("Title")))


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

    missing_columns = set(["Title", "Abstract", "Author Keywords"]) - set(papers.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    relevant_data = papers.select([pl.col("Title"), pl.col("Abstract"), pl.col("Author Keywords")])

    papers_context_message = "\n\n".join(create_paper_context_message(paper) for paper in relevant_data.to_dicts())

    return f"{QUERY_CONTEXT}\n\n{papers_context_message}"


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

    missing_columns = set(["Title", "Abstract", "Author Keywords"]) - set(papers.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    relevant_data = papers.select([pl.col("Title"), pl.col("Abstract"), pl.col("Author Keywords")])

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
