# Interpretation Guide for the Quality Evaluation Questionnaire for Experimental Studies

This guide is intended to support consistent application of the questionnaire in systematic evidence synthesis, including by multiple human reviewers and LLM-based reviewers. It should be applied conservatively.

Although the original rubric was designed for respondent-based experimental studies, this version is adapted for artifact-based data strategy studies on deep learning model quantization. In this review context, the primary studies do not involve human subjects. As a result, some items should be reinterpreted in terms of models, datasets, benchmarks, hardware platforms, baselines, and evaluation pipelines, while a smaller set of items should legitimately be scored `Not applicable`.

## Adaptation to quantization studies

Use the following reinterpretations consistently throughout the assessment:

- "experimental units" means the units analyzed or compared in the study, such as models, datasets, tasks, benchmarks, hardware targets, layers, or quantization configurations.
- "subjects" should usually be read as the evaluated artifacts or data sources, not people. Only use the human-subject interpretation if the study truly includes participants.
- "treatments" means the compared quantization methods, calibration strategies, compression pipelines, or baseline methods.
- "controls" means baseline models, full-precision references, prior quantization methods, ablated variants, or standard evaluation settings.
- "skills and experience of subjects" should usually be read as relevant characteristics of the artifacts, such as model family, pretraining status, architecture scale, dataset composition, or hardware constraints. If the wording cannot be meaningfully reinterpreted, use `Not applicable`.
- "data collection procedures" includes how evaluation data, benchmark outputs, calibration data, profiling data, and measurement logs were produced and validated.
- "quality control methods" includes sanity checks, repeated runs, seed control, calibration protocol checks, hardware profiling controls, benchmark validation, and verification of numerical correctness.
- "drop-outs" should be interpreted as missing runs, failed experiments, excluded models, excluded datasets, or discarded measurements when that is relevant.
- "experimenter bias" should be interpreted narrowly and conservatively. Use it only where the item can meaningfully be translated to artifact-based comparisons; otherwise prefer `Not applicable`.

This adaptation should not be used to rescue missing reporting. A question should be reinterpreted only when the underlying methodological intent still applies in a quantization-study setting.

## General scoring principles

1. Score only what is explicitly reported in the paper or in material directly cited by the paper as part of the study report.
2. Do not infer methodological quality from reputation, venue, common practice, or what the authors probably did.
3. If a criterion requires reporting and the information is missing, unclear, or only weakly implied, score `No`.
4. Use `Not applicable` only when the criterion genuinely does not apply to the study design or reporting situation after reasonable reinterpretation for quantization studies. It should not be used as a substitute for missing information.
5. When a question contains multiple elements, score `Yes` only if the paper addresses the core required element well enough for another reviewer to verify it.
6. In line with common empirical software engineering quality assessment practice, the focus is primarily on adequacy of reporting, not on whether the study made the best possible design choice.
7. For quantization studies, favor a methodological reinterpretation before assigning `Not applicable`, but only when the item's intent remains meaningful for model-, dataset-, or hardware-based evaluation.

## Question 1.1

### Intent
Assesses whether the study states the research questions that motivate the experiment and define what the study is trying to learn.

### What qualifies as Yes
The paper explicitly states one or more research questions, goals framed as research questions, or equivalent study questions tied to the experimental evaluation.

Sufficient:
- Numbered research questions.
- Explicit statements such as "We investigate whether..." or "The study addresses the following questions..." when these clearly function as research questions.

Insufficient:
- A vague motivation section with no explicit evaluative question.
- Only a generic objective such as "to evaluate the approach" without specifying what is being examined.

### What qualifies as No
Mark `No` when the paper does not explicitly state research questions or equivalent evaluative questions. Missing information should be scored `No`.

### When Not applicable should be used
Almost never. Experimental studies are expected to have explicit evaluative questions even if hypotheses are not formalized.

### Examples of acceptable evidence
- "RQ1 How does the energy consumption of models vary across typical software-related tasks?"
- "The primary goal of this paper is to determine whether there are Pareto-optimal quantization strategies, when considering various hardware and software factors that affect the accuracy-cost trade-off."

### Examples that are not sufficient
- "We present an evaluation of our method."
- "The goal is to improve software quality."

### Common ambiguities
Some papers use "objective," "aim," or "goal" instead of "research question." Treat these as sufficient only if they clearly specify what relationship, difference, or effect is being examined.

### Decision rule
Score `Yes` only if the paper explicitly states what empirical question the experiment is meant to answer. If the reader must infer the question from the motivation or results, score `No`.

## Question 1.2

### Intent
Assesses whether the study states testable hypotheses and links them to an underlying rationale or theory.

### What qualifies as Yes
The paper explicitly reports hypotheses, expectations, or predicted relationships and provides some rationale for them.

Sufficient:
- Null and alternative hypotheses.
- Directional hypotheses with justification from prior theory, literature, or mechanism.

Insufficient:
- Reporting only research questions.
- Stating an expectation without explaining why.

### What qualifies as No
Mark `No` when no explicit hypotheses are stated, or when hypotheses are stated without any underlying rationale. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` only if the study is explicitly exploratory and does not frame the analysis as hypothesis testing. This should be used sparingly.

### Examples of acceptable evidence
- "In addition, we propose that a small portion of the benefits achieved when using lower precisions can be forfeited to increase the network size and therefore the accuracy."
- "Low-Bit Neural Network (LBNN) is a promising technique... Although LBNN has the advantages of low memory usage... Low-bit design requires additional computation units and may cause large accuracy drop."

### Examples that are not sufficient
- "We expected the tool to help."
- "The study compares two techniques."

### Common ambiguities
In empirical software engineering, many experiments state RQs but not formal hypotheses. For this item, RQs alone are not enough. A theory can be modest, such as literature-based reasoning, but it must be explicit.

### Decision rule
Score `Yes` only if the paper explicitly states one or more hypotheses or predicted effects and gives a stated rationale for them. Otherwise score `No`, unless the paper is clearly exploratory, in which case `Not applicable` may be used.

## Question 2.1

### Intent
Assesses whether the paper describes the application domain or industry context in which the studied software or tasks are situated.

### What qualifies as Yes
The paper explicitly identifies the industry, domain, or application area relevant to the study setting or artifacts.

### What qualifies as No
Mark `No` when the study context is described only generically, such as "industrial project" or "real-world system," without naming the domain. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` when there is no industry context to report, such as classroom-only studies using synthetic tasks detached from any domain.

### Examples of acceptable evidence
- "The goal of this study is to investigate the energy consumption of using LLMs during the inference phase in typical software development tasks, namely code generation, bug fixing, docstring generation, and test case generation."
- "Regarding the context of this research, we focus on a specific ML task, namely image classification. This task involves categorizing an image into one of predefined classes. The choice is justified by its widespread use in research [33] [57] and its significance in practical applications, ranging from medical imaging to facial recognition."

### Examples that are not sufficient
- "The task was based on a real application."
- "An industrial case was used."

### Common ambiguities
If the paper identifies only a technical area, such as "compiler optimization," that may still count as context if it meaningfully describes the product domain. Do not require fine-grained business detail.

### Decision rule
Score `Yes` if the application or industry domain is explicitly identified. If no usable domain information is given, score `No`; use `Not applicable` only when no such context exists.

## Question 2.2

### Intent
Assesses whether the organizational setting is described well enough to understand the development environment.

### What qualifies as Yes
The paper describes the nature of the organization when an organization exists and matters to the study.

Sufficient:
- In-house department, vendor, consultancy, open-source community, startup, large enterprise.
- Organizational role in development or deployment.

### What qualifies as No
Mark `No` when the study is situated in an organization but the paper gives no meaningful description of that organizational setting. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` when there is no relevant development organization, for example a purely academic lab study with student participants and no organizational setting.

### Examples of acceptable evidence
- "Air Force Research Laboratory Information Directorate."
- "The authors are with the University Center for Research in Space Technologies, Mohammadia School of Engineers, Mohammed V University..."

### Examples that are not sufficient
- "An industrial partner participated."
- "The study was performed in company X."

### Common ambiguities
Do not require company identity if confidentiality prevents it. Anonymous but informative descriptions are enough.

### Decision rule
If an organizational setting exists, score `Yes` only when the paper describes its nature in a way that helps interpret the findings. If the study has no relevant organization, use `Not applicable`.

## Question 2.3

### Intent
Assesses whether the paper reports participant characteristics relevant to interpreting performance and validity.

### What qualifies as Yes
The paper reports subjects' skills, experience, background, or proficiency relevant to the studied task, tool, language, method, or domain.

### What qualifies as No
Mark `No` when participants are named only broadly, such as "students" or "professionals," without relevant experience details. Missing information should be scored `No`.

### When Not applicable should be used
In quantization studies, use `Not applicable` when there are no human subjects and the item cannot be meaningfully reinterpreted as characteristics of artifacts, datasets, or model families. Use `Yes` only when the paper reports analogous characteristics that play the same role in interpreting the findings.

### Examples of acceptable evidence
- In the current quantization-study corpus, there is no strong `Yes` example for this item. Most studies appropriately score `Not applicable`.

### Examples that are not sufficient
- "The subjects were experienced."
- "Participants were software engineers."

### Common ambiguities
For quantization studies, reviewers may disagree on whether this item should be reinterpreted. Resolve this by asking whether the paper reports characteristics analogous to subject proficiency that materially affect outcomes, such as model scale, architecture type, pretraining regime, dataset difficulty, or hardware capability. If yes, `Yes` is possible. If the item remains fundamentally about people, use `Not applicable`.

### Decision rule
In human-subject studies, score `Yes` only when concrete participant experience is reported. In quantization studies, score `Yes` only if the paper reports analogous artifact characteristics that materially affect interpretation; otherwise use `Not applicable`.

## Question 2.4

### Intent
Assesses whether the paper identifies the kind of software artifact or product involved in the experiment.

### What qualifies as Yes
The paper explicitly describes the type of software product, system, or artifact used in the study.

### What qualifies as No
Mark `No` when only generic labels such as "a system" or "a tool" are used. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` when no software product or artifact is involved, which should be uncommon in empirical software engineering experiments.

### Examples of acceptable evidence
- "In this study, we evaluate a diverse set of LLMs, comprising 18 model families in 3 different precision formats."
- "We use the PyTorch library, specifically version 2.2.1, for ML optimization due to its extensive adoption and versatility."

### Examples that are not sufficient
- "Participants used the software."
- "A prototype was evaluated."

### Common ambiguities
If the study centers on code snippets or small tasks rather than full products, a clear description of those artifacts still counts.

### Decision rule
Score `Yes` if the type of software artifact is explicitly described with enough specificity to understand what was studied. Otherwise score `No`.

## Question 2.5

### Intent
Assesses whether the process context is described, since development processes can affect how findings should be interpreted.

### What qualifies as Yes
The paper describes relevant software processes, workflows, or quality procedures when such processes are part of the study context.

### What qualifies as No
Mark `No` when a process context likely exists but is not described. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` when the study is detached from an organizational development process, such as a short lab task with no surrounding process context.

### Examples of acceptable evidence
- "The conversion and quantization process is illustrated in Fig. 2."
- "To implement the AI model for onboard cloud detection, the flowchart depicted in Fig. 5 demonstrates a structured methodology..."

### Examples that are not sufficient
- "The company used its normal process."
- "The task was performed in a realistic setting."

### Common ambiguities
Do not require exhaustive process documentation. A concise description of the relevant process features is sufficient.

### Decision rule
If the study is embedded in a development process, score `Yes` only when that process is explicitly described in a way that matters for interpretation. If no process context exists, use `Not applicable`.

## Question 3.1

### Intent
Assesses whether the paper explains what the experimental units were and how they were chosen.

### What qualifies as Yes
The paper identifies the experimental units and explains the selection or recruitment procedure.

Sufficient:
- Subjects, teams, tasks, code artifacts, or projects are defined as the units.
- Recruitment, sampling, inclusion criteria, or assignment source is reported.

### What qualifies as No
Mark `No` when the units can be guessed but are not explicitly defined or the selection method is omitted. Missing information should be scored `No`.

### When Not applicable should be used
Almost never. All experiments have experimental units.

### Examples of acceptable evidence
- "We begin our exploration by gathering a list of top LLMs based on the following criteria: Popularity: Popularity is measured by the number of downloads of the model sourced from HuggingFace [34]."
- "We examined Hugging Face's [14] top image classification datasets, with ImageNet-1k and CIFAR-10 being the most popular based on likes and downloads."

### Examples that are not sufficient
- "We conducted an experiment with several participants."
- "A set of projects was analyzed."

### Common ambiguities
Reviewers should distinguish between describing the sample and explaining selection. Reporting only who participated is not enough if the selection method remains unclear.

### Decision rule
Score `Yes` only if the paper states both what the experimental units were and how they were selected or recruited. If either part is missing, score `No`.

## Question 3.2

### Intent
Assesses whether the paper discusses how representative the selected units are of the target population.

### What qualifies as Yes
The paper explicitly comments on representativeness, generalizability of the sample, or how closely the selected units match the intended population.

### What qualifies as No
Mark `No` when the paper provides sample details but does not state how representative the units are. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` only when the study has no meaningful target population, which is rare.

### Examples of acceptable evidence
- "This will ensure that the results apply to the most commonly used models in real-world scenarios."
- "We executed a pipeline using the Hugging Face Hub API [13] to collect information on all models uploaded to the platform until March 3rd, 2024. The metrics include model size, training datasets, download and like counts, and the library used. We derived each model's popularity by summing the normalized number of likes and downloads."

### Examples that are not sufficient
- "Participants were volunteers."
- "We used real projects."

### Common ambiguities
A discussion of threats to external validity can satisfy this item if it explicitly addresses representativeness. Merely naming the sample does not.

### Decision rule
Score `Yes` only if the authors explicitly discuss how representative the units are, or are not, of the population they want to say something about.

## Question 3.3

### Intent
Assesses whether the paper justifies why the chosen units are appropriate for answering the research question.

### What qualifies as Yes
The paper explains why the selected participants, artifacts, teams, or projects are suitable for generating the intended knowledge.

### What qualifies as No
Mark `No` when the units are described but not justified. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "We begin our exploration by gathering a list of top LLMs based on the following criteria: Popularity: [...]. Reputability of creator: [...]."
- "The choice is justified by its widespread use in research [33] [57] and its significance in practical applications, ranging from medical imaging to facial recognition."

### Examples that are not sufficient
- "The sample was available."
- "We used these tasks because they were used in earlier studies," without explaining relevance.

### Common ambiguities
Convenience alone is not a justification. Prior use in the literature counts only if the paper explains why that makes the units suitable for the present question.

### Decision rule
Score `Yes` only if the paper explicitly explains why these units are appropriate for the study objective. Description without justification is `No`.

## Question 3.4

### Intent
Assesses whether the paper reports how many units were included in the study.

### What qualifies as Yes
The paper provides the sample size for the relevant experimental units.

### What qualifies as No
Mark `No` when the total number of units cannot be clearly determined. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "In this study, we evaluate a diverse set of LLMs, comprising 18 model families in 3 different precision formats."
- "Using stratified sampling, as described by Thompson [69] and applied in the context of Hugging Face models by Castaño et al. [24], we selected a representative sample of 42 models."

### Examples that are not sufficient
- "Several developers participated."
- A table from which sample size can only be guessed inconsistently.

### Common ambiguities
If different analyses use different sample sizes, the paper should make that clear. A single unambiguous study-wide sample size is sufficient for `Yes`.

### Decision rule
If a reviewer can identify the sample size directly from the paper, score `Yes`. If not, score `No`.

## Question 4.1

### Intent
Assesses whether the experimental design is described clearly enough to understand comparison structure and threat controls.

### What qualifies as Yes
The paper clearly states the design, such as within-subjects, between-subjects, crossover, blocked, factorial, repeated measures, or quasi-experimental structure, and explains treatment levels or grouping as relevant.

### What qualifies as No
Mark `No` when the reader must reconstruct the design from scattered details. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "The optimization strategies involved in this study represent the independent variables. The control group for this experiment is represented by the absence of optimization measures, referred to as 'no optimization'."
- "We identify each machine in terms of its GPU. Resource-demanding experiments, including full-precision models and their quantized counterparts, were conducted on a machine equipped with an NVIDIA A100 with 80GB memory, accessed via SSH."

### Examples that are not sufficient
- "We compared two techniques."
- "Participants performed both tasks," without clarifying order, blocking, or grouping.

### Common ambiguities
The study need not use textbook terminology if the structure is explicit enough to determine the design mechanically.

### Decision rule
Score `Yes` only if another reviewer could identify the design and treatment structure without inference. Otherwise score `No`.

## Question 4.2

### Intent
Assesses whether all interventions and comparison conditions are defined clearly enough to support replication and interpretation.

### What qualifies as Yes
The paper describes every treatment condition and any control or baseline condition in concrete terms.

### What qualifies as No
Mark `No` when one or more conditions are left vague or only named. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` only if the study truly has no control or comparison condition, though most experiments will still have at least one treatment condition that must be described.

### Examples of acceptable evidence
- "Out of the strategies detailed in Section 2.1, our experimental groups encompass dynamic quantization, torch.compile, and pruning... The control group for this experiment is represented by the absence of optimization measures, referred to as 'no optimization'."
- "We collected all full-precision, i.e., non-quantized, models and their quantized versions in GGUF format... The suffixes indicate the precision or quantization level used..."

### Examples that are not sufficient
- "One group used the tool and the other used the normal approach."
- "We evaluated our method against a baseline."

### Common ambiguities
If a cited prior paper fully describes a reused treatment and the current paper clearly identifies that reference, that can count as sufficient.

### Decision rule
Score `Yes` only if all study conditions are explicitly defined well enough to understand what each group actually received or did.

## Question 5.1

### Intent
Assesses measurement clarity and whether outcomes are defined in a reproducible way.

### What qualifies as Yes
All primary measures are defined with enough detail to know what was measured and how values were obtained.

Sufficient:
- Units, scales, scoring rules, defect counting rules, task completion definitions, questionnaire scales.

### What qualifies as No
Mark `No` when key outcomes are reported without operational definitions. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "To assess the correctness of the outputs generated by LLMs, we employed the mean pass@k (mean success rate) evaluation metric, as defined in the Codex evaluation set [21]."
- "Energy denotes the capacity to do work, measured in joules (J) or kilowatt-hours (kWh) (1kWh=3,600,000J)."

### Examples that are not sufficient
- "We measured quality."
- "Productivity was recorded," without defining the metric.

### Common ambiguities
Not every secondary variable must be defined exhaustively, but all core outcome measures must be.

### Decision rule
Score `Yes` only if the primary measures are operationally defined clearly enough that another reviewer could identify the scale, unit, or counting rule.

## Question 5.2

### Intent
Assesses whether the paper makes clear what form the collected data took.

### What qualifies as Yes
The paper reports the form or source of the collected data, such as logs, questionnaires, recordings, observations, code artifacts, or extracted repository data.

### What qualifies as No
Mark `No` when results are presented but the nature of the underlying data record is unclear. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "The output of this stage consists of five .csv files named after the corresponding task to which the models belong and the dataset used for training them."
- "Table I provides a comparative analysis of the performance of different versions of the YOLOv5m model, focusing on precision (P), recall (R), model size, and inference speed (measured in fps and converted to ms per frame)."

### Examples that are not sufficient
- "Data were collected during the sessions."
- "Measurements were taken automatically," without saying what records existed.

### Common ambiguities
This item is about data form, not measure definition. A paper may define a metric yet still fail to state whether the data came from logs, questionnaires, observations, or artifacts.

### Decision rule
Score `Yes` when the paper explicitly states what kinds of records or materials constituted the data. Otherwise score `No`.

## Question 5.3

### Intent
Assesses whether the paper reports procedures to improve data quality during collection.

### What qualifies as Yes
The paper describes quality control measures such as pilot testing, observer training, double coding, calibration, data validation scripts, protocol standardization, completeness checks, or instrument checks.

### What qualifies as No
Mark `No` when no such procedures are reported. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` only when no meaningful data-generation or measurement-quality procedure exists to evaluate, which should be rare even in quantization studies. In most quantization studies this item remains applicable because measurement pipelines, benchmarking, and profiling can vary substantially.

### Examples of acceptable evidence
- "Each experiment was run three times to reduce the impact of transient variations. The average of all recorded values was reported."
- "Next, we verify consistency by ensuring that the sum of CPU energy (from EnergiBridge) and GPU energy (from nvidia-smi) is lower than the global energy recorded by the wattmeter."

### Examples that are not sufficient
- "We followed the standard evaluation setup," without describing any checks or controls.
- "Measurements were taken on a GPU server," without explaining how consistency or accuracy was ensured.

### Common ambiguities
Routine use of a benchmark suite or profiling tool does not by itself count as quality control unless the paper explains how consistency, reproducibility, or correctness was ensured. For quantization studies, repeated runs, fixed seeds, fixed hardware settings, calibration protocol controls, and explicit validation of metrics are all relevant.

### Decision rule
Score `Yes` only if the paper explicitly reports one or more concrete procedures used to improve the consistency, completeness, or accuracy of benchmark, profiling, or evaluation data.

## Question 5.4

### Intent
Assesses reporting transparency about attrition, exclusions, failed runs, or incomplete observations.

### What qualifies as Yes
The paper reports whether any runs, models, datasets, measurements, or participants were excluded, failed, or missing, and ideally why or how they were handled.

### What qualifies as No
Mark `No` when attrition status is unclear. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` only when the study design makes attrition or exclusion genuinely irrelevant, for example a purely deterministic analytical comparison where no runs, measurements, datasets, or artifacts could be omitted after selection. In many quantization studies this item remains applicable because runs can fail, models can be omitted, and measurements can be discarded.

### Examples of acceptable evidence
- "Also, the fixed-point (4,4) fails to converge for all three networks on CIFAR-10 and the respective rows have been removed from the table."
- "... attempting to deploy the C++ LiteRT DynQ model on the STM32F7 microcontroller resulted in an error... As a result, data on SRAM/SDRAM usage and execution time for this model could not be reported in Table 3."

### Examples that are not sufficient
- Reporting only the final number of models or datasets with no indication whether any selected units were later excluded.
- "Invalid runs were filtered out," without saying how many or why.

### Common ambiguities
For artifact-based studies, "drop-out" should not be read literally as human withdrawal. It includes failed training runs, failed quantization, removed benchmarks, filtered measurements, and omitted baselines. Explicitly stating that none occurred counts as `Yes`. Silence does not.

### Decision rule
Score `Yes` only if the paper explicitly reports whether any selected units, runs, or measurements were excluded or missing, including an explicit statement that none were. Otherwise score `No`.

## Question 6.1

### Intent
Assesses transparency and appropriateness of analysis procedures.

### What qualifies as Yes
The paper describes the analysis procedures used and either justifies the choice or references a procedure description sufficiently.

### What qualifies as No
Mark `No` when statistical or analytic results are shown without explaining how they were obtained. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "To study the average impact of optimizations on the dependent variables (RQ1.1), we use a structured approach. Initially, we group the data according to each model, optimization, and repetition, and calculate the mean value for each variable."
- "The training process was carried out using Stochastic Gradient Descent (SGD) with a learning rate of 0.01 and a batch size of 16. The model was trained for 100 epochs."

### Examples that are not sufficient
- "We analyzed the data statistically."
- Reporting only p-values in tables with no analysis description.

### Common ambiguities
This item does not require the analysis choice to be perfect. It requires that the procedure be reported and justified enough for readers to understand it.

### Decision rule
Score `Yes` if the paper states what analysis procedures were used and gives either a rationale or a clear reference. Otherwise score `No`.

## Question 6.2

### Intent
Assesses whether the paper reports inferential results in a way that conveys both statistical significance and magnitude of effect.

### What qualifies as Yes
The paper reports significance levels and effect sizes for relevant inferential comparisons.

### What qualifies as No
Mark `No` when only significance or only effect size is reported, or when neither is reported in a study making inferential comparisons. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` when the study does not perform inferential statistical comparison, for example when it reports only deterministic benchmark values, descriptive comparisons, or engineering measurements with no significance testing.

### Examples of acceptable evidence
- "The matrix only presents coefficients for which the p-value is statistically significant, e.g., the correlation between Parameter Count and Accuracy is not statistically significant."
- "The significance levels used are as follows: high significance ('***'), very significant ('**'), significant ('*'), and minimal significance ('.')."

### Examples that are not sufficient
- "The result was significant."
- "Group A performed better than Group B," without p-value or effect size.

### Common ambiguities
Many quantization studies report accuracy, model size, latency, and energy comparisons without significance testing. Do not penalize these by forcing a `No` if inferential statistics are outside the study design; in such cases `Not applicable` is usually more consistent with the rubric's original intent. Confidence intervals may support interpretation, but they do not replace effect size unless the interval itself is clearly around an effect estimate.

### Decision rule
If inferential testing is used, score `Yes` only when both significance information and effect size information are reported. If no inferential comparison is performed, use `Not applicable`.

## Question 6.3

### Intent
Assesses transparency and justification of outlier handling.

### What qualifies as Yes
The paper explicitly mentions excluding outliers and explains the criteria or rationale for doing so.

### What qualifies as No
Mark `No` when outliers were excluded without justification. If outliers are not mentioned at all, do not score `Yes`.

### When Not applicable should be used
Use `Not applicable` when no outliers were mentioned or excluded.

### Examples of acceptable evidence
- "Also, the fixed-point (4,4) fails to converge for all three networks on CIFAR-10 and the respective rows have been removed from the table... 8 bits fails to capture the necessary range of the numbers."
- "... attempting to deploy the C++ LiteRT DynQ model on the STM32F7 microcontroller resulted in an error... As a result, data on SRAM/SDRAM usage and execution time for this model could not be reported in Table 3."

### Examples that are not sufficient
- "We removed anomalous values."
- "After cleaning the data, the sample size was 28," without explaining exclusions.

### Common ambiguities
This item applies only when exclusion of outliers or anomalous cases is discussed. If the paper is silent, reviewers should use `Not applicable`, not speculate.

### Decision rule
If outliers or anomalous cases were excluded, score `Yes` only when the exclusion is explicitly justified. If no such exclusions are discussed, use `Not applicable`.

## Question 6.4

### Intent
Assesses whether the paper provides enough underlying numerical information for verification and interpretation.

### What qualifies as Yes
The paper reports raw data, links to raw data, or at least descriptive statistics sufficient to understand the distributions and replicate basic interpretation.

### What qualifies as No
Mark `No` when only high-level inferential outcomes are reported without raw data or useful descriptive statistics. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "The replication package for this study, along with the appendices, is publicly available [11]."
- "Data availability statement: All research components are publicly available on Zenodo [20], including models, datasets, and the complete Python code (data downloading, preprocessing, inference, and analysis)."

### Examples that are not sufficient
- "Results are summarized in the text as better/worse."
- Reporting only p-values.

### Common ambiguities
The item allows either raw data or descriptive statistics. Raw data are not mandatory if descriptive statistics are adequate.

### Decision rule
Score `Yes` if the paper provides either raw data access or descriptive statistics that meaningfully characterize the outcomes. Otherwise score `No`.

## Question 7.1

### Intent
Assesses possible authorship-related bias arising when researchers evaluate methods, tools, or pipelines they developed themselves.

### What qualifies as Yes
Score `Yes` in either of these cases:
- The authors were not developers of the evaluated quantization method or key treatment components.
- The authors were developers of some or all treatments and explicitly discuss the implications of that involvement, such as benchmark choice, tuning advantage, or comparison risk.

### What qualifies as No
Mark `No` when the authors appear to be developers of the evaluated method, toolchain, or custom benchmark setup and do not discuss implications, or when that involvement is evident from context but left unaddressed.

### When Not applicable should be used
Use `Not applicable` only when there is no meaningful intervention or treatment ownership to assess, such as a purely descriptive benchmark aggregation with no proposed method. For most quantization-comparison papers this item remains applicable.

### Examples of acceptable evidence
- In the current quantization-study corpus, there is no strong `Yes` example for this item. Most studies were scored `Not applicable`, and the few applicable cases rarely discuss authorship-related comparison bias explicitly.

### Examples that are not sufficient
- "We propose a new quantization method and compare it to prior work," with no discussion of tuning, benchmark, or implementation bias.
- "All methods were implemented by the authors," with no reflection on comparability.

### Common ambiguities
For quantization studies, bias may arise less from interpersonal conduct and more from asymmetric tuning, hardware choices, implementation quality, or benchmark selection. Do not require a dedicated ethics discussion, but if the authors evaluate their own method, some explicit acknowledgment of this risk is needed for `Yes`.

### Decision rule
If the authors developed the evaluated method or pipeline, score `Yes` only if they explicitly discuss the implications of that role for fair comparison. If they clearly did not, score `Yes`. Otherwise score `No`.

## Question 7.2

### Intent
Assesses whether treatment conditions were executed under comparable conditions so that observed differences are not artifacts of unequal setup.

### What qualifies as Yes
The paper states that compared methods were evaluated under equivalent or appropriately controlled conditions, such as the same datasets, preprocessing, calibration budget, fine-tuning budget, hardware, software stack, and measurement protocol.

### What qualifies as No
Mark `No` when comparability of evaluation conditions is not reported, or when methods are run under different settings that could affect outcomes without justification. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` only when there are no distinct methods, baselines, or treatment conditions to compare.

### Examples of acceptable evidence
- "As before, we ensure that all other network parameters, including the frequency, are kept constant across different precision experiments."
- "Training for both models was conducted using mini-batch gradient descent with a batch size of 32 and the Adam optimizer... ensuring stable and efficient learning over 20 epochs."

### Examples that are not sufficient
- "We compared fairly against prior work," without stating the shared evaluation conditions.
- "Baselines were reimplemented," without clarifying whether the training and measurement budgets were comparable.

### Common ambiguities
Equivalent does not always mean identical. For example, different quantization methods may require method-specific calibration steps. The key question is whether the paper reports enough to judge that the comparison was materially fair rather than advantaging one method through better tuning, hardware, or preprocessing.

### Decision rule
Score `Yes` only if the paper explicitly shows that compared methods were evaluated under equivalent or appropriately controlled conditions.

## Question 7.3

### Intent
Assesses risk of allocation or selection bias arising from how units are assigned to treatments or comparison conditions.

### What qualifies as Yes
The paper explicitly states that assignment to treatment conditions followed a predefined, automated, or otherwise non-manipulable rule, or that selection into conditions could not be influenced by the researchers once the units were chosen.

### What qualifies as No
Mark `No` when treatment assignment or inclusion into comparison conditions could plausibly have been manipulated and the paper does not describe protections, or when the paper explicitly indicates selective assignment without justification. Missing information should usually be scored `No` if assignment is a meaningful design feature.

### When Not applicable should be used
Use `Not applicable` when there is no meaningful allocation step, which will be common in quantization studies where all selected models or datasets are evaluated under all conditions.

### Examples of acceptable evidence
- In the current quantization-study corpus, there is no strong `Yes` example for this item. Most studies evaluate the same artifacts across conditions, making allocation concealment genuinely not meaningful.

### Examples that are not sufficient
- "We selected representative models for each method," without explaining the selection rule.
- "Different models were used for different methods," with no justification or assignment logic.

### Common ambiguities
This item fits respondent studies better than artifact-comparison studies. In quantization papers, use it only when assignment or selective inclusion into conditions could realistically bias results. If every unit is evaluated under every relevant condition, `Not applicable` is often the cleanest score.

### Decision rule
If units are selectively assigned to different treatment conditions, score `Yes` only when the paper describes a predefined or non-manipulable assignment process. If all units are evaluated across all conditions and allocation is not meaningful, use `Not applicable`.

## Question 8.1

### Intent
Assesses whether the paper explicitly discusses external validity threats related to who performed the tasks, what materials were used, and what tasks were studied.

### What qualifies as Yes
The paper discusses external validity in relation to subjects, materials, and tasks, either individually or together.

### What qualifies as No
Mark `No` when external validity is not discussed or is mentioned only in a token sentence with no concrete linkage to subjects, materials, or tasks. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "Internal Validity: There is a dependency on the chosen models due to their reliance on the models available on Hugging Face, some of which had to be excluded due to missing files or incomplete training information."
- "However, we do not claim that this conclusion is generalizable to all neural network architectures or hardware designs."

### Examples that are not sufficient
- "External validity may be limited."
- "More studies are needed."

### Common ambiguities
A dedicated threats-to-validity section is not required, but the discussion must be specific enough to the study's subjects, materials, or tasks.

### Decision rule
Score `Yes` only if the paper explicitly discusses external validity in concrete relation to the sample, artifacts, or tasks used.

## Question 8.2

### Intent
Assesses whether quasi-experiments acknowledge design weaknesses and explain compensating design features.

### What qualifies as Yes
For quasi-experiments, the paper discusses weaknesses such as non-random assignment and explains design components used to mitigate them, such as matching, blocking, covariates, pretests, or comparison groups.

### What qualifies as No
Mark `No` when the study is quasi-experimental and no such discussion is provided. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` when the study is not a quasi-experiment.

### Examples of acceptable evidence
- "There is a dependency on the chosen models due to their reliance on the models available on Hugging Face, some of which had to be excluded due to missing files or incomplete training information. As a mitigation strategy, we have implemented a stratified sampling based on model popularity and size to ensure a representative selection."
- "A few models did not include clear starting and ending markers for code sections... To address this, we first employed a regular expression approach and subsequently used an open-access powerful LLM (gpt4o-mini) to extract the code..."

### Examples that are not sufficient
- "This was a quasi-experiment."
- "Randomization was not possible," with no mitigation discussion.

### Common ambiguities
Do not use this item to penalize true randomized experiments. It is conditional on quasi-experimental design.

### Decision rule
If the study is quasi-experimental, score `Yes` only when the paper explains design features used to address its inherent weaknesses. Otherwise use `No`; if not quasi-experimental, use `Not applicable`.

## Question 8.3

### Intent
Assesses whether the paper addresses construct validity when introducing new or adapted measures.

### What qualifies as Yes
When novel measures are used, the paper discusses why the measure captures the intended construct, including validation, rationale, pilot evidence, or relation to prior measures.

### What qualifies as No
Mark `No` when novel measures are used but construct validity is not discussed. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` when the study relies only on established measures and does not introduce a novel or adapted measure needing construct validity discussion.

### Examples of acceptable evidence
- "Although the ZigZag framework provides detailed estimates of execution costs, there is a key limitation: it assumes ..."
- "Following the AHA statement [84], the annotations were categorized into shockable (VF) and non-shockable (NSR, ONR, ASYS) rhythms."

### Examples that are not sufficient
- "We designed a new metric for this study."
- "A questionnaire was created by the authors," with no validity discussion.

### Common ambiguities
Minor operational changes to well-known measures may still trigger this item if they plausibly change the construct being captured.

### Decision rule
If the study uses a novel or materially adapted measure, score `Yes` only when construct validity is explicitly discussed. Otherwise use `No`; if no such measure is used, `Not applicable`.

## Question 9.1

### Intent
Assesses clarity of result reporting.

### What qualifies as Yes
Results are presented in an organized and understandable way, with tables, figures, or text that clearly link outcomes to conditions or questions.

### What qualifies as No
Mark `No` when results are fragmented, hard to interpret, or insufficiently connected to the reported analyses. Missing clarity should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "Figures 2 and 3 present the energy footprint of the analyzed models in the four tasks, for the two machines."
- "Finding 1.1: Dynamic quantization demonstrates a nearly two-fold increase in speed at the cost of slightly reducing accuracy and consuming more GPU resources."

### Examples that are not sufficient
- Results scattered across discussion and conclusion without clear reporting.
- Statements such as "the method performed better overall" with no structured presentation.

### Common ambiguities
This item assesses reporting clarity, not whether the results are positive or statistically strong.

### Decision rule
Score `Yes` if a reviewer can follow what was found, for which outcomes, and under which conditions without reconstructing the results manually.

## Question 9.2

### Intent
Assesses clarity of the study's take-home conclusions.

### What qualifies as Yes
The paper states its conclusions explicitly and in a way that a reader can distinguish from the raw results.

### What qualifies as No
Mark `No` when conclusions are absent, vague, or buried in diffuse discussion. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "Our findings indicate that while the energy footprint of the same model varies widely across different software development tasks, its energy usage per generated token remains consistent."
- "Finding 1.1: Dynamic quantization demonstrates a nearly two-fold increase in speed at the cost of slightly reducing accuracy and consuming more GPU resources."

### Examples that are not sufficient
- Ending the paper with a general statement about future work only.
- Repeating results numerically without drawing any conclusion.

### Common ambiguities
The conclusion need not be long. It must simply be explicit and identifiable.

### Decision rule
Score `Yes` if the paper clearly states what conclusions the authors draw from the study. Otherwise score `No`.

## Question 9.3

### Intent
Assesses whether the conclusions are supported by the reported evidence and whether that linkage is explicit.

### What qualifies as Yes
The paper's conclusions stay within the scope of the reported results and clearly explain how the evidence supports the claims.

### What qualifies as No
Mark `No` when conclusions overreach, imply unsupported causality, or are not explicitly tied back to the results. Missing justification should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "Our findings indicate that while the energy footprint of the same model varies widely across different software development tasks, its energy usage per generated token remains consistent."
- "The experimental results demonstrated that although the INT8 quantized models exhibited a decrease in precision compared to the original FP32 version, they remained effective for real-time object detection applications."

### Examples that are not sufficient
- "The technique should be adopted widely," based on a small student lab study with narrow outcomes.
- "Our approach is superior," when some outcomes were mixed or non-significant.

### Common ambiguities
Moderate interpretive language is acceptable if it stays within the data. Problems arise when authors generalize beyond the measured outcomes, population, or setting.

### Decision rule
Score `Yes` only if the conclusions are explicitly supported by the reported results and do not go materially beyond them.

## Question 9.4

### Intent
Assesses whether the paper closes the loop between findings and the original research questions.

### What qualifies as Yes
The paper explicitly discusses conclusions or results in relation to the original research questions or hypotheses.

### What qualifies as No
Mark `No` when research questions are stated earlier but never revisited directly. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` only if the study had no explicit research questions or equivalent framing.

### Examples of acceptable evidence
- "4.2 How do model optimization techniques, specifically dynamic quantization, pruning, and torch.compile affect quality attributes? (RQ1)"
- "The primary goal of this paper is to determine whether there are Pareto-optimal quantization strategies... We identify int8 as Pareto-optimal..."

### Examples that are not sufficient
- Presenting results in order without explicitly relating them to the research questions.
- "The findings are discussed above," with no direct mapping.

### Common ambiguities
If the paper uses hypotheses instead of RQs, mapping conclusions back to those hypotheses is sufficient.

### Decision rule
Score `Yes` only if the paper explicitly connects its findings or conclusions back to the original questions or hypotheses.

## Question 9.5

### Intent
Assesses whether study limitations are acknowledged explicitly.

### What qualifies as Yes
The paper explicitly discusses limitations of the study, weaknesses, or constraints affecting interpretation.

### What qualifies as No
Mark `No` when limitations are absent or only implied indirectly. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "A potential threat to internal validity in this study relates to the use of prompt templates."
- "In this section, we address potential threats to the study's validity, aiming to clarify the constraints and biases that could affect the interpretation of the findings."

### Examples that are not sufficient
- "Future work will include more participants."
- "Results should be interpreted with care," without saying why.

### Common ambiguities
General threats-to-validity discussion usually satisfies this item if it explicitly states limitations rather than merely naming validity categories.

### Decision rule
Score `Yes` if the paper explicitly identifies concrete study limitations. Otherwise score `No`.

## Question 10.1

### Intent
Assesses whether the paper discusses how findings may be transferred, applied, or used beyond the immediate study.

### What qualifies as Yes
The paper discusses transferability to other populations, contexts, or practical uses, including boundaries on such transfer.

### What qualifies as No
Mark `No` when the paper stops at reporting the study itself and does not discuss broader use. Missing information should be scored `No`.

### When Not applicable should be used
Almost never.

### Examples of acceptable evidence
- "Implications. These findings highlight that the software development tasks that a model supports directly impact its energy use... Selecting models based on their expected tasks is crucial to reduce energy consumption."
- "The extensive database created during this study can serve as a foundation for categorizing models, by clustering them according to energy consumption and performance metrics."

### Examples that are not sufficient
- "The study has implications for practice," with no further explanation.
- "More research is needed to generalize."

### Common ambiguities
This item is broader than external validity alone. Practical use discussion can satisfy it if it clearly addresses where and how the findings may apply.

### Decision rule
Score `Yes` only if the paper explicitly discusses how the findings may transfer to other settings or be used in research or practice.

## Question 10.2

### Intent
Assesses whether the authors interpret findings in relation to prior studies or the broader evidence base.

### What qualifies as Yes
The paper compares, situates, or interprets its results in the context of prior research, theory, or the existing body of knowledge.

### What qualifies as No
Mark `No` when prior work is cited only in the introduction and not used to interpret the current findings. Missing information should be scored `No`.

### When Not applicable should be used
Use `Not applicable` only in the rare case that no prior body of knowledge exists or the study is explicitly first-of-its-kind in a genuinely new area.

### Examples of acceptable evidence
- "Our results identify int8 as the Pareto-optimal bitwidth for MobileNetV2, in contrast to related work by [3] and [4], which suggest int4 as Pareto-optimal. Several factors could explain this discrepancy."
- "Our study builds on previous research by examining how quantization, pruning, and torch.compile affect not only model performance and resource usage, but also the economic costs associated with optimization and inference."

### Examples that are not sufficient
- A related work section that never revisits prior studies in the discussion.
- "Our findings are consistent with the literature," with no examples or explanation.

### Common ambiguities
Citing prior work is not enough. The paper must actively use prior knowledge to interpret what its own results mean.

### Decision rule
Score `Yes` only if the discussion or conclusion explicitly places the findings in relation to prior studies or existing knowledge. Otherwise score `No`.
