# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The primary goal of this paper is to determine whether there are Pareto-optimal quantization strategies, when considering various hardware and software factors that affect the accuracy-cost trade-off." (De Putter et al., 2025, p. 1)

1.2) Do the authors state hypotheses and their underlying theories?  
[x] No  
[ ] Yes  
[ ] Not applicable

---

## 2. Is there an adequate description of the context in which the research was carried out?

2.1) The industry in which products are used (e.g., banking, telecommunications, consumer goods, travel, etc.)  
[ ] No  
[ ] Yes  
[x] Not applicable

2.2) If applicable, the nature of the software development organization (e.g., in-house department or independent software supplier)  
[ ] No  
[ ] Yes  
[x] Not applicable

2.3) The skills and experience of the subjects (e.g., with a language, a method, a tool, an application domain)  
[ ] No  
[ ] Yes  
[x] Not applicable

2.4) The type of software products used (e.g., a design tool, a compiler)  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Table 1 summarizes the overall design space that we examine in both the baseline scenario and the subsequent analyses." (De Putter et al., 2025, p. 3)

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[ ] Yes  
[x] Not applicable

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "We evaluate the top-1 accuracy of each quantized model on the ImageNet [24] classification task." (De Putter et al., 2025, p. 6)

3.2) Do the authors state to what degree the experimental units are representative?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This approach begins with a baseline setup that represents a typical scenario, involving a specific neural network architecture, quantization strategy, and hardware platform." (De Putter et al., 2025, p. 3)

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "For the baseline scenario, we use the MobileNetV2 architecture [9]. This network is well-suited for resource-constrained environments due to its high accuracy at a small model size." (De Putter et al., 2025, p. 3)

3.4) Do the authors report the sample size?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Analysis of the factors that influence the accuracycost trade-off involving multiple deep learning architectures (MobileNetV2, ResNet18)..." (De Putter et al., 2025, p. 2)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This work presents a Pareto-Optimal Quantization (POQ) methodology aimed at mapping a neural network architecture to a specific hardware platform while systematically exploring the design space in between to identify the most effective quantization strategy." (De Putter et al., 2025, p. 1)

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Each quantized model is initialized by a pre-trained full-precision ( fp16 ) model..." (De Putter et al., 2025, p. 4)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[x] No  
[ ] Yes  
[ ] Not applicable

> Figures have missing data for some filter multiplier levels.

5.3) Are quality control methods used to ensure consistency, completeness, and accuracy of collected data?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "We use standard data preprocessing for the training data (random resize cropping, random horizontal flip, and normalization), while the test data are center-cropped." (De Putter et al., 2025, p. 4)

5.4) Do the authors report drop-outs?  
[ ] No  
[ ] Yes  
[x] Not applicable

---

## 6. Do the authors define the data analysis procedures?

6.1) Do authors justify their choice / describe the procedures / provide references to descriptions of the procedures?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Following [3] and [11], we adopt quantization-aware finetuning for all experiments." (De Putter et al., 2025, p. 4)

6.2) Do the authors report significance levels and effect sizes?  
[x] No  
[ ] Yes  
[ ] Not applicable

6.3) If outliers are mentioned and excluded from the analysis, is this justified?  
[ ] No  
[ ] Yes  
[x] Not applicable

6.4) Do the authors report or give references to raw data and/or descriptive statistics?  
[ ] No  
[x] Yes  
[ ] Not applicable

> Data in form of Figures.

---

## 7. Do the authors discuss potential experimenter bias?

7.1) Were the authors the developers of some or all of the treatments? If yes, do the authors discuss the implications anywhere in the paper?  
[x] No  
[ ] Yes  
[ ] Not applicable

7.2) Was training and conduct equivalent for all treatment groups?  
[ ] No  
[x] Yes  
[ ] Not applicable

7.3) Was there allocation concealment, i.e., did the researchers know to what treatment each subject was assigned?  
[ ] No  
[ ] Yes  
[x] Not applicable

---

## 8. Do the authors discuss the limitations of their study?

8.1) Do the authors discuss external validity with respect to subjects, materials, and tasks?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "However, we do not claim that this conclusion is generalizable to all neural network architectures or hardware designs." (De Putter et al., 2025, p. 11)

8.2) If the study was a quasi-experiment, do the authors discuss the design components that were used to address any study weaknesses?  
[ ] No  
[ ] Yes  
[x] Not applicable

8.3) If the study used novel measures, is the construct validity of the measures discussed?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Although the ZigZag framework provides detailed estimates of execution costs, there is a key limitation: it assumes ..." (De Putter et al., 2025, p. 6)

---

## 9. Do the authors state the findings clearly?

9.1) Do the authors present results clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Our results reveal that 8-bit integer ( int8 ) quantization is Pareto-Optimal for MobileNetV2..." (De Putter et al., 2025, p. 1)

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "We identify int8 as Pareto-optimal if limited to a single precision quantization strategy." (De Putter et al., 2025, p. 2)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Our results reveal that 8-bit integer ( int8 ) quantization is Pareto-Optimal for MobileNetV2, providing up to 2.8 × energy savings or 10% higher accuracy..." (De Putter et al., 2025, p. 1)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The primary goal of this paper is to determine whether there are Pareto-optimal quantization strategies... We identify int8 as Pareto-optimal..." (De Putter et al., 2025, p. 1-2)

9.5) Are limitations of the study discussed explicitly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Limitations: Our findings are based on a specific set of evaluated configurations." (De Putter et al., 2025, p. 11)

---

## 10. Is there evidence that the results can be used by other researchers/practitioners?

10.1) Do the authors discuss whether or how the findings can be transferred to other populations, or consider other ways in which the research can be used?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "New future models, such as transformers, along with hardware architecture evolution like compute-in-memory, and novel learning strategies, require new evaluations using our POQ methodology..." (De Putter et al., 2025, p. 11)

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Our results identify int8 as the Pareto-optimal bitwidth for MobileNetV2, in contrast to related work by [3] and [4], which suggest int4 as Pareto-optimal. Several factors could explain this discrepancy." (De Putter et al., 2025, p. 7)
