# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This study aims to show the complete workflow for deploying a pre-trained DNN model from a GPU-based development platform to two popular ARM-based microcontrollers" (Krasteva et al., 2025, p. 3)  

1.2) Do the authors state hypotheses and their underlying theories?  
[x] No  
[ ] Yes  
[ ] Not applicable

---

## 2. Is there an adequate description of the context in which the research was carried out?

2.1) The industry in which products are used (e.g., banking, telecommunications, consumer goods, travel, etc.)  
[ ] No  
[x] Yes  
[ ] Not applicable

> "simulates real-time rhythm analysis in automated external defibrillators" (Krasteva et al., 2025, p. 1)  

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

> "Raspberry Pi 4 and ARM Cortex-M7..." (Krasteva et al., 2025, p. 3)  

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Figure 1 illustrates the main steps of the conversion pipeline for DNNmodels followed in this study" (Krasteva et al., 2025, p. 4)  

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This paper considers a TF DNN model trained for detection of ventricular fibrillation in a previous deep learning study [67]." (Krasteva et al., 2025, p. 4)  

3.2) Do the authors state to what degree the experimental units are representative?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The workflow is applied to a high-performing CNN model for detection of ventricular fibrillation during out-of-hospital cardiac arrest [67]" (Krasteva et al., 2025, p. 4)

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This approach can be extended to numerous published DNN models optimized for rhythm analysis during cardiopulmonary resuscitation." (Krasteva et al., 2025, p. 4)

3.4) Do the authors report the sample size?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This paper considers a TF DNN model trained for detection of ventricular fibrillation in a previous deep learning study [67]." (Krasteva et al., 2025, p. 4)  

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Figure 5 illustrates the deployment of each DNN model in this study for testing on its respective target platform" (Krasteva et al., 2025, p. 10)  

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "the TF DNN on a GPU-based workstation, the LiteRT DNN on both the Raspberry Pi 4 and STM32F7 microcontroller, and the quantized LiteRT models (DynQ, IntQ and Ful-IntQ) on the STM32F7 microcontroller" (Krasteva et al., 2025, p. 10)  

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Performance is evaluated in terms of accuracy, latency, and memory usage" (Krasteva et al., 2025, p. 3)  

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[x] No  
[ ] Yes  
[ ] Not applicable

> Table 3 reports the resource efficiency metrics aggregated for all four target tasks while Table 4 reports the accuracy metrics for each target task separately.

5.3) Are quality control methods used to ensure consistency, completeness, and accuracy of collected data?  
[x] No  
[ ] Yes  
[ ] Not applicable

> No mention of control of external factors that could affect latency and energy consumption measurements.

5.4) Do the authors report drop-outs?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "... attempting to deploy the C++ LiteRT DynQ model on the STM32F7 microcontroller resulted in an error caused by the LiteRT library for microcontrollers [86], as it does not currently support hybrid models. As a result, data on SRAM/SDRAM usage and execution time for this model could not be reported in Table 3." (Krasteva et al., 2025, p. 16)

---

## 6. Do the authors define the data analysis procedures?

6.1) Do authors justify their choice / describe the procedures / provide references to descriptions of the procedures?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The test interface... feeds the data... result is sent back... for comparison with the ground truth" (Krasteva et al., 2025, p. 10)  

6.2) Do the authors report significance levels and effect sizes?  
[x] No  
[ ] Yes  
[ ] Not applicable

6.3) If outliers are mentioned and excluded from the analysis, is this justified?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "... attempting to deploy the C++ LiteRT DynQ model on the STM32F7 microcontroller resulted in an error caused by the LiteRT library for microcontrollers [86], as it does not currently support hybrid models. As a result, data on SRAM/SDRAM usage and execution time for this model could not be reported in Table 3." (Krasteva et al., 2025, p. 16)

6.4) Do the authors report or give references to raw data and/or descriptive statistics?  
[ ] No  
[x] Yes  
[ ] Not applicable

> Tables 3 and 4.

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

> "none of the above conversion methods require retraining" (Krasteva et al., 2025, p. 6)  

7.3) Was there allocation concealment, i.e., did the researchers know to what treatment each subject was assigned?  
[ ] No  
[ ] Yes  
[x] Not applicable

---

## 8. Do the authors discuss the limitations of their study?

8.1) Do the authors discuss external validity with respect to subjects, materials, and tasks?  
[x] No  
[ ] Yes  
[ ] Not applicable

8.2) If the study was a quasi-experiment, do the authors discuss the design components that were used to address any study weaknesses?  
[x] No  
[ ] Yes  
[ ] Not applicable

8.3) If the study used novel measures, is the construct validity of the measures discussed?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Following the AHA statement [84], the annotations were categorized into shockable (VF) and non-shockable (NSR, ONR, ASYS) rhythms" (Krasteva et al., 2025, p. 11)  

---

## 9. Do the authors state the findings clearly?

9.1) Do the authors present results clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Table 4 presents the shock advisory performance of all DNN models" (Krasteva et al., 2025, p. 18)  

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This study makes a significant contribution to the application of neural networks for VFdetection in AEDs by presenting a methodological workflow" (Krasteva et al., 2025, p. 29)  

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The LiteRT model achieves the same high accuracy as the original TensorFlow DNN model... while delivering a rapid and highly reproducible inference time" (Krasteva et al., 2025, p. 29)  

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This study aims to show a useful workflow... This study makes a significant contribution... by presenting a methodological workflow" (Krasteva et al., 2025, p. 1, 29)  

9.5) Are limitations of the study discussed explicitly?  
[x] No  
[ ] Yes  
[ ] Not applicable

---

## 10. Is there evidence that the results can be used by other researchers/practitioners?

10.1) Do the authors discuss whether or how the findings can be transferred to other populations, or consider other ways in which the research can be used?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This approach can be extended to numerous published DNN models optimized for rhythm analysis during cardiopulmonary resuscitation" (Krasteva et al., 2025, p. 3)  

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The SRAM usage observed in this study is comparable to the 157 kB reported in [59]" (Krasteva et al., 2025, p. 28)  
