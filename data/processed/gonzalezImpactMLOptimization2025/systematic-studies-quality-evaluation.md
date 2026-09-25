# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "RQ1: How do model optimization techniques, specifically dynamic quantization, pruning, and torch.compile affect quality attributes?[...]" (González Álvarez et al., 2024, p. 6)

1.2) Do the authors state hypotheses and their underlying theories?  
[X] No  
[ ] Yes  
[ ] Not applicable  

---

## 2. Is there an adequate description of the context in which the research was carried out?

2.1) The industry in which products are used (e.g., banking, telecommunications, consumer goods, travel, etc.)  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Regarding the context of this research, we focus on a specific ML task, namely image classification. This task involves categorizing an image into one of predefined classes. The choice is justified by its widespread use in research [33] [57] and its significance in practical applications, ranging from medical imaging to facial recognition." (González Álvarez et al., 2024, p. 5)

2.2) If applicable, the nature of the software development organization (e.g., in-house department or independent software supplier)  
[ ] No  
[ ] Yes  
[X] Not applicable  

2.3) The skills and experience of the subjects (e.g., with a language, a method, a tool, an application domain)  
[ ] No  
[ ] Yes  
[X] Not applicable  

2.4) The type of software products used (e.g., a design tool, a compiler)  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "We use the PyTorch library, specifically version 2.2.1, for ML optimization due to its extensive adoption and versatility (see Section 5.2)." (González Álvarez et al., 2024, p. 6)

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[ ] Yes  
[X] Not applicable  

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "We examined Hugging Face's [14] top image classification datasets, with ImageNet-1k and CIFAR-10 being the most popular based on likes and downloads." (González Álvarez et al., 2024, p. 3)

3.2) Do the authors state to what degree the experimental units are representative?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "We executed a pipeline using the Hugging Face Hub API [13] to collect information on all models uploaded to the platform until March 3rd, 2024. The metrics include model size, training datasets, download and like counts, and the library used. We derived each model's popularity by summing the normalized number of likes and downloads." (González Álvarez et al., 2024, p. 6)

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "The choice is justified by its widespread use in research [33] [57] and its significance in practical applications, ranging from medical imaging to facial recognition." (González Álvarez et al., 2024, p. 5)

3.4) Do the authors report the sample size?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Using stratified sampling, as described by Thompson [69] and applied in the context of Hugging Face models by Castaño et al. [24], we selected a representative sample of 42 models (see Section 4.1)." (González Álvarez et al., 2024, p. 7)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "The optimization strategies involved in this study represent the independent variables. The control group for this experiment is represented by the absence of optimization measures, referred to as 'no optimization'." (González Álvarez et al., 2024, p. 7)

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Out of the strategies detailed in Section 2.1, our experimental groups encompass dynamic quantization, torch.compile, and pruning, chosen for their potential to enhance computational efficiency and reduce energy consumption, with quantization and pruning being the most commonly used techniques [10]." (González Álvarez et al., 2024, p. 7)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Energy denotes the capacity to do work, measured in joules (J) or kilowatt-hours (kWh) (1kWh=3,600,000J)." (González Álvarez et al., 2024, p. 5)

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "The output of this stage consists of five .csv files named after the corresponding task to which the models belong and the dataset used for training them." (González Álvarez et al., 2024, p. 9)

5.3) Are quality control methods used to ensure consistency, completeness, and accuracy of collected data?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Next, we verify consistency by ensuring that the sum of CPU energy (from EnergiBridge) and GPU energy (from nvidia-smi) is lower than the global energy recorded by the wattmeter." (González Álvarez et al., 2024, p. 9)

5.4) Do the authors report drop-outs?  
[ ] No  
[ ] Yes  
[X] Not applicable  

---

## 6. Do the authors define the data analysis procedures?

6.1) Do authors justify their choice / describe the procedures / provide references to descriptions of the procedures?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "To study the average impact of optimizations on the dependent variables (RQ1.1), we use a structured approach. Initially, we group the data according to each model, optimization, and repetition, and calculate the mean value for each variable." (González Álvarez et al., 2024, p. 10)

6.2) Do the authors report significance levels and effect sizes?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "The significance levels used are as follows: high significance ('\*\*\*'), very significant ('\*\*'), significant ('\*'), and minimal significance ('.')." (González Álvarez et al., 2024, p. 16)

6.3) If outliers are mentioned and excluded from the analysis, is this justified?  
[ ] No  
[ ] Yes  
[X] Not applicable  

6.4) Do the authors report or give references to raw data and/or descriptive statistics?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Data availability statement: All research components are publicly available on Zenodo [20], including models, datasets, and the complete Python code (data downloading, preprocessing, inference, and analysis)." (González Álvarez et al., 2024, p. 3)

---

## 7. Do the authors discuss potential experimenter bias?

7.1) Were the authors the developers of some or all of the treatments? If yes, do the authors discuss the implications anywhere in the paper?  
[ ] No  
[ ] Yes  
[X] Not applicable  

7.2) Was training and conduct equivalent for all treatment groups?  
[ ] No  
[ ] Yes  
[X] Not applicable  

7.3) Was there allocation concealment, i.e., did the researchers know to what treatment each subject was assigned?  
[ ] No  
[ ] Yes  
[X] Not applicable  

---

## 8. Do the authors discuss the limitations of their study?

8.1) Do the authors discuss external validity with respect to subjects, materials, and tasks?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Internal Validity: There is a dependency on the chosen models due to their reliance on the models available on Hugging Face, some of which had to be excluded due to missing files or incomplete training information." (González Álvarez et al., 2024, p. 24)

8.2) If the study was a quasi-experiment, do the authors discuss the design components that were used to address any study weaknesses?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "There is a dependency on the chosen models due to their reliance on the models available on Hugging Face, some of which had to be excluded due to missing files or incomplete training information. As a mitigation strategy, we have implemented a stratified sampling based on model popularity and size to ensure a representative selection." (González Álvarez et al., 2024, p. 24)

8.3) If the study used novel measures, is the construct validity of the measures discussed?  
[ ] No  
[ ] Yes  
[X] Not applicable  

---

## 9. Do the authors state the findings clearly?

9.1) Do the authors present results clearly?  
[ ] No  
[X] Yes  
[ ] Not applicable  

9.2) Do the authors present conclusions clearly?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Finding 1.1: Dynamic quantization demonstrates a nearly two-fold increase in speed at the cost of slightly reducing accuracy and consuming more GPU resources." (González Álvarez et al., 2024, p. 16)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Finding 1.1: Dynamic quantization demonstrates a nearly two-fold increase in speed at the cost of slightly reducing accuracy and consuming more GPU resources." (González Álvarez et al., 2024, p. 16)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "4.2 How do model optimization techniques, specifically dynamic quantization, pruning, and torch.compile affect quality attributes? (RQ1)" (González Álvarez et al., 2024, p. 14)

9.5) Are limitations of the study discussed explicitly?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "In this section, we address potential threats to the study's validity, aiming to clarify the constraints and biases that could affect the interpretation of the findings." (González Álvarez et al., 2024, p. 24)

---

## 10. Is there evidence that the results can be used by other researchers/practitioners?

10.1) Do the authors discuss whether or how the findings can be transferred to other populations, or consider other ways in which the research can be used?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "The extensive database created during this study can serve as a foundation for categorizing models, by clustering them according to energy consumption and performance metrics." (González Álvarez et al., 2024, p. 25)

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Our study builds on previous research by examining how quantization, pruning, and torch.compile affect not only model performance and resource usage, but also the economic costs associated with optimization and inference." (González Álvarez et al., 2024, p. 22)
