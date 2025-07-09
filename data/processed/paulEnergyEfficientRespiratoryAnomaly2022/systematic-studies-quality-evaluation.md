# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[X] No  
[ ] Yes  
[ ] Not applicable  

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

> "Abstract: Precise monitoring of respiratory rate in premature newborn infants is essential to initiating medical interventions as required." (Paul et al., 2022, p. 1).

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

> "We propose a deep-learning-enabled wearable monitoring system for premature newborn infants, where respiratory cessation is predicted using signals that are collected wirelessly from a non-invasive wearable Bellypatch put on the infant’s body." (Paul et al., 2022, p. 1).

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[X] Yes  
[ ] Not applicable  

> "We propose a five-stage design pipeline involving data collection and labeling, feature scaling, deep learning model selection with hyperparameter tuning, model training and validation, and model testing and deployment." (Paul et al., 2022, p. 1).

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[ ] No  
[X] Yes  
[ ] Not applicable  

> The baseline 1DCNN model (control/comparison point) is described (Section 3.3).

3.2) Do the authors state to what degree the experimental units are representative?  
[X] No  
[ ] Yes  
[ ] Not applicable  

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[X] No  
[ ] Yes  
[ ] Not applicable  

3.4) Do the authors report the sample size?  
[ ] No  
[X] Yes  
[ ] Not applicable  

> "We use a 1-D convolutional neural network (1DCNN) for classification of the features extracted from the SimBaby sensor data." (Paul et al., 2022, p. 2)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[X] Yes  
[ ] Not applicable  

> "We conduct our experiment to perform quantization of our Conv1D model using 2 bits, 4 bits, 8 bits, 16 bits, 32 bits, and 64 bits." (Paul et al., 2022, p. 10).

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[X] Yes  
[ ] Not applicable  

> The baseline 1DCNN model (control/comparison point) is described (Section 3.3). The treatments, namely quantization techniques (Section 4).

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[X] Yes  
[ ] Not applicable  

> Performance metrics are defined or described (Section 6.1). Energy units and model size units are stated (Table 4).

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[X] Yes  
[ ] Not applicable  

5.3) Are quality control methods used to ensure consistency, completeness, and accuracy of collected data?  
[ ] No  
[X] Yes  
[ ] Not applicable  

> "We first cleaned our feature set by filtering the missing values (NaN)." (Paul et al., 2022, p. 6). "We trained our 1DCNN model... and used repeated k-fold cross-validation with 10 splits to validate our model performance." (Paul et al., 2022, p. 9). "We defined Early Stopping as a regularization technique..." (Paul et al., 2022, p. 9).

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

> "Keras [36] is used to implement the baseline 1DCNN... QKeras [65] is used for training and testing the quantized neural network." (Paul et al., 2022, p. 14).

6.2) Do the authors report significance levels and effect sizes?  
[X] No  
[ ] Yes  
[ ] Not applicable  

6.3) If outliers are mentioned and excluded from the analysis, is this justified?  
[ ] No  
[ ] Yes  
[X] Not applicable  

6.4) Do the authors report or give references to raw data and/or descriptive statistics?  
[X] No  
[ ] Yes  
[ ] Not applicable  

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
[X] No  
[ ] Yes  
[ ] Not applicable  

8.2) If the study was a quasi-experiment, do the authors discuss the design components that were used to address any study weaknesses?  
[ ] No  
[ ] Yes  
[X] Not applicable  

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

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[X] Yes  
[ ] Not applicable  

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[X] No  
[ ] Yes  
[ ] Not applicable  

9.5) Are limitations of the study discussed explicitly?  
[X] No  
[ ] Yes  
[ ] Not applicable  

---

## 10. Is there evidence that the results can be used by other researchers/practitioners?

10.1) Do the authors discuss whether or how the findings can be transferred to other populations, or consider other ways in which the research can be used?  
[X] No  
[ ] Yes  
[ ] Not applicable  

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Table 3 compares the classification performance using the proposed 1DCNN against three state-of-the-art approaches: (1) Support Vector Machine (SVM) classifier of [18], (2) Logistic Regression (LR) classifier of [19], and (3) Random Forest classifier of [19]." (Paul et al., 2022, p. 15).
