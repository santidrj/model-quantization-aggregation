# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

### 1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Therefore, this study proposes a neural network training and inference framework on resource-constrained embedded systems, taking atrial fibrillation detection from electrocardiography (ECG) as a remote health monitoring case." (p. 1)

> "The contributions of this work are:
> • We propose a neural network training framework with training acceleration and model compression for extremely resource-constrained edge training.
> • We discuss the impact of neural network structure on model compression.
> • We implement the proposed framework on an ATmega2560 MCU and verify the feasibility of its on-device neural networks training and inference for atrial fibrillation detection from ECG." (p. 2)

### 1.2) Do the authors state hypotheses and their underlying theories?

[x] No  
[ ] Yes  
[ ] Not applicable  

---

## 2. Is there an adequate description of the context in which the research was carried out?

### 2.1) The industry in which products are used (e.g., banking, telecommunications, consumer goods, travel, etc.)

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Implementing internet of things technologies in health monitoring systems attracts a lot of attention. Running the model at edge can continuously and in real-time monitor the user’s physiological information, which can be adopted in universal medical care." (p. 1)

### 2.2) If applicable, the nature of the software development organization (e.g., in-house department or independent software supplier)

[ ] No  
[ ] Yes  
[x] Not applicable  

### 2.3) The skills and experience of the subjects (e.g., with a language, a method, a tool, an application domain)

[ ] No  
[ ] Yes  
[x] Not applicable  

### 2.4) The type of software products used (e.g., a design tool, a compiler)

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "This study adopted Arduino Mega, which has 256 KB of FLASH, 8 KB of SRAM, 4 KB of EEPROM, and a clock frequency of 16 MHZ. The embedded processors with floating-point support are able to run the general neural network, like the Arm Cortex-M family... The AI-special edge devices usually have GPU, like Raspberry Pi and Jetson Nano." (pp. 2–3)

### 2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The proposed framework includes model training, model storage, model compression and model inference, aiming to accelerate the training processing and reach inference performance lossless model compression..." (p. 2)

> "Section 3 introduces a remote Atrial fibrillation detection system with edge computing, including neural networks training, model storage, model quantitation, ECG dataset and feature extraction. Section 4 presents the hardware environment, experiment design and evaluation metrics." (p. 2)

---

## 3. Do the authors explain how experimental units were defined and selected?

### 3.1) Do the authors explain how experimental units were defined and selected?  

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The ECG signals come from the 2017 ECG Recognition Atrial Fibrillation Challenge (Goldberger et al., 2000), including 8528 ECG signals from different individuals... In this study, only normal signals (N) and atrial fibrillation signals (AF) in the data set were chosen." (p. 5)

> "In order to finish our task, some ECG signals are chosen from this dataset to train the model and other ECG signals to test." (p. 5)

### 3.2) Do the authors state to what degree the experimental units are representative?  

[x] No  
[ ] Yes  
[ ] Not applicable  

### 3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The ECG signals come from the 2017 ECG Recognition Atrial Fibrillation Challenge (Goldberger et al., 2000)... In this study, only normal signals (N) and atrial fibrillation signals (AF) in the data set were chosen." (p. 5)

> "The proposed edge training and inference methods were implemented and tested with five ECG features for atrial fibrillation detection..." (p. 9)

### 3.4) Do the authors report the sample size?  

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The ECG signals come from the 2017 ECG Recognition Atrial Fibrillation Challenge (Goldberger et al., 2000), including 8528 ECG signals from different individuals... There are 5174 Normal recordings and 771 AF recordings in this dataset." (p. 5)

> "The composition and description of the training dataset and testing dataset is shown in the following table." (p. 5)

---

## 4. Do the authors describe the design of the experiment?

### 4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The experiment aims to prove the feasibility of training neural network models directly on embedded devices and compare the model performance and the training time in two platforms. By measuring the performance and training time with different sizes of training data in the embedded devices, the size ranges from 10 to 100, all trained models will be tested on the testing dataset." (p. 6)

> "The 5-3-1 network structure and these model hyper-parameters are adopted in the following experiments without special instructions. The input layer is five ECG features, and the hidden layer consists of three neurons and one output. Considering different situations, we used the normalized method for data, and the transfer functions were tansig and linear, respectively. The learning rate is 0.4 and the momentum is 0.5." (p. 6)

### 4.2) Do the authors define/describe all treatments and all controls?  

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The proposed methods were Simple-int and Max–min, which will be compared and discussed in this experiment." (p. 6)

> "By discarding some parameters randomly in the stored process and generating parameters according to the standard normal distribution during the rebuilding process, three methods of parameter discarding are proposed: discarding the latter or former parameter whose names are PD1 and PD2, respectively, and selecting the most important component in order of absolute value, which is called as PDs." (p. 6)

---

## 5. Do the authors describe the data collection procedures and define the measures?

### 5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "This paper will use the following indicators." (p. 6)

> "Accuracy = (TP + TN)/(TP + TN + FP + FN)." (Eq. 16, p. 6)

> "Precision = TP/(TP + FP)." (Eq. 17, p. 6)

> "Recall = TP/(TP + FN)." (Eq. 18, p. 6)

> "F1-score = 2TP/(2TP + FN + FP)." (Eq. 19, p. 6)

### 5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The ECG signals come from the 2017 ECG Recognition Atrial Fibrillation Challenge (Goldberger et al., 2000), including 8528 ECG signals from different individuals..." (p. 5)

> "The composition and description of the training dataset and testing dataset is shown in the following table." (p. 5)

### 5.3) Are quality control methods used to ensure consistency, completeness, and accuracy of collected data?  

[x] No  
[ ] Yes  
[ ] Not applicable  

### 5.4) Do the authors report drop-outs?  

[ ] No  
[ ] Yes  
[x] Not applicable  

---

## 6. Do the authors define the data analysis procedures?

### 6.1) Do authors justify their choice / describe the procedures / provide references to descriptions of the procedures?  

[x] No  
[ ] Yes  
[ ] Not applicable  

### 6.2) Do the authors report significance levels and effect sizes?  

[x] No  
[ ] Yes  
[ ] Not applicable  

### 6.3) If outliers are mentioned and excluded from the analysis, is this justified?  

[ ] No  
[ ] Yes  
[x] Not applicable  

### 6.4) Do the authors report or give references to raw data and/or descriptive statistics?  

[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Fig. 4 presents the test performance of models trained on CPU with increasing size of the training dataset... Results show that the average F1 score in the test dataset gradually increased..." (p. 6)

> "The 500 parallel experiments tell the mean Ae was 11.04, while the mean Re was 88.69, and model performance is shown in Table 3." (p. 7)

> "The parallel experimental results are shown in Table 4." (p. 8)

---

## 7. Do the authors discuss potential experimenter bias?

### 7.1) Were the authors the developers of some or all of the treatments? If yes, do the authors discuss the implications anywhere in the paper?

[x] No  
[ ] Yes  
[ ] Not applicable

### 7.2) Was training and conduct equivalent for all treatment groups?

[ ] No  
[x] Yes  
[ ] Not applicable

**Evidence:**
> "The 5-3-1 network structure and these model hyper-parameters are adopted in the following experiments without special instructions. The input layer is five ECG features, and the hidden layer consists of three neurons and one output. Considering different situations, we used the normalized method for data, and the transfer functions were tansig and linear, respectively. The learning rate is 0.4 and the momentum is 0.5." (p. 6)

### 7.3) Was there allocation concealment, i.e., did the researchers know to what treatment each subject was assigned?

[ ] No  
[ ] Yes  
[x] Not applicable

---

## 8. Do the authors discuss the limitations of their study?

### 8.1) Do the authors discuss external validity with respect to subjects, materials, and tasks?

[x] No  
[ ] Yes  
[ ] Not applicable

### 8.2) If the study was a quasi-experiment, do the authors discuss the design components that were used to address any study weaknesses?

[x] No  
[ ] Yes  
[ ] Not applicable  

### 8.3) If the study used novel measures, is the construct validity of the measures discussed?

[ ] No  
[ ] Yes  
[x] Not applicable

---

## 9. Do the authors state the findings clearly?

### 9.1) Do the authors present results clearly?

[ ] No  
[x] Yes  
[ ] Not applicable

**Evidence:**
> "Fig. 4 presents the test performance of models trained on CPU with increasing size of the training dataset... Results show that the average F1 score in the test dataset gradually increased..." (p. 6)

> "Among the tested neural network weights compression methods, the Max–min method has the least performance loss after model rebuilding with a compression ratio about 0.30." (p. 9)

### 9.2) Do the authors present conclusions clearly?

[ ] No  
[x] Yes  
[ ] Not applicable

**Evidence:**
> "The open core challenge is mainly embodied in the implementation of edge computing, especially on resource-constrained embedded platforms. To solve it, this study proposed a novel neural network training and inference framework..." (p. 9)

> "The proposed could be easily applied to other similar health monitoring applications." (p. 9)

### 9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?

[ ] No  
[x] Yes  
[ ] Not applicable

**Evidence:**
> "The experimental results validate the feasibility of edge training and the necessity of training acceleration, on the extremely resource-constrained device." (p. 7)

> "The proposed edge training and inference methods were implemented and tested with five ECG features for atrial fibrillation detection, with F1-score ranges from 0.8 to 0.9." (p. 9)

### 9.4) Do the authors discuss their conclusions in relation to the original research questions?

[ ] No  
[x] Yes  
[ ] Not applicable

**Evidence:**
> "The experiment aims to prove the feasibility of training neural network models directly on embedded devices and compare the model performance and the training time in two platforms." (p. 6)

> "The proposed edge training and inference methods were implemented and tested with five ECG features for atrial fibrillation detection..." (p. 9)

### 9.5) Are limitations of the study discussed explicitly?

[x] No  
[ ] Yes  
[ ] Not applicable

---

## 10. Is there evidence that the results can be used by other researchers/practitioners?

### 10.1) Do the authors discuss whether or how the findings can be transferred to other populations, or consider other ways in which the research can be used?

[ ] No  
[x] Yes  
[ ] Not applicable

**Evidence:**
> "The proposed could be easily applied to other similar health monitoring applications." (p. 9)

> "Therefore, it could be applied to developing or remote areas to promote overall health awareness." (p. 3)

### 10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?

[ ] No  
[x] Yes  
[ ] Not applicable

**Evidence:**
> "To highlight the advantages of the method in this paper, it is important to compare the model performance loss under the same compression ratio in different methods." (p. 8)

> "The compression ratio of PD1, PD2, and PDs is 0.273... Besides, the clustering quantification method will be adopted in this experiment (Han et al., 2015)." (p. 8)

> "The binaryNet (Courbariaux & Bengio, 2016) is adopted and developed recently... Based on the experimental results, the model performance will reduce largely, compared with the 8-bit model." (p. 9)
