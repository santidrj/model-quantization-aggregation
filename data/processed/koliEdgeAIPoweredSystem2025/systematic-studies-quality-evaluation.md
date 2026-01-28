# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The objective of this learning was to develop a strong and real-time Aloe Vera leaf disease diagnostics and prediction system based on Machine Learning & Edge AI. [...] In addition, the study seeks to evaluate the system's performance against conventional approaches in terms of accuracy, effectiveness, and computational cost." (Koli et al., 2025, p. 3)

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

> "Aloe Vera is an important species in sustainable agriculture, pharmaceuticals, and cosmetics due to its potential for drought resistance, medicinal effects, and commercial value" (Koli et al., 2025, p. 1)

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

> "The models were developed and trained in Google Colab, a cloud-based platform that provides free GPU acceleration and supports large-scale deep learning experiments efficiently. TensorFlow, an open-source deep learning framework, was utilized for model implementation, training, and evaluation" (Koli et al., 2025, p. 6)

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Figure 3 illustrates a comprehensive framework for an Edge AI-powered Aloe Vera Plant Disease Detection System, which comprises two major components: real-time data acquisition and application of lightweight algorithms on a Raspberry Pi device, followed by communication and data flow management." (Koli et al., 2025, p. 334)

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "In this study, two architectures were employed: a baseline ResNet50 model and an improved Enhanced ResNet50 model specifically tailored for the task." (Koli et al., 2025, p. 10)

3.2) Do the authors state to what degree the experimental units are representative?  
[x] No  
[ ] Yes  
[ ] Not applicable

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[x] No  
[ ] Yes  
[ ] Not applicable

3.4) Do the authors report the sample size?  
[ ] No  
[x] Yes  
[ ] Not applicable

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "During the training phase with TensorFlow, the proposed model underwent post-training quantization to optimize its performance for deployment on edge devices [39] ... As part of this transformation, quantization was pragmatic to diminish the model size and boost its computational efficiency. This is achieved by lowering the correctness of model weights from 32-bit floating point to 8-bit integers [41]" (Koli et al., 2025, p. 11)

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "As part of this transformation, quantization was pragmatic to diminish the model size and boost its computational efficiency. This is achieved by lowering the correctness of model weights from 32-bit floating point to 8-bit integers [41]" (Koli et al., 2025, p. 11)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The performance of the projected model was assessed using numerous metrics, including accuracy, precision, recall, and F1-score. These metrics are consequent from the confusion matrix, which comprises true-positive (T.P.), true-negative (T.N.), false-positive (F.P.), and false-negative (F.N.) predictions." (Koli et al., 2025, p. 7)

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[x] Yes  
[ ] Not applicable

5.3) Are quality control methods used to ensure consistency, completeness, and accuracy of collected data?  
[x] No  
[ ] Yes  
[ ] Not applicable

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

> "Training for both models was conducted using mini-batch gradient descent with a batch size of 32 and the Adam optimizer, applying a learning rate decay schedule with decay steps of 1000, a decay rate of 0.96, and staircase=True, ensuring stable and efficient learning over 20 epochs. Sparse Categorical Cross-Entropy loss was utilized, and the final classification output was achieved through a softmax activation function over three classes." (Koli et al., 2025, p. 6)

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

> "Table 6. Performance comparison of model deployment on different devices, including raspberry Pi 4" (Koli et al., 2025, p. 14)

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

> "Training for both models was conducted using mini-batch gradient descent with a batch size of 32 and the Adam optimizer, applying a learning rate decay schedule with decay steps of 1000, a decay rate of 0.96, and staircase=True, ensuring stable and efficient learning over 20 epochs." (Koli et al., 2025, p. 6)

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
[ ] Yes  
[x] Not applicable

---

## 9. Do the authors state the findings clearly?

9.1) Do the authors present results clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The proposed Aloe Vera leaf disease classification model, based on ResNet50, achieved high classification performance with an accuracy of 99.15%, precision of 99.20%, recall of 99.21%, and an F1-score of 99.20%. [...] The model achieves an inference latency of 4,922 ms (~4.9s) on Raspberry Pi 4, utilizing higher RAM with a quantized model size of 23.4MB, significantly reducing memory usage from 102.4MB (FP32) to 23.4MB (INT8) while maintaining classification accuracy." (Koli et al., 2025, p. 11)

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "These conclusions highlight the potential of edge-based deep learning for scalable and cost-effective plant disease detection, paving the way for further advancements in smart precision farming." (Koli et al., 2025, p. 11)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The deployment of the quantized TFLite model on Raspberry Pi 4 B ensures efficient edge computing, enabling real-time disease detection with reduced latency and computational overhead." (Koli et al., 2025, p. 11)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "These advancements have contributed to the growing field of AI-driven precision farming, offering practical and scalable solutions for farmers." (Koli et al., 2025, p. 11)

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

> "These advancements have contributed to the growing field of AI-driven precision farming, offering practical and scalable solutions for farmers. These conclusions highlight the potential of edge-based deep learning for scalable and costeffective plant disease detection, paving the way for further advancements in smart precision farming." (Koli et al., 2025, p. 15)

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[x] No  
[ ] Yes  
[ ] Not applicable
