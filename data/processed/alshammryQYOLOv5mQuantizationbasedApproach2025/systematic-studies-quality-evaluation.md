# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "To address this issue, this study proposes a novel quantization method tailored to accelerate object detection using a quantized version of the YOLOv5m model, called Q_YOLOv5m. This method reduces the model's computational complexity and memory footprint, allowing for faster inference and lower power consumption, making it ideal for real-time applications on embedded systems." (Alshammry et al., 2025, p. 19750)

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

> "The findings underscore the capability of Q_YOLOv5m for edge applications, including autonomous vehicles, intelligent surveillance, and IoT-based monitoring systems." (Alshammry et al., 2025, p. 19750)

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

> "This study proposes a novel quantization method tailored to accelerate object detection using a quantized version of the YOLOv5m model, called Q_YOLOv5m." (Alshammry et al., 2025, p. 19750)

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

> "This study proposes Q_YOLOv5m, a quantized version of the YOLOv5m model specifically designed for embedded platforms." (Alshammry et al., 2025, p. 19750)

3.2) Do the authors state to what degree the experimental units are representative?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Models such as the YOLO (You Only Look Once) family, particularly YOLOv5 [1], have gained prominence due to their ability to achieve a balance between speed and accuracy." (Alshammry et al., 2025, p. 19750)

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Despite these advantages, even smaller models, such as  YOLOv5m, pose significant computational and memory  challenges when deployed on embedded platforms with  constrained resources." (Alshammry et al., 2025, p. 19750)

3.4) Do the authors report the sample size?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "This study used the COCO 128 dataset, which consists of the first 128 images of the MS COCO 2017 training set" (Alshammry et al., 2025, p. 19752)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Table I provides a comparative analysis of the performance of different versions of the YOLOv5m model... The three configurations include the standard floating-point 32 (FP32) version and two quantized versions, Q_YOLOv5m, using QAT and PTQ with 8-bit integers (Int8)." (Alshammry et al., 2025, p. 19753)

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "YOLOv5m (FP32) serves as the baseline... In contrast, Q_YOLOv5m QAT IΝΤ8... Q_YOLOv5m PTQ IΝΤ8..." (Alshammry et al., 2025, p. 19753)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Detection accuracy is quantified using Average Precision (AP) and mean AP (mAP) metrics... The mAP metric measures detection accuracy by calculating the mean of the AP across all classes. The calculations for precision (P), recall (R), AP, and mAP are outlined as follows:" (Alshammry et al., 2025, p. 19752)

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Table I provides a comparative analysis of the performance of different versions of the YOLOv5m model, focusing on precision (P), recall (R), model size, and inference speed (measured in fps and converted to ms per frame)." (Alshammry et al., 2025, p. 19753)

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

> "The training process was carried out using Stochastic Gradient Descent (SGD) with a learning rate of 0.01 and a batch size of 16. The model was trained for 100 epochs." (Alshammry et al., 2025, p. 19752)

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

> "Table I provides a comparative analysis of the performance of different versions of the YOLOv5m model..." (Alshammry et al., 2025, p. 19753)

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

> "The training process was carried out using Stochastic Gradient Descent (SGD) with a learning rate of 0.01 and a batch size of 16. The model was trained for 100 epochs." (Alshammry et al., 2025, p. 19752)

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

> "Additionally, the performance of Q_YOLOv5m should be evaluated on various embedded platforms and real-world scenarios to better understand its applicability in practical applications." (Alshammry et al., 2025, p. 19753)

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

> "Table I provides a comparative analysis of the performance of different versions of the YOLOv5m model, focusing on precision (P), recall (R), model size, and inference speed" (Alshammry et al., 2025, p. 19753)

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The Q_YOLOv5m model achieved a balance between efficiency and accuracy, making it a viable option for deployment in resource-constrained environments." (Alshammry et al., 2025, p. 19753)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The experimental results demonstrated that although the IΝΤ8 quantized models exhibited a decrease in precision compared to the original FP32 version, they remained effective for real-time object detection applications." (Alshammry et al., 2025, p. 19753)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Through the implementation of Quantization Aware Training (QAT) and Post-Training Quantization (PTQ), the model size was significantly reduced and inference speed was improved while maintaining a satisfactory level of accuracy." (Alshammry et al., 2025, p. 19753)

9.5) Are limitations of the study discussed explicitly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Despite the drop in precision associated with the IΝΤ8 quantization, it remains within an acceptable range... Additionally, the performance of Q_YOLOv5m should be evaluated on various embedded platforms and real-world scenarios" (Alshammry et al., 2025, p. 19753)

---

## 10. Is there evidence that the results can be used by other researchers/practitioners?

10.1) Do the authors discuss whether or how the findings can be transferred to other populations, or consider other ways in which the research can be used?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "The Q_YOLOv5m model achieved a balance between efficiency and accuracy, making it a viable option for deployment in resource-constrained environments." (Alshammry et al., 2025, p. 19754)

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[x] No  
[ ] Yes  
[ ] Not applicable
