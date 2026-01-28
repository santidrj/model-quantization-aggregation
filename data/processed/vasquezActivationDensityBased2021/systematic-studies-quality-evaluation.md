# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "we propose an in-training quantization method based on a novel activation density metric that yields a mixed-precision network and eliminates the need for a fully pre-trained model." (Vasquez et al., 2021, p. 1)

1.2) Do the authors state hypotheses and their underlying theories?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Based on Activation Density—the proportion of non-zero activations in a layer—we propose a novel intraining quantization method." (Vasquez et al., 2021, p. 1)  

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
[x] No  
[ ] Yes  
[ ] Not applicable

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Algorithm 1: Activation Density Based Quantization" (Vasquez et al., 2021, p. 4)

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[x] No  
[ ] Yes  
[ ] Not applicable

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

> "We run experiments on benchmark datasets like CIFAR-10, CIFAR-100, TinyImagenet on VGG19/ResNet18 architectures" (Vasquez et al., 2021, p. 1)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[x] No  
[ ] Yes  
[ ] Not applicable

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "To validate our method, we run a series of experiments measuring accuracy, energy (section IV-A) and training complexity (section IV-B) on CIFAR-10, CIFAR-100 and TinyImagenet datasets. The model is trained using Adam optimizer under standard settings." (Vasquez et al., 2021, p. 4)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "In a PIM architecture, energy is primarily expended during MAC operation as memory access energy is greatly reduced. Also, energy due to peripheral components is fairly minimal and have not been considered for evaluation. In our architecture, energy is consumed by the PIM block and the ShiftAccumulator block. Table IV lists the energy consumed for a single MAC operation for multiple bit-precisions evaluated on 45nm CMOS. Using this, we compute the total energy consumption for our model with AD-based quantization/pruning and compare the results with baseline 16-bit full precision models." (Vasquez et al., 2021, p. 6)

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[x] Yes  
[ ] Not applicable

5.3) Are quality control methods used to ensure consistency, completeness, and accuracy of collected data?  
[ ] No  
[ ] Yes  
[x] Not applicable

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

> "As discussed in Section I, we perform a realistic energy analysis pertaining to more practical hardware architecture considerations as opposed to the analytical estimates shown in Section IV-A. Thus, during analytical estimations in Table III, we get overestimated energy efficiencies ∼ 5 − 7× greater than practical hardware implementations (Table VI)" (Vasquez et al., 2021, p. 6)

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

> "Table V reports the energy comparison for VGG19, Reset18 networks on CIFAR-10, CIFAR-100 data with and without ADbased-quantization." (Vasquez et al., 2021, p. 4)

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

> "The model is trained using Adam optimizer under standard settings." (Vasquez et al., 2021, p. 4)

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
[ ] No  
[ ] Yes  
[x] Not applicable

8.3) If the study used novel measures, is the construct validity of the measures discussed?  
[x] No  
[ ] Yes  
[ ] Not applicable

---

## 9. Do the authors state the findings clearly?

9.1) Do the authors present results clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Table V reports the energy comparison for VGG19, Reset18 networks on CIFAR-10, CIFAR-100 data with and without ADbased-quantization." (Vasquez et al., 2021, p. 4)

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "We present a simple in-training quantization method based on Activation Density (AD) that enables us to compute the optimal bit-precision of each layer of a network during training. Our approach yields an energy-efficient mixed-precision model with iso-accuracy with baseline." (Vasquez et al., 2021, p. 6)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "We find that the AD-based quantization approach reduces the training complexity by 50% in our experiments alongside providing up to 4.5x benefit with respect to OPS reductions." (Vasquez et al., 2021, p. 6)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable

> "Since we train lower precision models progressively during training, our approach yields the final quantized model at lower training complexity and also eliminates the need for re-training." (Vasquez et al., 2021, p. 1)

9.5) Are limitations of the study discussed explicitly?  
[x] No  
[ ] Yes  
[ ] Not applicable

---

## 10. Is there evidence that the results can be used by other researchers/practitioners?

10.1) Do the authors discuss whether or how the findings can be transferred to other populations, or consider other ways in which the research can be used?  
[x] No  
[ ] Yes  
[ ] Not applicable

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[x] No  
[ ] Yes  
[ ] Not applicable
