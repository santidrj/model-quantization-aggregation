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
> "Verifiable and Energy Efficient Medical Image Analysis with Quantised Self-attentive Deep Neural Networks" (Sathish et al., 2022, p. 1). The datasets used are also from the medical domain.

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
> "Quantised Self-attentive Deep Neural Networks" (Sathish et al., 2022, p. 1) for "medical image classification and segmentation" (Sathish et al., 2022, p. 2).

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
> "Performance of the proposed quantised self-attention network for the classification task is compared with ResNet-18, ResNet-50 and their 8-bit quantised versions q-ResNet-18, and q-ResNet-50. To assess the performance of the segmentation network, we chose a modified UNet [18] (UNet-small) and SUMNet [13] architecture trained on the same dataset split and their quantised versions q-UNet-small and q-SUMNet as baselines." (Sathish et al., 2022, p. 6). Dataset selection: "To evaluate the performance... on classification tasks, we have used the NIH Chest X-ray dataset... A subset of the medical segmentation decathlon dataset [2] is used to evaluate the performance... for liver segmentation." (Sathish et al., 2022, p. 5).

3.2) Do the authors state to what degree the experimental units are representative?  
[X] No  
[ ] Yes  
[ ] Not applicable  

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Inspired by the success of [17] in natural image classification tasks, we propose the design of a new class of networks for medical image classification and segmentation..." (Sathish et al., 2022, p. 2).

3.4) Do the authors report the sample size?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Performance of the proposed quantised self-attention network for the classification task is compared with ResNet-18, ResNet-50 and their 8-bit quantised versions q-ResNet-18, and q-ResNet-50. To assess the performance of the segmentation network, we chose a modified UNet [18] (UNet-small) and SUMNet [13] architecture trained on the same dataset split and their quantised versions q-UNet-small and q-SUMNet as baselines." (Sathish et al., 2022, p. 6).

---

## 4. Do the authors describe the design of the experiment?
4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "Performance of the proposed quantised self-attention network... is compared with ResNet-18, ResNet-50 and their 8-bit quantised versions..." (Sathish et al., 2022, p. 6).

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> "The architecture of the proposed Self-attentive Deep Neural Networks (SaDNN). Detailed architecture of the networks for classification and segmentation are shown in (a) and (b) respectively. Components of the various blocks in these networks are detailed in (c)." (Sathish et al., 2022, p. 6). Baselines are listed in Section 3.2.

---
## 5. Do the authors describe the data collection procedures and define the measures?
5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?    
[ ] No  
[X] Yes  
[ ] Not applicable  
> "The performance of the proposed quantised fully self-attentive network and baselines for multi-label classification task is reported in terms of accuracy in Table 1." (Sathish et al., 2022, p. 7). "Table 2 shows the comparison of the proposed segmentation network with the baselines in terms of DSC." (Sathish et al., 2022, p. 7). Efficiency metrics (#Params, MACs, Model size, Energy) are reported in Tables 4 and 5. "A rough estimate of energy cost per operation in 45nm 0.9V IC design can be calculated using Table 3 presented in [7, 14, 23]." (Sathish et al., 2022, p. 8).

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[X] Yes  
[ ] Not applicable  

5.3) Are quality control methods used to ensure consistency, completeness, and accuracy of collected data?  
[X] No  
[ ] Yes  
[ ] Not applicable  

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
> The analysis procedure is primarily a direct comparison of the collected metrics (performance and efficiency) across the different models, presented in tables and charts.

6.2) Do the authors report significance levels and effect sizes?  
[X] No  
[ ] Yes  
[ ] Not applicable  

6.3) If outliers are mentioned and excluded from the analysis, is this justified?  
[ ] No  
[ ] Yes  
[X] Not applicable  

6.4) Do the authors report or give references to raw data and/or descriptive statistics?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> The paper reports descriptive/summary statistics (e.g., mean accuracy, mean DSC, total parameters, total MACs, total model size, estimated total energy) in Tables 1, 2, 4, and 5.

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

## 10. Is there evidence that the results can be used by other researchers/practitioners?

10.1) Do the authors discuss whether or how the findings can be transferred to other populations, or consider other ways in which the research can be used?  
[X] No  
[ ] Yes  
[ ] Not applicable  

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[ ] No  
[X] Yes  
[ ] Not applicable  
> The results are interpreted by comparing them directly against established baseline methods (ResNet, UNet, SUMNet), thus contextualizing the findings within the current state-of-the-art.
