# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "In this paper we aim to address this issue by providing a quantitative analysis of different precisions and available trade-offs. More specifically, our paper makes the following contributions: [...]" (Hashemi et al., 2017, p. 1474)

1.2) Do the authors state hypotheses and their underlying theories?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "In addition, we propose that a small portion of the benefits achieved when using lower precisions can be forfeited to increase the network size and therefore the accuracy." (Hashemi et al., 2017, p. 1474)

---

## 2. Is there an adequate description of the context in which the research was carried out?

2.1) The industry in which products are used (e.g., banking, telecommunications, consumer goods, travel, etc.)  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Deep neural networks (DNN) have provided state-of-the-art results in many different applications specifically related to computer vision and machine learning." (Hashemi et al., 2017, p. 1474)

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
> "To measure accuracy, we adopt Ristretto [9], a Caffe-based framework [10] extended to simulate fixed-point operation." (Hashemi et al., 2017, p. 1477)
> "We compile our designs using Synopsys Design Compiler using a 65 nm industry strength technology node library." (Hashemi et al., 2017, p. 1477)

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
> "Benchmarks: We consider three well-recognized neural network architectures utilized with three different datasets, MNIST [15] using the LeNet [14] architecture, SVHN using CONVnet [19], and CIFAR-10 [12] using the network described by Alex Krizhevsky [12]" (Hashemi et al., 2017, p. 1477)

3.2) Do the authors state to what degree the experimental units are representative?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "We evaluate our experiments, using three well-recognized networks and datasets to show its generality." (Hashemi et al., 2017, p. 1474)

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Here, we focus on CIFAR-10 since MNIST and SVHN do not provide a large range in accuracy differences between various precisions and quantizations." (Hashemi et al., 2017, p. 1477)

3.4) Do the authors report the sample size?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Benchmarks: We consider three well-recognized neural network architectures utilized with three different datasets..." (Hashemi et al., 2017, p. 1477)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "We consider a broad range of numerical precisions and quantizations, from 32-bit floating-point arithmetic to binary nets, as well as several precision points in between." (Hashemi et al., 2017, p. 1476)

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "1) Floating-Point Arithmetic: ... 2) Fixed-Point Arithmetic: ... 3) Power-of-Two Quantization: ... 4) Binary Representation:" (Hashemi et al., 2017, p. 1476)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "We evaluate our designs both in terms of accuracy and design metrics (i.e., power, energy, memory requirements, design area)." (Hashemi et al., 2017, p. 1477)
> "Values shown as (w, in) represent the number of bits required for representing weight and input values, respectively." (Hashemi et al., 2017, p. 1477)

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Table III summarizes the design metrics of the accelerator for each of the numerical precisions considered." (Hashemi et al., 2017, p. 1477)

5.3) Are quality control methods used to ensure consistency, completeness, and accuracy of collected data?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "In different experiments, we ensure that all design parameters except for the bit precision are the same. This is critical to ensure the isolation of the effects of bit precision from any other factor." (Hashemi et al., 2017, p. 1477)

5.4) Do the authors report drop-outs?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Also, the fixed-point (4,4) fails to converge for all three networks on CIFAR-10 and the respective rows have been removed from the table." (Hashemi et al., 2017, p. 1478)

---

## 6. Do the authors define the data analysis procedures?

6.1) Do authors justify their choice / describe the procedures / provide references to descriptions of the procedures?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "To measure accuracy, we adopt Ristretto [9]... We compile our designs using Synopsys Design Compiler using a 65 nm industry strength technology node library. We use a 250 MHz clock frequency and synthesize in nominal processing corner." (Hashemi et al., 2017, p. 1477)

6.2) Do the authors report significance levels and effect sizes?  
[x] No  
[ ] Yes  
[ ] Not applicable  

6.3) If outliers are mentioned and excluded from the analysis, is this justified?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Also, the fixed-point (4,4) fails to converge for all three networks on CIFAR-10 and the respective rows have been removed from the table. Furthermore, we find that the accuracy for fixed-point++ (8,8) is lower in comparison to the other networks with the same precision. We observe that for this network, there is a significant difference in the range of parameter and feature map values and as a result, 8 bits fails to capture the necessary range of the numbers." (Hashemi et al., 2017, p. 1478)

6.4) Do the authors report or give references to raw data and/or descriptive statistics?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Table III summarizes the design metrics of the accelerator for each of the numerical precisions considered." (Hashemi et al., 2017, p. 1477)

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
> "As before, we ensure that all other network parameters, including the frequency, are kept constant across different precision experiments." (Hashemi et al., 2017, p. 1477)

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
> "Table IV summarizes the results for MNIST and SVHN datasets." (Hashemi et al., 2017, p. 1477)
> "The summary of the performances for the ALEX as well as the two larger networks (ALEX+ and ALEX++) is provided in Table V." (Hashemi et al., 2017, p. 1478)

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "We also show that lower-precision, larger networks can be utilized which outperform the smaller full-precision counterparts in both energy and accuracy." (Hashemi et al., 2017, p. 1479)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "As shown in the Figure 4, this methodology can eliminate the accuracy drop (for example in the case of Power of Two++ (6,16)) while still delivering energy savings of 35.93%." (Hashemi et al., 2017, p. 1478)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Our results showcase low precision networks capable of achieving equivalent accuracy compared to smaller floating-point networks while offering significant improvements in energy consumption and design area." (Hashemi et al., 2017, p. 1474)

9.5) Are limitations of the study discussed explicitly?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Additional runtime savings can be achieved by increasing the frequency or changing the accelerator specification which is not explored in this work." (Hashemi et al., 2017, p. 1478)

---

## 10. Is there evidence that the results can be used by other researchers/practitioners?

10.1) Do the authors discuss whether or how the findings can be transferred to other populations, or consider other ways in which the research can be used?  
[x] No  
[ ] Yes  
[ ] Not applicable  

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[ ] No  
[x] Yes  
[ ] Not applicable  
> "Chen et al. proposed Eyeriss... Sankaradas et al. empirically determine an acceptable precision... While use of limited precision in neural networks has been proposed before [16], [4], [17], there exists no comprehensive exploration of their effect on energy consumption..." (Hashemi et al., 2017, p. 1475)
