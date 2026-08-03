# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "In this paper, we evaluate the various features of the quantization process supported in Pytorch and Tensorflow on CNN and GNN based Recommendation models." (p. 1)

> "We compare the accuracy and memory efficiency of these quantization libraries with different supported features of the quantization process." (p. 1)

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

**Evidence:**  
> "CNN and GNN based Recommendation models" (p. 1)

> "real-world problems such as SRGNNs" (p. 5)

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

**Evidence:**  
> "TensorQuant [1] and Pytorch.quantization [5]" (p. 1)

> "LeNet [4], InceptionV1 [3], and SRGNN [6]" (p. 1)

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The types of quantizations supported are Static, Dynamic and Quantization-aware-training (QAT)." (p. 3)

> "We can apply static quantization by two methods: manually preparing ... and converting ... [or] calling the quantization.quantize() function." (p. 3)

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Lenet and Inception are trained and tested with the MNIST dataset. The experiments with SRGNN model are conducted through a subset of the Diginetica dataset [6]." (p. 2)

> "We apply quantization ... on several CNN and GNN Neural networks, namely LeNet [4], InceptionV1 [3], and SRGNN [6]." (p. 1)

3.2) Do the authors state to what degree the experimental units are representative?  
[x] No  
[ ] Yes  
[ ] Not applicable  

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The SRGNN model comprises Embedding layers and Linear layers... for larger datasets, the model occupies a tremendous amount of memory." (p. 2)

> "run it through the same dataset to observe the effect of quantization on a more complex CNN." (p. 2)

3.4) Do the authors report the sample size?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We will quantize the layers to the following bit representations: (16,8), (16,4), (16,0), (12,8), (12,4), (12,0), (8,4), (8,0), (4,0)" (p. 2)

> "we run only one-tenth of the MNIST dataset." (p. 2)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The quantization process of TensorFlow is compared on following features. Precision..." (p. 2)

> "The types of quantizations supported are Static, Dynamic and Quantization-aware-training (QAT)." (p. 3)

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The baseline here was an unquantized version of both the models" (p. 2)

> "No quantization", "Using convert()", and "Using quantize()" are explicitly reported in Table IV. (p. 4)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "the metric used is testing accuracy." (p. 2)

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> The results are tabulated as training/testing accuracy, training/testing time, and memory occupied in Tables I, III, and IV. (pp. 4-5)

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

**Evidence:**  
> "PyTorch has a more detailed module with a variety of options where we can apply quantization." (p. 3)

> "Dynamic quantization is chosen when throughput has higher priority ... Statically quantized methods are preferred when the memory usage needs to be controlled..." (p. 3)

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

**Evidence:**  
> Tables I, III, and IV report descriptive statistics for accuracy, time, and memory. (pp. 4-5)

> "The results obtained by quantizing the weights ... are mentioned in Table 1 and can be seen in Figure 3." (p. 4)

---

## 7. Do the authors discuss potential experimenter bias?

7.1) Were the authors the developers of some or all of the treatments? If yes, do the authors discuss the implications anywhere in the paper?  
[ ] No  
[ ] Yes  
[x] Not applicable  

7.2) Was training and conduct equivalent for all treatment groups?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "All the experiments were conducted on the personal computer system with 8GBs of RAM and AMD Radeon graphics card. All the computations were executed on the CPU." (p. 2)

> "Lenet and Inception are trained and tested with the MNIST dataset." (p. 2)

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

**Evidence:**  
> "The results obtained by quantizing the weights ... are mentioned in Table 1 and can be seen in Figure 3." (p. 4)

> "The experimental results for SRGNN using PyTorch.quantization can be observed in the change in memory occupied and computation time, tabulated in Table V..." (p. 4)

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Thus, it leads us to conclude that TensorQuant is not yet ripe to be applied to real-world problems such as SRGNNs." (p. 5)

> "PyTorch.quantization gives much more assuring returns..." (p. 5)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "When static quantization is implemented to the LeNet model, accuracy persists almost unchanged, whereas the model’s size is reduced to nearly 30 percent." (p. 4)

> "Since it only offers the user to quantize extrinsically and not intrinsically, there is a lot yet to be developed." (p. 5)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We have benchmarked the quantization libraries of Pytorch and Tensorflow..." (p. 5)

> "We have compared them on the memory space, accuracy, and training time of the quantized model." (p. 5)

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

**Evidence:**  
> "when employed to real-world problems such as SRGNNs" (p. 5)

> "As progress will be made in quantization, we can see better results with recommendation systems accordingly." (p. 5)

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[x] No  
[ ] Yes  
[ ] Not applicable  
