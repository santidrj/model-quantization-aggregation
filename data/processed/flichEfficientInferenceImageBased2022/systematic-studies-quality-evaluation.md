# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "In this paper, we contribute with an open source platform to develop customized accelerators on FPGAs specifically tailored for pruned and quantized models." (p. 1)

> "Our evaluation demonstrates that quantized and pruned models can largely benefit performance when combined with HLSinf. Specifically, results show that up to 90x speed up can be achieved on typical medical image-based applications using NN models on FPGAs." (p. 1)

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
> "We considered two use cases combining pruning and quantization. The ISIC dataset for skin lesion [16] is used for both classification and segmentation for melanoma diagnosis." (p. 3)

> "With HLSinf, significant inference speedups can be obtained for typical medical image-based applications." (p. 1)

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
> "In this paper, we present HLSinf, an open source framework for the development of custom NN accelerators for FPGAs which provides efficient support to quantized and pruned NN models." (p. 1)

> "VGG16 [8] and SegNet [17] were used for classification and segmentation, respectively." (p. 3)

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "EDDL allows training NN models or loading them using ONNX [15]. In order to use our HLSinf accelerator, we have designed a new functionality in EDDL, which transforms an input model into a new one with added HLSinf layers where needed." (p. 3)

> "The open source N2D2 framework is used as a quantization tool in this study. PTQ is chosen [22] which takes a model trained using FP32 and directly quantizes it to INT8, without any re-training or fine-tuning." (p. 4)

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We considered two use cases combining pruning and quantization. The ISIC dataset for skin lesion [16] is used for both classification and segmentation for melanoma diagnosis." (p. 3)

> "VGG16 [8] and SegNet [17] were used for classification and segmentation, respectively." (p. 3)

3.2) Do the authors state to what degree the experimental units are representative?  
[x] No  
[ ] Yes  
[ ] Not applicable  

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We considered two use cases combining pruning and quantization. The ISIC dataset for skin lesion [16] is used for both classification and segmentation for melanoma diagnosis." (p. 3)

> "VGG16 [8] and SegNet [17] were used for classification and segmentation, respectively." (p. 3)

3.4) Do the authors report the sample size?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "During the pruning phase we used a validation set to monitor the performance of the pruned network. For our test we produced two models for the classification task and one for the segmentation problem." (p. 3)

> "Table 1 shows some details of the models used in our experiments." (p. 4)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Hardware support We run our experiments on a Intel i7-7800-X CPU at 3.45GHz and a Xilinx ALVEO U200 FPGA board. The CPU-only results use all the 12 threads available in the machine. FPGA results use a single core to offload computations to the ALVEO board attached to this CPU." (p. 3)

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "PTQ is chosen [22] which takes a model trained using FP32 and directly quantizes it to INT8, without any re-training or fine-tuning." (p. 4)

> "Table 2: Inference time, FPS, and speedup for ISIC classification and segmentation models for CPU and FPGA." (p. 4)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Table 1a summarises the pruned models in terms of performance (classification error for VGG and Dice score for SegNet) and percentage of remaining neurons. Table 1b shows the results of PTQ applied to the original VGG model for Nbits = 8." (p. 4)

> "Table 2: Inference time, FPS, and speedup for ISIC classification and segmentation models for CPU and FPGA." (p. 4)

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The ISIC dataset for skin lesion [16] is used for both classification and segmentation for melanoma diagnosis." (p. 3)

> "Performance Table 2 shows the inference time of a single 224 × 224 input image on different models and devices for the skin lesion segmentation and classification problems." (p. 4)

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
> "Quantization methods can be roughly divided into two categories [21]: Quantization Aware Training (QAT) and Post-Training Quantization (PTQ)." (p. 4)

> "PTQ is chosen [22] which takes a model trained using FP32 and directly quantizes it to INT8, without any re-training or fine-tuning... In N2D2, the post-training quantization algorithm is done in 3 steps..." (p. 4)

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
> "Table 1 shows some details of the models used in our experiments." (p. 4)

> "Table 2: Inference time, FPS, and speedup for ISIC classification and segmentation models for CPU and FPGA." (p. 4)

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

**Evidence:**  
> "Hardware support We run our experiments on a Intel i7-7800-X CPU at 3.45GHz and a Xilinx ALVEO U200 FPGA board." (p. 3)

> "The accelerator is implemented with CPI and CPO factors of 4 and uses 32-bit floating-point precision arithmetic... In particular, the quantized model requires 8-bit integer weights and 32-bit integer activations and bias." (p. 4)

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
> "Table 1 shows some details of the models used in our experiments." (p. 4)

> "Table 2: Inference time, FPS, and speedup for ISIC classification and segmentation models for CPU and FPGA." (p. 4)

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Results achieved demonstrate the approach’s effectiveness and suggest new deployments and support in the platform for combined pruning and quantization strategies." (p. 4)

> "Our evaluation demonstrates that quantized and pruned models can largely benefit performance when combined with HLSinf." (p. 1)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Execution time is reduced by a factor of 2.59... The execution time is reduced by a factor of 6.22 (229.93 ms for inference)." (p. 4)

> "Results achieved demonstrate the approach’s effectiveness and suggest new deployments and support in the platform for combined pruning and quantization strategies." (p. 4)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "In this paper, we contribute with an open source platform to develop customized accelerators on FPGAs specifically tailored for pruned and quantized models." (p. 1)

> "Our evaluation demonstrates that quantized and pruned models can largely benefit performance when combined with HLSinf." (p. 1)

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
> "With HLSinf, significant inference speedups can be obtained for typical medical image-based applications." (p. 1)

> "Results achieved demonstrate the approach’s effectiveness and suggest new deployments and support in the platform for combined pruning and quantization strategies." (p. 4)

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[x] No  
[ ] Yes  
[ ] Not applicable  
