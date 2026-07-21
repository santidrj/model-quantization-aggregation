# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "The main contributions of this paper are summarized as follows." (p. 2)

> "Experiments were conducted on state-of-the-art LF-MMI CNN-TDNN and TDNN systems ... on two tasks, Switchboard telephone speech [81] and AMI meeting transcription [82]." (p. 2)

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
> "Switchboard telephone speech [81] and AMI meeting transcription [82]." (p. 2)

> "automatic speech recognition (ASR) systems" (p. 1)

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
> "LSTM-RNN and Transformer based neural LMs" (p. 1)

> "LF-MMI CNN-TDNN and TDNN systems" (p. 2)

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Various LSTM and Transformer LMs ... were used to rescore the 4-gram LM produced N-best lists (N = 20)." (p. 9)

> "All other experimental configurations remain the same as the Switchboard experiments of Section VI-A." (p. 11)

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "All the LSTM-RNN LMs investigated in this paper consist of 2 LSTM layers..." (p. 8)

> "All the Transformer LMs used in this paper contain 6 Transformer layers." (p. 8)

3.2) Do the authors state to what degree the experimental units are representative?  
[x] No  
[ ] Yes  
[ ] Not applicable  

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Experiments were conducted on state-of-the-art LF-MMI CNN-TDNN and TDNN systems..." (p. 2)

> "The proposed mixed precision quantization techniques achieved 'lossless' quantization on both tasks..." (p. 1)

3.4) Do the authors report the sample size?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "All the LSTM-RNN LMs investigated in this paper consist of 2 LSTM layers..." (p. 8)

> "All the Transformer LMs used in this paper contain 6 Transformer layers." (p. 8)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "All the mixed precision quantized LSTM-RNN LMs and Transformer LMs of this paper use layer or node level precision settings that are set either manually as equal bit-widths ... or automatically learned..." (p. 8)

> "Statistical significance test was conducted at level α = 0.05 based on matched pairs sentence segment word error (MAPSSWE)." (p. 8)

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "PERFORMANCE OF THE BASELINE FULL PRECISION (LM 1), UNIFORM PRECISION QUANTIZED (LM 2-11) AND MIXED PRECISION QUANTIZED..." (p. 9)

> "using KL, curvature (Hes) or NAS based mixed precision quantization methods" (pp. 9-11)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "All WER changes of no statistical significance (MAPSSWE, α = 0.05) ... are marked with '*'" (pp. 9-11)

> "Perplexity and convergence speed comparison" (p. 10)

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Table III" through "Table VI" report PPL, WER, model size, compression ratio and evaluation time. (pp. 9-11)

> "Table I" and "Table II" report additional descriptive comparisons. (pp. 6-7)

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
> "The first two approaches are based on quantization sensitivity metrics..." (p. 1)

> "The ADMM based optimization decomposes a dual ascent problem into alternating updates of two variables." (p. 4)

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
> "Table III" through "Table VI" report PPL, WER, model size, compression ratio and evaluation time. (pp. 9-11)

> "Table I" and "Table II" report additional descriptive comparisons. (pp. 6-7)

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
> "All NNLMs were trained using a single NVIDIA Tesla V100 Volta GPU card." (p. 8)

> "All other experimental configurations remain the same as the Switchboard experiments of Section VI-A." (p. 11)

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
> "Experimental results conducted on two state-of-the-art speech recognition tasks suggest..." (p. 11)

> "Table III" through "Table VI" present the main comparative results. (pp. 9-11)

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "This paper presents a set of novel mixed precision based neural network LM quantization techniques..." (p. 11)

> "can produce 'lossless' quantization and large model size compression ratios of up to around 16 times" (p. 11)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Experimental results ... suggest the proposed mixed precision neural network LM quantization methods outperform traditional uniform precision based quantization approaches" (p. 11)

> "while incurring no statistically significant recognition accuracy degradation." (p. 11)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "the first work in the speech technology community to apply mixed precision DNN quantization techniques to both LSTM-RNN and Transformer based NNLMs." (p. 2)

> "Transformer LM model size compression ratios of up to approximately 16 times..." (p. 2)

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
> "possible future research into alternative quantization methods and their application to other neural network components of speech recognition systems and end-to-end based neural architectures." (p. 11)

> "There is a pressing need of developing ultra-compact, low footprint language modelling methods..." (p. 1)

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "In contrast, prior researches within the speech community in this direction largely focused on uniform precision based quantization..." (p. 2)

> "To the best of our knowledge, this is the best low-bit Transformer language model compression ratio published so far..." (p. 3)
