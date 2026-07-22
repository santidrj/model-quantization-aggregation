# Quality Evaluation Questionnaire for Experimental Studies

## 1. Do the authors clearly state the aims of the research?

1.1) Do the authors state research questions, e.g., related to time-to-market, cost, product quality, process quality, developer productivity, and developer skills?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "As such, research on model quantisation for NMT tasks remains limited. We find that the model can be compressed at up to 4-bit precision without sacrificing quality." (p. 35)

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
> "English-to-German news translation task" (p. 38)

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
> "Transformer or RNN architectures" (abstract)

> "Our Transformer model consists of six encoder and six decoder layers... Our deep RNN model consists of eight layers of bidirectional LSTM." (p. 38)

2.5) If applicable, the software processes being used (e.g., a company standard process, the quality assurance procedures, the configuration management process)  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We first pre-train baseline models with both Transformer and RNN architectures." (p. 38)

> "We prepare our 4-bit quantisation model by re-training from a full precision model. We also store the quantisation errors to be considered for the next update." (p. 39)

---

## 3. Do the authors explain how experimental units were defined and selected?

3.1) Do the authors explain how experimental units were defined and selected?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We use systems for the WMT 2017 English-to-German news translation task for our experiment" (p. 38)

> "We first pre-train baseline models with both Transformer and RNN architectures." (p. 38)

3.2) Do the authors state to what degree the experimental units are representative?  
[x] No  
[ ] Yes  
[ ] Not applicable  

3.3) Do the authors explain why the experimental units they selected were the most appropriate for providing insight into the type of knowledge sought by the experiment?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We compare several compression approaches to our 4-bit logarithmic quantisation method" (p. 39)

> "The RNN model seems to be more robust towards the compression." (p. 39)

3.4) Do the authors report the sample size?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> Table 6 compares 5 bit-width conditions for each of the 2 model families, i.e. 10 reported experimental configurations in total. (p. 40)

---

## 4. Do the authors describe the design of the experiment?

4.1) Do the authors clearly describe the chosen design (blocking, within or between subject design, do treatments have levels)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "we sweep several bit widths." (p. 40)

> Table 6 compares Transformer and RNN under 32-, 4-, 3-, 2-, and 1-bit settings. (p. 40)

4.2) Do the authors define/describe all treatments and all controls?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "32-bit FP model (Baseline)" and "4-bit log model" are explicitly defined in Table 1. (p. 39)

> Table 6 reports the full-precision 32-bit baseline and the compressed models at lower bit widths. (p. 40)

---

## 5. Do the authors describe the data collection procedures and define the measures?

5.1) Are all measures clearly defined (e.g., scale, unit, counting rules)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Quality is measured based on BLEU (Papineni et al., 2002) score using sacreBLEU script (Post, 2018)." (p. 38)

> Table 6 reports "Size (rate)" and "BLEU(∆)". (p. 40)

5.2) Is the form of the data clear (e.g., tape recording, video material, notes, etc.)?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We use systems for the WMT 2017 English-to-German news translation task" (p. 38)

> "We use wmt2016 as the test set." (p. 39)

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
> "To minimise the mean squared encoding error, values should be quantised to the nearest centre." (p. 37)

> "We optimise S for each tensor independently." (p. 37)

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
> Tables 1, 2, 3, 4, 5, and 6 report descriptive performance statistics. (pp. 39-40)

> Table 6 reports model size, compression rate, BLEU, and BLEU delta for each bit width. (p. 40)

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
> "We first pre-train baseline models with both Transformer and RNN architectures." (p. 38)

> "The rest of the hyperparameter settings on both models follow the suggested configurations" (p. 39)

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
> "As shown in Table 6, model performance degrades with fewer bits being used." (p. 40)

> Table 6 presents the main comparison across bit widths and architectures. (p. 40)

9.2) Do the authors present conclusions clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We compress the model size in neural machine translation to approximately 7.9x smaller than 32-bit floats by using a 4-bit logarithmic quantisation." (p. 40)

> "4-bit precision performs better compared to the full-precision model with (near) 8x compression rate." (p. 40)

9.3) Are the conclusions warranted by the results and are the connections between the results and conclusions presented clearly?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "With 4-bit quantisation and uncompressed biases, we obtain a 7.9x compression rate." (p. 40)

> "Training an NMT system below 4-bit precision remains a challenge. As shown in Table 6, model performance degrades with fewer bits being used." (p. 40)

9.4) Do the authors discuss their conclusions in relation to the original research questions?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "We compress the model size in neural machine translation..." (p. 40)

> "We also find that re-training after quantisation is necessary..." (p. 40)

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
> "of interest for local deployment on mobile devices." (p. 40)

> "Should future hardware also support 4-bit instructions natively, 4-bit models could also improve decoding efficiency." (p. 40)

10.2) To what extent do authors interpret results in the context of other studies / the existing body of knowledge?  
[ ] No  
[x] Yes  
[ ] Not applicable  

**Evidence:**  
> "Logarithmic-based quantisation has been shown to perform better when compared to fixed-point quantisation using both architectures." (p. 39)

> "Our finding is in line with prior research..." (p. 39)
