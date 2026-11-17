# Primary source reading record

Read and checked 2026-11-17. Bibliographic details were checked against paper or
publisher metadata. All literature discussion is paraphrased. No paper figure
has been copied; the report figures are original drawings based on project code
and recorded results. External model results are not compared numerically
without matching task, labels, data processing and selection protocol.

1. Pontiki et al., SemEval-2014 Task 4, pp. 27–35. Task definitions and the
   distinction between neutral and conflict. https://aclanthology.org/S14-2004/
2. Devlin et al., BERT, NAACL 2019, pp. 4171–4186. Pretraining and task fine-tuning.
   https://aclanthology.org/N19-1423/
3. Sun, Huang and Qiu, auxiliary sentence ABSA, NAACL 2019, pp. 380–385.
   Inspiration for pairs; our bare aspect query is not their full QA/NLI setup.
   https://aclanthology.org/N19-1035/
4. He, Gao and Chen, DeBERTaV3, ICLR 2023. Replaced token detection and gradient
   disentangled embedding sharing. https://openreview.net/pdf?id=sE7-XhLxHA
5. Chai et al., aspect-to-scope learning, Findings EMNLP 2023, pp. 10902–10913.
   Scope interference motivates a hypothesis, not a causal conclusion about our
   models. https://aclanthology.org/2023.findings-emnlp.727/
6. Cao, Mirjalili and Raschka, rank consistent ordinal regression, Pattern
   Recognition Letters 140, pp. 325–331, 2020. Our legacy ordinal code is not
   CORAL; official layer shares one linear projection and has threshold biases.
   https://arxiv.org/abs/1901.07884
   https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/layers.py
7. Guo, Pleiss, Sun and Weinberger, ICML 2017, pp. 1321–1330. Probability
   calibration concerns correctness probabilities, not validation F1 alone.
   https://proceedings.mlr.press/v70/guo17a.html
8. Saerens, Latinne and Decaestecker, Neural Computation 14(1), pp. 21–41, 2002.
   Prior adjustment is related mathematically but our one-scalar grid search
   does not establish label shift or reproduce their EM procedure.
   https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf
9. Cawley and Talbot, JMLR 11, pp. 2079–2107, 2010. Selection variance and
   selection bias; repeated test inspection limits confirmatory interpretation.
   https://jmlr.org/papers/v11/cawley10a.html
10. Cabello and Akujuobi, PFInstruct, Findings ACL 2024, pp. 6597–6610.
    Instruction approaches cover several ABSA subtasks, not directly comparable
    with this three-class gold-target experiment.
    https://aclanthology.org/2024.findings-acl.394/
11. NIST, Sign Test, Dataplot reference. Ties excluded; independence must still be
    justified. https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm

Figure provenance: architecture diagram is our schematic of implemented PairABSA;
seed chart and decision-rule chart use the curated recorded_results.json. No
third-party artwork, screenshot or copyrighted paper figure is included.
