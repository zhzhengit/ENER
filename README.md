## E-NER: Evidential Deep Learning for Trustworthy Named Entity Recognition

This repository contains the code for our paper [Evidential Deep Learning for Trustworthy Named Entity Recognition) (ACL Findings, 2023).

## Overview

We study the problem of trustworthy NER by leveraging evidential deep learning. To address the issues of sparse entities and OOV/OOD entities, we propose E-NER with two uncertainty-guided loss terms. The uncertainty estimation quality of E-NER is improved without harming performance. Additionally, the well-qualified uncertainties contribute to detecting OOV/OOD, generalization, and sample selection.


Run the following script to install the dependencies,
```
pip3 install -r requirements.txt
```

## Dataset

The download links of the datasets used in this work are shown as follows:
- [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- [CONLL-2023-Typos&oov](https://github.com/BeyonderXX/MINER)
- [TwitterNER](https://github.com/BeyonderXX/MINER)
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
- [WNUT-2016](http://noisy-text.github.io/2016/ner-shared-task.html)
- [WNUT-2017](http://noisy-text.github.io/2017/emerging-rare-entities.html)

## Prepare Models

For [SpanNER](https://github.com/neulab/spanner) and [BERT-Tagger](), we use [BERT-Large](https://github.com/google-research/bert).
For Seq2Seq Model (https://github.com/yhcc/BARTNER), we use [BART-Large](https://paperswithcode.com/paper/bart-denoising-sequence-to-sequence-pre)


