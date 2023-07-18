## E-NER: Evidential Deep Learning for Trustworthy Named Entity Recognition

This repository contains the code for our paper [E-NER: Evidential Deep Learning for Trustworthy Named Entity Recognition](http://arxiv.org/abs/2305.17854) (ACL Findings, 2023).

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

If you find this model helpful, please consider citing the following related papers:
@inproceedings{zhang-etal-2023-e,
    title = "{E}-{NER}: Evidential Deep Learning for Trustworthy Named Entity Recognition",
    author = "Zhang, Zhen  and
      Hu, Mengting  and
      Zhao, Shiwan  and
      Huang, Minlie  and
      Wang, Haotian  and
      Liu, Lemao  and
      Zhang, Zhirui  and
      Liu, Zhe  and
      Wu, Bingzhe",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.103",
    pages = "1619--1634",
    abstract = "Most named entity recognition (NER) systems focus on improving model performance, ignoring the need to quantify model uncertainty, which is critical to the reliability of NER systems in open environments. Evidential deep learning (EDL) has recently been proposed as a promising solution to explicitly model predictive uncertainty for classification tasks. However, directly applying EDL to NER applications faces two challenges, i.e., the problems of sparse entities and OOV/OOD entities in NER tasks. To address these challenges, we propose a trustworthy NER framework named E-NER by introducing two uncertainty-guided loss terms to the conventional EDL, along with a series of uncertainty-guided training strategies. Experiments show that E-NER can be applied to multiple NER paradigms to obtain accurate uncertainty estimation. Furthermore, compared to state-of-the-art baselines, the proposed method achieves a better OOV/OOD detection performance and better generalization ability on OOV entities.",
}
