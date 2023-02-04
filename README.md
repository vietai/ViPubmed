# ViPubmed: Enriching Biomedical Knowledge for Low-resource Language Through Large-Scale Translation
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg?style=flat-square)](http://makeapullrequest.com)
[![arXiv](https://img.shields.io/badge/arXiv-2210.05598-b31b1b.svg)](https://arxiv.org/abs/2210.05598)


## Overview
Biomedical data and benchmarks are highly valuable yet very limited in low-resource languages other than English, such as Vietnamese. In this paper, we use a state-of-the-art translation model in English-Vietnamese to translate and produce both pretrained and supervised data in the biomedical domains. Further, we release ViMedNLI - a new NLP task in Vietnamese translated from MedNLI using the recently public En-vi translation model and carefully refined by human experts.

Refer to our [paper](https://arxiv.org/pdf/2210.05598.pdf) for more details.

## Methods
### We large scale translate 20M Pubmed Abstract from English to Vietnamese and pretrained a biomedical Encoder-Decoder model on this translated dataset.
![image](https://user-images.githubusercontent.com/44376091/216741921-d3e64cf5-56d7-423b-a7ba-83f220dbf90b.png)

## 1. Pretrained Models (ViPubmedT5)
**Vocabulary:**
[ViT5_vocab](https://storage.googleapis.com/vietai_public/viT5/viT5_base_1024/spiece.model)

Model        | Gin File Location                                                                  | Checkpoint Location| Domain| Pretraining Corpus	
------------ | ---------------------------------------------------------------------------------- | -------------------| -------------------| -------------------
ViPubmedT5 Base | [ViT5_base.gin](https://github.com/justinphan3110/ViPubmed/blob/main/configs/t5/vit5_base.gin) | [gs://vietai_public/vipubmedt5_base/checkpoint_1500000](https://console.cloud.google.com/storage/browser/vietai_public/vipubmedt5_base) | Biomedical | [Translated ViPubmed](https://huggingface.co/datasets/VietAI/vi_pubmed)


## 2. Finetunning
### Finetunning example with T5X and Flaxformer:  [finetunning_vipubmedt5_example.ipynb](https://github.com/justinphan3110/ViPubmed/blob/main/example/finetunning_vipubmedt5_example.ipynb)


## 3. Released Datasets
- [ViMedNLI](https://github.com/justinphan3110/ViPubmed/tree/main/data/vi_mednli): A Natural Language Inference Dataset For The Vietnamese Clinical Domain
- [ViPubmed](https://huggingface.co/datasets/VietAI/vi_pubmed): 20M Vietnamese Biomedical abstracts generated by large scale translation

## Citation
If you find our work helpful, please cite the following:

```bib
@misc{https://doi.org/10.48550/arxiv.2210.05598,
  doi = {10.48550/ARXIV.2210.05598},
  url = {https://arxiv.org/abs/2210.05598},
  author = {Phan, Long and Dang, Tai and Tran, Hieu and Trinh, Trieu H. and Phan, Vy and Chau, Lam D. and Luong, Minh-Thang},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Enriching Biomedical Knowledge for Low-resource Language Through Large-Scale Translation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Acknowledgment
We would like to thank the Google TPU Research
Cloud (TRC) program and Soonson Kwon (Google
ML Ecosystem programs Lead) for their supports.





