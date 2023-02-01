# ViPubmed: Enriching Biomedical Knowledge for Low-resource Language Through Large-Scale Translation

## Introduction
The field of biomedical research heavily relies on data and benchmarks to advance our understanding of various health issues. However, such data and benchmarks are often limited in low-resource languages, including Vietnamese. To address this issue, we present a state-of-the-art English-Vietnamese translation model to translate large-scale biomedical data. This results in the creation of ViPubmedT5, a pretrained Encoder-Decoder Transformer model trained on 20 million translated biomedical abstracts from the PubMed corpus. The model achieves impressive results on biomedical summarization and acronym disambiguation benchmarks. Additionally, we also introduce ViMedNLI, a new NLP task in Vietnamese, translated from MedNLI and refined by human experts.

Refer to our [paper](https://arxiv.org/pdf/2210.05598.pdf) for more details


## Results
![image](ViPubmed/result1.png)
We benchmarked our ViPubmedT5 on three different tasks: FAQSum, acrDrAid and our brand-new dataset ViMedNLI.
![image](ViPubmed/result2.png)

## Code Usage
To train ViPubmedT5 model on the ... dataset
```
```

To eval ViPubmedT5 model on the ... dataset
```
```

### Released Datasets
Our released MedNLI and ViPubMed can be found here 

## How to Cite
```
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
We would like to thank Google for the support of Cloud credits and TPU quota!






