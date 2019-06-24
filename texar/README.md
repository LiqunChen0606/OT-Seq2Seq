# Sequence Generation#

This example provide implementations of some classic and advanced training algorithms that tackles the exposure bias. The base model is an attentional seq2seq.

* **Maximum Likelihood (MLE)**: attentional seq2seq model with maximum likelihood training.
* **Maximum Likelihood (MLE) + Optimal transport (OT)**: Described in [OT-seq2seq](https://arxiv.org/pdf/1901.06283.pdf) and we use the sampling approach (n-gram replacement) by [(Ma et al., 2017)](https://arxiv.org/abs/1705.07136).

## Usage ##

### Dataset ###

Two example datasets are provided:

  * iwslt14: The benchmark [IWSLT2014](https://sites.google.com/site/iwsltevaluation2014/home) (de-en) machine translation dataset, following [(Ranzato et al., 2015)](https://arxiv.org/pdf/1511.06732.pdf) for data pre-processing.
  * gigaword: The benchmark [GIGAWORD](https://catalog.ldc.upenn.edu/LDC2003T05) text summarization dataset. we sampled 200K out of the 3.8M pre-processed training examples provided by [(Rush et al., 2015)](https://www.aclweb.org/anthology/D/D15/D15-1044.pdf) for the sake of training efficiency. We used the refined validation and test sets provided by [(Zhou et al., 2017)](https://arxiv.org/pdf/1704.07073.pdf).

Download the data with the following commands:

```
python utils/prepare_data.py --data iwslt14
python utils/prepare_data.py --data giga
```

### Train the models ###

#### Baseline Attentional Seq2seq with OT

```
python baseline_seq2seq_attn_main.py \
    --config_model configs.config_model \
    --config_data configs.config_iwslt14
```

Here:
  * `--config_model` specifies the model config. Note not to include the `.py` suffix.
  * `--config_data` specifies the data config.

[configs.config_model.py](./configs/config_model.py) specifies a single-layer seq2seq model with Luong attention and bi-directional RNN encoder. Hyperparameters taking default values can be omitted from the config file. 


For demonstration purpose, [configs.config_model_full.py](./configs/config_model_full.py) gives all possible hyperparameters for the model. The two config files will lead to the same model.

## Results ##

### Machine Translation
| Model      | BLEU Score   |
| -----------| -------|
| MLE        | 26.44 ± 0.18  |
| Scheduled Sampling   | 26.76  ± 0.17  |
| RAML | 27.22  ± 0.14  |
| Interpolation | 27.82  ± 0.11  |
| MLE + OT | 27.79 ± 0.12  |

### Text Summarization
| Model      | Rouge-1   | Rouge-2 | Rouge-L |
| -----------| -------|-------|-------|
| MLE        | 36.11 ± 0.21  | 16.39 ± 0.16 | 32.32 ± 0.19 |
| RAML | 36.30  ± 0.24 | 16.69 ± 0.20 | 32.49 ± 0.17 |
| Interpolation | 36.72  ± 0.29  |16.99 ± 0.17 | 32.95 ± 0.33|
| MLE + OT | 36.82 ± 0.25 | 17.35 ± 0.10 | 33.35 ± 0.14 |

 
