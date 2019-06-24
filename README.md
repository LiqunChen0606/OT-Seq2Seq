# OT-Seq2Seq
This is the repository for ICLR 2019 paper [IMPROVING SEQUENCE-TO-SEQUENCE LEARNING
VIA OPTIMAL TRANSPORT](https://arxiv.org/pdf/1901.06283.pdf)

## Usage ##
Folder [nmt](./nmt) is built upon [GoogleNMT](https://github.com/tensorflow/nmt).
Please follow the instructions in that repo for dataset downloading and code testing.

Folder [texar](./texar) is built upon [Texar](https://github.com/asyml/texar).
Details about summarization and translation tasks, please follow this [link](./texar).

## Brief introduction ##
![Model intuition](./image/draw2.jpg)
We present a novel Seq2Seq learning scheme that leverages optimal transport (OT) to construct sequence-level  loss.   Specifically,  the  OT  objective  aims  to  find  an  optimal  matching  of  similarwords/phrases between two sequences, providing a way to promote their semantic similarity (Kusneret al., 2015). Compared with the above RL and adversarial schemes, our approach has: (i) semantic-invariance, allowing better preservation of sequence-level semantic information; and (ii) improved robustness, since neither the reinforce gradient nor the mini-max game is involved. The OT loss allows end-to-end supervised training and acts as an effective sequence-level regularization to the MLE loss.

The training algorithm can be represented as:
![Model algorithm](./image/alg.JPG)