![](https://img.shields.io/static/v1?label=python&message=>3.8&color=yellow)
# SeisWav2Vec: A Contrastive Self-Supervised Framework for Scalable Time-Domain Seismic Representation Learning
PyTorch implementation of "SeisWav2Vec" with Fairseq.

This implementation is currently designed for inference purposes only and is not intended for training.
Please note that the current version supports inference functionalities exclusively.

## Abstract
Earthquake monitoring and early warning systems are critical for disaster preparedness and risk mitigation. Inspired by the success of self-supervised pretraining in fields such as natural language and speech processing, we propose SeisWav2Vec, a self-supervised framework designed to learn generalizable representations from raw seismic waveforms without the need for labeled data. SeisWav2Vec adopts a contrastive masked predic- tion objective to capture temporally rich features and is pretrained on large-scale unlabeled seismic datasets. Once pretrained, the model can be efficiently fine-tuned for a variety of downstream tasks, substantially reducing the dependency on annotated data. To evaluate its effectiveness, we apply SeisWav2Vec to two representative tasks: P-phase picking and earthquake magnitude estimation. Experimental results show that our fine-tuned models not only outperform advanced baselines but also maintain robust performance even under extremely low-resource conditions, using as little as 0.01% of labeled training data. Additionally, SeisWav2Vec exhibits strong generalization across domains and offers a scalable, label-efficient solution for seismic signal processing.

## Proposed Model
We propose a **Large Earthquake Model (LEM)** to evaluate the effectiveness of self-supervised learning in earthquake-related tasks, focusing on **p-phase picking** and **magnitude estimation**.
![LEM](/docs/model.png)

## Key Results
Our experiments show that LEM delivers competitive results compared to state-of-the-art models, achieving satisfactory performance using only **0.1% of the labeled data**. This approach offers a potential solution to reduce the time and effort required for seismic data labeling.

## Pre-trained models
You need to download the pre-trained weights and input the savepath of the weights into the required model to perform fine-tuning.

(New)Download Link: [download](https://drive.google.com/file/d/1sXjPTJ5Y8bNmJERgkAUZx7BLOTsrbeFP/view?usp=sharing)

(Old)Download Link: [download](https://drive.google.com/file/d/1QRpMPg4Q-gOQpfDoS5NbmiVzIMb6njS9/view?usp=sharing)

## Requirements and Installation
```
git clone https://github.com/Johnn9sb/Eq-Pretrain.git
```
+ [PyTorch](https://pytorch.org/) version >= 1.10.0
+ Python version >= 3.8
+ To install and develop this implementation locally:
```
cd Eq-Pretrain
pip install -r requirements.txt
pip install --editable ./model/fairseq/
```
+ If there is a need to test whether the environment installation is successful, the corresponding method can be found in the `load_model.py` file.

