# QANet

## Source:
This work is mainly coresponding to:
  * **paper**: [QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension](https://arxiv.org/abs/1804.09541),[DuReader- a Chinese Machine Reading Comprehension Dataset](https://arxiv.org/abs/1711.05073v2) <br>
  * **code**: a tensorflow implemention from [NLPLearn](https://github.com/NLPLearn/QANet), a tensorflow implemention from [SeanLee97](https://github.com/SeanLee97/QANet_dureader), a keras approach from [ewrfcas](https://github.com/ewrfcas/QANet_keras), a basic introduction for Reading Comprehension (RC) task from [facebookresearch](https://github.com/facebookresearch/DrQA) <br>

## Dataset
The dataset used for this work is mainly from [DuReader Dataset](http://ai.baidu.com/broad/subordinate?dataset=dureader).

## Requirements
  * Python>=3.5
  * TensorFlow>=1.5
  * NumPy
  * tqdm
  * ujson

## TODO
- [x] Simplize the code in model.py, layer.py, utils.py, main.py 
- [] Add prepare.py for training and testing DuReader Dataset
- [] Train and test the model
