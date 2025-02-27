# Combating Heterogeneous Model Biases in Recommendations via Boosting

## Basic Usage
- Change the experimental settings in `main_config.cfg` and the model hyperparameters in `model_config`. </br>
- Run `main.py` to train and test models. </br>
- Command line arguments are also acceptable with the same naming in configuration files. (Both main/model config)

For example: ```python main.py --model_name MultVAE --lr 0.001```

## Running LOCA
Before running LOCA, you need (1) user embeddings to find local communities and (2) the global model to cover users who are not considered by local models. </br>

1. Run a single MultVAE to get user embedding vectors and the global model: 

`python main.py --model_name MultVAE` 

2. Train LOCA with the specific backbone model:

`python main.py --model_name LOCA_VAE` 

## Running CFBoost and CFAdaboost
Change different designs of &alpha;, design1 and design2 in the code.

`python main.py --model_name MF_adaboost`

---

## Requirements
- Python 3.7 or higher
- Torch 1.5 or higher

## Appendix
Complete Appendix can be found [here](https://github.com/JP-25/CFBoost/blob/main/Appendix.pdf)

## Citation
cited papaer:
```
@inproceedings{10.1145/3701551.3703505,
author = {Pan, Jinhao and Caverlee, James and Zhu, Ziwei},
title = {Combating Heterogeneous Model Biases in Recommendations via Boosting},
year = {2025},
isbn = {9798400713293},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3701551.3703505},
doi = {10.1145/3701551.3703505},
booktitle = {Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining},
pages = {222–231},
numpages = {10},
keywords = {boosting, collaborative filtering, model biases, recommender systems},
location = {Hannover, Germany},
series = {WSDM '25}
}
```
