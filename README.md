# CFBoost

## Basic Usage
- Change the experimental settings in `main_config.cfg` and the model hyperparameters in `model_config`. </br>
- Run `main.py` to train and test models. </br>
- Command line arguments are also acceptable with the same naming in configuration files. (Both main/model config)

For example: ```python main.py --model_name MultVAE --lr 0.001```

## Running LOCA
Before running LOCA, you need (1) user embeddings to find local communities and (2) the global model to cover users who are not considered by local models. </br>

1. Run single MultVAE to get user embedding vectors and the global model: 

`python main.py --model_name MultVAE` 

2. Train LOCA with the specific backbone model:

`python main.py --model_name LOCA_VAE` 

## Running TALL (MoE)
`python main.py --model_name MF_adaboost`

---

## Requirements
- Python 3.7 or higher
- Torch 1.5 or higher
