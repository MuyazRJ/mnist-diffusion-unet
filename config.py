import torch
from util.embedding import get_timestep_embedding
from util.cosine_schedule import cosine_beta_schedule

# Training hyperparameters
EPOCHS = 101
BATCH_SIZE = 64
T_DIM = 64

training_data_path = 'mnist/train-images-idx3-ubyte/train-images-idx3-ubyte'

# Diffusion hyperparameters
T = 1000
beta_start = 0.0001
beta_end = 0.02

# Generate cosine betas
betas = cosine_beta_schedule(T)

# Compute alphas
alphas = 1.0 - betas

# Compute cumulative product of alphas
alpha_bars = torch.cumprod(alphas, dim=0)

# Timestep embeddings
t_embeddings = get_timestep_embedding(torch.arange(T), dim=T_DIM)
