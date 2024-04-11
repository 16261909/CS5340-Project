import torch


resize = 224

time_base = 0.9
time_min = 0.3

augmentation_num = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# train.py
learning_rate = 1e-3
step_size = 20
gamma = 0.8
train_epoch = 5
batch_size = 8
