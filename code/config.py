import torch
import numpy as np

resize = (224, 224)
ori_size = (854, 480)

# mrf.py
e_u_true_possibility = 0.7
log_e_u_true_possibility = np.log(e_u_true_possibility)
e_u_false_possibility = 1 - e_u_true_possibility
log_e_u_false_possibility = np.log(e_u_false_possibility)
theta_u = 1
theta_t = 1
theta_s = 1
flow_range = 2
uncalculated_err = -20000
out_of_range_err = -10000
time_base = 0.96
time_min = 0.3

# train.py
device = 'cuda' if torch.cuda.is_available() else 'cpu'
augmentation_num = 32
learning_rate = 1e-3
step_size = 1
gamma = 0.8
train_epoch = 20
batch_size = 8

# eval.py
ICM_iter = 2
