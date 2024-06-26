import torch
import numpy as np

OutputResize = (224, 112)
Resize = (224, 224)
OriginalSize = (854, 480)

# mrf.py
# total_iter = 3
ICM_iter = 5
e_u_true_possibility = 0.7
log_e_u_true_possibility = np.log(e_u_true_possibility)
e_u_false_possibility = 1 - e_u_true_possibility
log_e_u_false_possibility = np.log(e_u_false_possibility)
# theta_u = 1 # 1
# theta_t = 3 # 3
# theta_s = 1.5 # 1.5
s_coefficient = 0.98
flow_range = 2
not_calculated_err = -20000
out_of_range_err = -10000
time_base = 0.96
time_min = 0.4

# train.py
device = 'cuda' if torch.cuda.is_available() else 'cpu'
augmentation_num = 512
learning_rate = 5e-4
step_size = 29
gamma = 0.9
train_epoch = 90
batch_size = 1
