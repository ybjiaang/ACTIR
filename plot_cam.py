import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from misc import create_DF

# seaborn stuff
err_sty = 'bars'

# hyper parameters
lr = 0.01
n_finetune_loop = 20
resnet_dim = 128
dataset = 'camelyon17'


n_fine_tune_points = [0, 50, 100, 150 , 200, 250, 300]
dir = 'saved_model/' + dataset + '_full/'


fig = plt.figure()
plt.clf()
for model_name in ['erm', 'irm', 'maml', 'anti_causal']:
    real_lr = lr
    if dataset == 'vlcs' and model_name == 'maml':
        real_lr = 0.001
    avgs = np.zeros((5, len(n_fine_tune_points)))
    for random_seed in [0, 1, 2, 3, 4]:
        full_dir = dir + str(random_seed) + '_' + dataset +  '/saved_npy'
        filename = model_name + "_fine_lr_" + str(real_lr) + "_fine_nloops_" + str(n_finetune_loop)+".npy"
        array = np.load(full_dir + '/' + filename)
        avg = np.average(array, axis=1)
        avgs[random_seed, :] = avg
        print(model_name, random_seed)
        print(avg)


    df = create_DF(avgs, np.array(n_fine_tune_points))
    if model_name == "anti_causal":
        model_name = "Causal"
    if model_name == "erm":
        model_name = "ERM"
    if model_name == "irm":
        model_name = "IRM"
    if model_name == "maml":
        model_name = "MAML"
    sns.lineplot(x='num of finetuning points', y='finetuned accuary', err_style=err_sty, data = df, ci=68, label = model_name)


# other plot stuff
ax = plt.gca()
plt.xlabel('# Fine-tuning Points', fontsize=20)
plt.ylabel('Fine-tuned Accuary', fontsize=20)
plt.setp(ax.get_xticklabels(), fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=10)
plt.tight_layout()
plt.xticks(np.array(n_fine_tune_points))
plt.legend(loc=4, fontsize=15, title='Algo')
# plt.ylim([0, 1])

fig_name = dataset + ".png"

plt.savefig(fig_name)