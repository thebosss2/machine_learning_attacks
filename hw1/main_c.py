import utils
import consts
import models
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

# GPU available?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load model and dataset
model = utils.load_pretrained_cnn(1).to(device)
model.eval()
dataset = utils.TMLDataset(transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=consts.BATCH_SIZE)

# model accuracy
acc_orig = utils.compute_accuracy(model, data_loader, device)
print(f'Model accuracy before flipping: {acc_orig:0.4f}')

# layers whose weights will be flipped
layers = {'conv1': model.conv1,
          'conv2': model.conv2,
          'fc1': model.fc1,
          'fc2': model.fc2,
          'fc3': model.fc3}

# flip bits at random and measure impact on accuracy (via RAD)
RADs_bf_idx = dict([(bf_idx, []) for bf_idx in range(32)])  # will contain a list of RADs for each index of bit flipped
RADs_all = []  # will eventually contain all consts.BF_PER_LAYER*len(layers) RADs
for layer_name in layers:
    layer = layers[layer_name]
    with torch.no_grad():
        W = layer.weight
        W.requires_grad = False
        for _ in range(consts.BF_PER_LAYER):
            # start of the fill me code
            rand_weight_idx_flat = random.randrange(W.numel())
            multi_dim_idx = np.unravel_index(rand_weight_idx_flat, W.shape)
            original_weight_val = W[multi_dim_idx].item()
            flipped_weight_val, bf_idx = utils.random_bit_flip(original_weight_val)
            W[multi_dim_idx] = flipped_weight_val # Update the weight
            acc_bf = utils.compute_accuracy(model, data_loader, device)
            rad = (acc_orig - acc_bf) / acc_orig
            W[multi_dim_idx] = original_weight_val # restore to original value so we don't destroy it
            # end of fill me code
            RADs_bf_idx[bf_idx].append(rad)
            RADs_all.append(rad)

# Max and % RAD>15%
RADs_all = np.array(RADs_all)
print(f'Total # weights flipped: {len(RADs_all)}')
print(f'Max RAD: {np.max(RADs_all):0.4f}')
print(f'RAD>15%: {np.sum(RADs_all > 0.15) / RADs_all.size:0.4f}')

# boxplots: bit-flip index vs. RAD
plt.figure()
# start of fill me code
boxplot_data = [RADs_bf_idx[bf_idx] for bf_idx in range(32)]
plt.boxplot(boxplot_data)
plt.xlabel('Bit-flip index (0-31)')
plt.ylabel('RAD')
plt.title('RAD Distribution per Bit-flip Index')
plt.xticks(ticks=np.arange(1, 33), labels=np.arange(0, 32))
plt.grid(axis='y', linestyle='--')
# end of fill me code 
plt.savefig('bf_idx-vs-RAD.jpg')
