from utils import load_pretrained_cnn, TMLDataset, compute_accuracy, \
    compute_attack_success, run_whitebox_attack, run_blackbox_attack
import consts
import torch
import torchvision.transforms as transforms
from attacks import NESBBoxPGDAttack, PGDAttack
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
model = load_pretrained_cnn(0)
model.to(device)
model.eval()
dataset = TMLDataset(transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=consts.BATCH_SIZE)

# test accuracy
acc = compute_accuracy(model, data_loader, device)
print(f'The test accuracy of the model is: {acc:0.4f}')

# init attacks
wb_attack = PGDAttack(model)
bb_attack = NESBBoxPGDAttack(model)

# execute white-box
print('White-box attack:')
for targeted in [False, True]:
    x_adv, y = run_whitebox_attack(wb_attack, data_loader, targeted, device)
    sr = compute_attack_success(model, x_adv, y, consts.BATCH_SIZE, targeted, device)
    if targeted:
        print(f'\t- targeted success rate: {sr:0.4f}')
    else:
        print(f'\t- untargeted success rate: {sr:0.4f}')
        x_adv_untargeted_to_save = x_adv
        y_true_untargeted_to_save = y

# excecute targeted and untargeted black-box attacks w/ and wo/ momentum
n_queries_all = []
for momentum in [0, 0.9]:
    for targeted in [False, True]:
        bb_attack.momentum = momentum
        x_adv, y, n_queries = run_blackbox_attack(bb_attack, data_loader, targeted, device)
        sr = compute_attack_success(model, x_adv, y, consts.BATCH_SIZE, targeted, device)
        median_queries = torch.median(n_queries)
        if targeted:
            print(f'Targeted black-box attack (momentum={momentum:0.2f}):')
        else:
            print(f'Untargeted black-box attack (momentum={momentum:0.2f}):')
        print(f'\t- success rate: {sr:0.4f}\n\t- median(# queries): {median_queries}')
        n_queries_all.append(n_queries.detach().to('cpu'))

# box-plot # queries wo/ and w/ momentum for untargeted attacks
plt.figure()
plt.boxplot([n_queries_all[0], n_queries_all[2]])
plt.xticks(range(1, 3), ['0.0', '0.9'])
plt.title('untargeted')
plt.xlabel('momentum')
plt.ylabel('# queries')
plt.savefig('bbox-n_queries_untargeted.jpg')

# box-plot # queries wo/ and w/ momentum for targeted attacks
plt.figure()
plt.boxplot([n_queries_all[1], n_queries_all[3]])
plt.xticks(range(1, 3), ['0.0', '0.9'])
plt.title('targeted')
plt.xlabel('momentum')
plt.ylabel('# queries')
plt.savefig('bbox-n_queries_targeted.jpg')


# I added this peace of code to save images
# You can change it to true to save the images
DO_SAVE = False
if DO_SAVE:
    import os
    from  torchvision import utils as us
    print("Attempting to save untargeted attack examples...")
    num_examples_to_save = 5
    saved_count = 0
    save_dir = "saved_wb_untargeted_examples"

    if x_adv_untargeted_to_save is not None and y_true_untargeted_to_save is not None:
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            successful_indices = []
            predictions_on_adv = []
            for i in range(0, len(x_adv_untargeted_to_save), consts.BATCH_SIZE):
                x_batch_adv = x_adv_untargeted_to_save[i:i + consts.BATCH_SIZE].to(device)
                outputs = model(x_batch_adv)
                preds = torch.argmax(outputs, dim=1)
                predictions_on_adv.append(preds.cpu())
            all_preds_adv = torch.cat(predictions_on_adv)
            for idx in range(len(dataset)):
                true_label = y_true_untargeted_to_save[idx].item() # Get original true label
                predicted_label = all_preds_adv[idx].item() # Get prediction on adversarial example

                if predicted_label != true_label:
                    if saved_count < num_examples_to_save:
                        x_orig, _ = dataset[idx] 
                        x_adv_img = x_adv_untargeted_to_save[idx]
                        comparison_img = torch.cat((x_orig, x_adv_img), dim=2) # Concatenate horizontally
                        filename = f"untargeted_idx{idx}_orig{true_label}_adv{predicted_label}.png"
                        save_path = os.path.join(save_dir, filename)
                        us.save_image(comparison_img, save_path)
                        saved_count += 1
                    else:
                        break
        if saved_count > 0:
            print(f"Saved {saved_count} untargeted attack comparison examples to '{save_dir}'")
        else:
            print("No successful untargeted attacks found to save.")
    else:
        print("Untargeted attack results were not stored correctly.")