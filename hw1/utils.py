import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id < 0 or cnn_id > 2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model


class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """

    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model
    (a number in [0, 1]) on the labeled data returned by
    data_loader.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    all_x_adv_list = []
    all_y_list = []
    model = attack.model # Get model reference from attack object
    model.eval()     # Ensure model is in eval mode

    for x_batch, y_batch_true in data_loader:
        # Move data to the designated device
        x_batch_dev = x_batch.to(device)
        y_batch_true_dev = y_batch_true.to(device)

        if targeted:
            # Generate random target labels different from true labels
            # Ensure offset is between 1 and n_classes-1
            offset = torch.randint(1, n_classes, (y_batch_true_dev.size(0),), device=device)
            y_batch_target_dev = (y_batch_true_dev + offset) % n_classes

            # Execute targeted attack
            x_batch_adv = attack.execute(x_batch_dev, y_batch_target_dev, targeted=True)

            # Store results (move back to CPU for collection)
            all_x_adv_list.append(x_batch_adv.cpu().detach())
            all_y_list.append(y_batch_target_dev.cpu().detach()) # Store target labels
        else:
            # Execute untargeted attack
            x_batch_adv = attack.execute(x_batch_dev, y_batch_true_dev, targeted=False)

            # Store results (move back to CPU for collection)
            all_x_adv_list.append(x_batch_adv.cpu().detach())
            all_y_list.append(y_batch_true_dev.cpu().detach()) # Store true labels

    # Concatenate all batch results into single tensors
    final_x_adv = torch.cat(all_x_adv_list, dim=0)
    final_y = torch.cat(all_y_list, dim=0)

    return final_x_adv, final_y


def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    pass  # FILL ME


def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    model.eval()
    n_samples = x_adv.size(0)
    n_batches = int(np.ceil(n_samples / batch_size))
    successful_attacks = 0

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            x_batch = x_adv[start_idx:end_idx].to(device)
            y_batch = y[start_idx:end_idx].to(device)

            outputs = model(x_batch)
            predicted = torch.argmax(outputs, dim=1)

            if targeted:
                success = (predicted == y_batch).sum().item()
            else:
                success = (predicted != y_batch).sum().item()

            successful_attacks += success

    success_rate = successful_attacks / n_samples
    return success_rate


def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    pass  # FILL ME


def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    pass  # FILL ME


def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    pass  # FILL ME
