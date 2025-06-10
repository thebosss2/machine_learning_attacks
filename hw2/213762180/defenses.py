import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler, \
                   eps, device, m=4, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)

    # init delta (adv. perturbation) - FILL ME
    delta = torch.zeros(batch_size, 3, 32, 32, device=device)

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    # train - FILLE ME
    model.train()
    for epoch in range(epochs):

        for i, (images, labels) in enumerate(loader_tr):
            images, labels = images.to(device), labels.to(device)

            # Get a slice of delta that matches the current batch size
            delta_slice = delta[:images.shape[0]] 

            # repeat the mini-batch m times
            for _ in range(m):
                delta_slice.requires_grad = True
                # apply perturbation to the images and clamp to valid range
                perturbed_image = torch.clamp(images + delta_slice, 0, 1)

                outputs = model(perturbed_image)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update the delta slice
                delta_grad = delta_slice.grad.detach()
                delta_slice = delta_slice.detach()
                delta_slice = delta_slice + eps * delta_grad.sign()
                delta_slice = torch.clamp(delta_slice, -eps, eps)
                # ensure the new delta doesn't push the image out of bounds
                delta_slice = torch.clamp(images + delta_slice, 0, 1) - images

                # write the updated slice back to the main delta tensor
                delta.data[:images.shape[0]] = delta_slice.data

        # step the learning rate scheduler once per epoch
        lr_scheduler.step()

    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        n_classes = 4 # cifar-10 classes
        total_counts = np.zeros(n_classes, dtype=int)
        with torch.no_grad():
            n_batches = int(np.ceil(n / batch_size))
            for _ in range(n_batches):
                current_batch_size = min(batch_size, n)
                n -= current_batch_size

                batch = x.repeat((current_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch) * self.sigma
                noisy_batch = batch + noise
                predictions = self.model(noisy_batch).argmax(1)
                total_counts += np.bincount(predictions.cpu().numpy(), minlength=n_classes)
        return total_counts
        
    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """
        
        # find prediction (top class c) - FILL ME
        base_counts = self._sample_under_noise(x, n0, batch_size)
        top_class = base_counts.argmax().item()


        # compute lower bound on p_c - FILL ME
        certify_counts = self._sample_under_noise(x, n, batch_size)
        n_c = certify_counts[top_class]

        # Use the Clopper-Pearson method to find the lower bound of the confidence interval for the true probability of predicting the top class.
        p_c_lower = proportion_confint(n_c, n, alpha=2 * alpha, method="beta")[0]

        if p_c_lower <= 0.5:
            return self.ABSTAIN, 0.0
        else:
            # Calculate the certified radius using the Gaussian CDF.
            radius = self.sigma * norm.ppf(p_c_lower)

        return top_class, radius
        

class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.04,
                 step_size=0.05, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME
        mask = torch.rand((1, self.dim[1], self.dim[2], self.dim[3]), device=device, requires_grad=True)
        trigger = torch.rand((1, self.dim[1], self.dim[2], self.dim[3]), device=device, requires_grad=True)

        optimizer = torch.optim.SGD([mask, trigger], lr=self.step_size)

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME
        iters_count = 0
        while iters_count < self.niters:
            for source_samples, source_labels in data_loader:
                if iters_count >= self.niters:
                    break

                # per the description, only use inputs not originally from the target class
                clean_inputs = source_samples[source_labels != c_t].to(device)
                if clean_inputs.shape[0] == 0:
                    continue

                # create a tensor of target labels
                targets = torch.full((clean_inputs.shape[0],), c_t, dtype=torch.long, device=device)

                # apply the trigger to the clean inputs
                stamped_inputs = torch.clamp((1 - mask) * clean_inputs + mask * trigger, 0, 1)
    
                # zero gradients, calculate loss, and take an optimization step
                optimizer.zero_grad()
                outputs = self.model(stamped_inputs)
                class_loss = self.loss_func(outputs, targets)
                mask_norm = torch.sum(torch.abs(mask))
                total_loss = class_loss + self.lambda_c * mask_norm
                total_loss.backward()
                optimizer.step()

                # clamp the mask and trigger to ensure they remain valid
                with torch.no_grad():
                    mask.clamp_(0, 1)
                    trigger.clamp_(0, 1)

                iters_count += 1

        # done
        return mask.detach(), trigger.detach()
