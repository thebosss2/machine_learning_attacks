import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        x_orig = x.clone().detach()
        device = x.device

        # Initialize adversarial examples
        x_adv = x.clone().detach()

        # Optional: Random Initialization
        if self.rand_init:
            # Generate uniform noise in [-eps, eps]
            random_noise = torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
            # Add noise and immediately clamp/project to ensure initial validity
            x_adv = x_adv + random_noise
            # Clamp to valid image range [0, 1]
            x_adv = torch.clamp(x_adv, min=0., max=1.)
             # Project x_adv to be within eps-ball of x_orig
            x_adv = torch.max(torch.min(x_adv, x_orig + self.eps), x_orig - self.eps)
            # Detach after modification
            x_adv = x_adv.detach()

        # --- PGD iterations ---
        # Track which samples have already met the attack goal (for early stopping)
        batch_size = x.size(0)
        # Initialize all as not succeeded (False)
        has_succeeded = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Store the best adversarial example found so far for each sample (useful for early stopping)
        x_adv_best = x_adv.clone().detach()

        for i in range(self.n):

            # Create a mask for samples that still need processing
            active_mask = ~has_succeeded
            # If all samples have succeeded and early stopping is enabled, break
            if self.early_stop and not active_mask.any():
                # print(f"Early stopping at iteration {i}") # Optional debug
                break

            # Only compute gradients for active samples if possible (optimization, but complex)
            # Simpler: compute for all, but only update active ones
            x_adv_batch = x_adv.clone().detach().requires_grad_(True)

            # Forward pass through the model
            outputs = self.model(x_adv_batch)

            # Calculate the loss based on attack type (per sample)
            loss = self.loss_func(outputs, y)

            # Zero out previous gradients
            self.model.zero_grad()

            # Calculate gradients based on attack type
            if targeted:
                # Minimize loss w.r.t target label y
                # We want to take a step *down* the gradient
                objective = loss.sum() # Sum losses for backward pass
            else:
                # Maximize loss w.r.t true label y (minimize -loss)
                # We want to take a step *up* the gradient
                objective = (-loss).sum() # Sum losses for backward pass

            # Perform backpropagation
            objective.backward()

            # Get the gradient sign
            # Ensure grad exists even if requires_grad was False for some reason
            if x_adv_batch.grad is None:
                 continue # Should not happen if requires_grad_(True) worked
            grad_sign = x_adv_batch.grad.sign()

            # --- Update Step ---
            # Calculate the update based on targeted/untargeted goal
            if targeted:
                update = -self.alpha * grad_sign # Move opposite to gradient
            else:
                update = self.alpha * grad_sign # Move along gradient

            # Apply update only to samples that haven't succeeded yet
            x_adv[active_mask] = x_adv[active_mask] + update[active_mask]

            # --- Projection Steps ---
            # Project x_adv to stay within the epsilon-ball of x_orig
            # Calculate the perturbation relative to original
            eta = x_adv - x_orig
            # Clamp the perturbation to [-eps, eps]
            eta = torch.clamp(eta, min=-self.eps, max=self.eps)
            # Apply the clamped perturbation back to the original image
            x_adv = x_orig + eta

            # Project x_adv to be within the valid image range [0, 1]
            x_adv = torch.clamp(x_adv, min=0., max=1.)

            # Detach x_adv for the next iteration or final return
            x_adv = x_adv.detach()

            # --- Check Success & Update Best (for early stopping) ---
            if self.early_stop:
                with torch.no_grad():
                    # Evaluate the current adversarial examples
                    outputs_check = self.model(x_adv)
                    predicted_check = torch.argmax(outputs_check, dim=1)

                    # Check success condition based on attack type
                    if targeted:
                        current_success = (predicted_check == y)
                    else:
                        current_success = (predicted_check != y)

                    # Find samples that *newly* succeeded on this iteration
                    newly_succeeded_mask = current_success & (~has_succeeded)

                    # Update the master success tracker
                    has_succeeded = torch.logical_or(has_succeeded, current_success)

                    # Store the newly successful adversarial examples in x_adv_best
                    x_adv_best[newly_succeeded_mask] = x_adv[newly_succeeded_mask]


        # --- Final Assertions ---
        tolerance = 1e-6
        assert torch.all(x_adv >= 0. - tolerance) and torch.all(x_adv <= 1. + tolerance), \
            "Final x_adv values out of [0, 1] range"
        # Check L-inf distance from original using the final x_adv
        assert torch.all(torch.abs(x_adv - x_orig) <= self.eps + tolerance), \
            f"Final max perturbation {torch.max(torch.abs(x_adv - x_orig))} exceeds eps {self.eps}"

        # If early stopping, return the best adversarial example found for each sample
        # Otherwise, return the result from the last iteration
        return x_adv_best if self.early_stop else x_adv


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255., momentum=0.,
                 k=200, sigma=1 / 255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        pass  # FILL ME


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """

    def __init__(self, models, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        pass  # FILL ME
