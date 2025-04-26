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
        self.model.eval() # Ensure model is in evaluation mode
        x_orig = x.clone().detach() # Store original images
        x_adv = x.clone().detach() # Initialize adversarial examples
        device = x.device

        # Optional: Random Initialization
        if self.rand_init:
            noise = torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
            # Apply noise and immediately clamp/project to ensure initial validity
            x_adv = torch.clamp(x_adv + noise, 0., 1.)
            # Project x_adv to be within eps-ball of x_orig
            x_adv = torch.max(torch.min(x_adv, x_orig + self.eps), x_orig - self.eps)
            x_adv = x_adv.detach() # Detach after modification

        # PGD iterations
        for i in range(self.n):
            # Clone and set requires_grad for the current iteration's input
            x_adv_iter = x_adv.clone().detach().requires_grad_(True)

            # Forward pass
            outputs = self.model(x_adv_iter)
            # Calculate per-sample loss using the correct labels (y)
            loss_per_sample = self.loss_func(outputs, y)

            # Calculate the mean loss across the batch to get a scalar
            scalar_loss = loss_per_sample.mean() # <-- FIX: Get scalar loss

            # Determine objective based on attack type (for update direction, not backward call)
            # We always want the gradient of the actual mean loss
            objective_for_update = scalar_loss # Use mean loss for gradient calculation

            # Zero gradients, compute gradients via backward() on the scalar objective
            self.model.zero_grad()
            objective_for_update.backward() # <-- FIX: Call backward on the scalar loss

            # Ensure grad attribute exists
            if x_adv_iter.grad is None:
                print(f"Warning: No gradient computed for x_adv_iter at iteration {i}. Skipping update.")
                # Detach x_adv before next iteration if we skip update
                x_adv = x_adv.detach()
                continue

            # Get the sign of the gradient stored in the .grad attribute
            grad_sign = x_adv_iter.grad.sign()

            # Detach x_adv before modifying it based on the gradient
            x_adv = x_adv.detach()

            # Calculate the update step based on attack type
            if targeted:
                # Minimize loss: move opposite to the gradient direction
                update = -self.alpha * grad_sign
            else:
                # Maximize loss: move along the gradient direction
                # Ascending the gradient of the mean loss should generally increase loss
                update = self.alpha * grad_sign

            # Apply the update step
            x_adv = x_adv + update

            # --- Projection Steps ---
            # Project perturbation (eta) back into L-inf ball around x_orig
            eta = x_adv - x_orig
            eta = torch.clamp(eta, min=-self.eps, max=self.eps)
            # Apply clamped perturbation and clamp result to valid image range [0, 1]
            x_adv = torch.clamp(x_orig + eta, min=0., max=1.)

            # Detach for the next iteration or final return
            x_adv = x_adv.detach()

        # --- Final Assertions (Optional but Recommended) ---
        tolerance = 1e-6
        assert torch.all(x_adv >= 0. - tolerance) and torch.all(x_adv <= 1. + tolerance), \
            "Final x_adv values out of [0, 1] range"
        assert torch.all(torch.abs(x_adv - x_orig) <= self.eps + tolerance), \
            f"Final max perturbation {torch.max(torch.abs(x_adv - x_orig))} exceeds eps {self.eps}"

        # Return the result after n iterations (ignoring early stopping logic for now)
        return x_adv


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
