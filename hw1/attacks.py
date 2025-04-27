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
        self.model.eval()
        x_orig = x.clone().detach()
        x_adv = x.clone().detach()
        device = x.device
        batch_size = x.size(0)

        if self.rand_init:
            noise = torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv + noise, 0., 1.)
            x_adv = torch.max(torch.min(x_adv, x_orig + self.eps), x_orig - self.eps)
            x_adv = x_adv.detach()

        has_succeeded = torch.zeros(batch_size, dtype=torch.bool, device=device)
        x_adv_best = x_adv.clone().detach()

        for i in range(self.n):
            active_mask = ~has_succeeded
            if self.early_stop and not active_mask.any():
                # add the following line to check if the early_stop works, thats how i tested it
                # print(f"Early stopping triggered at iteration {i+1}/{self.n}")
                break

            x_adv_iter = x_adv.clone().detach().requires_grad_(True)
            outputs = self.model(x_adv_iter)
            loss_per_sample = self.loss_func(outputs, y)
            scalar_loss = loss_per_sample.mean()

            self.model.zero_grad()
            scalar_loss.backward()

            if x_adv_iter.grad is None:
                print(f"Warning: No gradient computed for x_adv_iter at iteration {i}. Skipping update.")
                x_adv = x_adv.detach()
                continue

            grad_sign = x_adv_iter.grad.sign()
            x_adv = x_adv.detach()

            if targeted:
                update = -self.alpha * grad_sign
            else:
                update = self.alpha * grad_sign

            x_adv[active_mask] = x_adv[active_mask] + update[active_mask]

            eta = x_adv - x_orig
            eta = torch.clamp(eta, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x_orig + eta, min=0., max=1.)
            x_adv = x_adv.detach()

            if self.early_stop:
                 with torch.no_grad():
                    outputs_check = self.model(x_adv)
                    predicted_check = torch.argmax(outputs_check, dim=1)
                    if targeted:
                        current_success = (predicted_check == y)
                    else:
                        current_success = (predicted_check != y)
                    newly_succeeded_mask = current_success & (~has_succeeded)
                    has_succeeded = torch.logical_or(has_succeeded, current_success)
                    x_adv_best[newly_succeeded_mask] = x_adv[newly_succeeded_mask]

        tolerance = 1e-6
        final_result = x_adv_best if self.early_stop else x_adv
        assert torch.all(final_result >= 0. - tolerance) and torch.all(final_result <= 1. + tolerance), \
            "Final x_adv values out of [0, 1] range"
        assert torch.all(torch.abs(final_result - x_orig) <= self.eps + tolerance), \
            f"Final max perturbation {torch.max(torch.abs(final_result - x_orig))} exceeds eps {self.eps}"

        return final_result


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

    def _nes_gradient_estimate(self, x_adv, y, n_queries_per_sample):
        """
        Estimates the gradient using NES with antithetic sampling.
        Returns the estimated gradient and updates query counts.
        """
        batch_size = x_adv.size(0)
        channels = x_adv.size(1)
        height = x_adv.size(2)
        width = x_adv.size(3)
        device = x_adv.device

        # Initialize gradient estimate tensor
        grad_estimate = torch.zeros_like(x_adv, device=device)

        self.model.eval() # Ensure model is in eval mode for queries

        with torch.no_grad(): # No gradients needed for model queries
            for _ in range(self.k):
                # Generate Gaussian noise
                u = torch.randn_like(x_adv, device=device)
                x_query_plus = x_adv + self.sigma * u
                x_query_minus = x_adv - self.sigma * u
                x_query_plus = torch.clamp(x_query_plus, 0., 1.)
                x_query_minus = torch.clamp(x_query_minus, 0., 1.)
                outputs_plus = self.model(x_query_plus)
                outputs_minus = self.model(x_query_minus)
                
                loss_plus = self.loss_func(outputs_plus, y)
                loss_minus = self.loss_func(outputs_minus, y)
                loss_diff_reshaped = (loss_plus - loss_minus).view(-1, 1, 1, 1)
                grad_estimate += loss_diff_reshaped * u

            # Final gradient estimate (average over k samples, scale by 1/(2*sigma))
            grad_estimate = grad_estimate / (2. * self.k * self.sigma)

        return grad_estimate, n_queries_per_sample

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
        self.model.eval()
        x_orig = x.clone().detach()
        x_adv = x.clone().detach()
        device = x.device
        batch_size = x.size(0)

        n_queries_per_sample = torch.zeros(batch_size, dtype=torch.float32, device='cpu')

        if self.rand_init:
            noise = torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv + noise, 0., 1.)
            x_adv = torch.max(torch.min(x_adv, x_orig + self.eps), x_orig - self.eps)
            x_adv = x_adv.detach()

        has_succeeded = torch.zeros(batch_size, dtype=torch.bool, device=device)
        x_adv_best = x_adv.clone().detach()

        momentum_grad = torch.zeros_like(x_adv, device=device)

        for i in range(self.n):
            active_mask = ~has_succeeded
            if self.early_stop and not active_mask.any():
                # print(f"BB Early stopping at iteration {i+1}/{self.n}") # Optional debug
                break

            # NES Gradient Estimation
            grad_est, n_queries_per_sample = self._nes_gradient_estimate(x_adv, y, n_queries_per_sample)
            # practicly right now I run the model on successful attacks as well for code siplicity, but if 
            # I needed to use minimum queries really, I wouldn't run them so I don't envolve them in the count
            n_queries_per_sample += 2 * self.k * ~has_succeeded.to("cpu")

            # Incorporate momentum
            # Normalize gradient estimate (using L1 norm)
            grad_flat = grad_est.view(batch_size, -1)
            l1_norm = torch.linalg.norm(grad_flat, ord=1, dim=1)
            norm_reshaped = l1_norm.view(-1, 1, 1, 1) + 1e-12 # Added epsilon for stability to avoid dividing by 0
            normalized_grad = grad_est / norm_reshaped
            momentum_grad = self.momentum * momentum_grad + normalized_grad

            grad_sign = momentum_grad.sign()
            x_adv = x_adv.detach()

            if targeted:
                update = -self.alpha * grad_sign
            else:
                update = self.alpha * grad_sign
            x_adv[active_mask] = x_adv[active_mask] + update[active_mask]

            eta = x_adv - x_orig
            eta = torch.clamp(eta, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x_orig + eta, min=0., max=1.)
            x_adv = x_adv.detach()

            if self.early_stop:
                 with torch.no_grad():
                    outputs_check = self.model(x_adv)
                    n_queries_per_sample += 1 # I think this counts as part of the queries count, right?
                    predicted_check = torch.argmax(outputs_check, dim=1)
                    if targeted:
                        current_success = (predicted_check == y)
                    else:
                        current_success = (predicted_check != y)
                    newly_succeeded_mask = current_success & (~has_succeeded) # Nice trick ha?
                    has_succeeded = torch.logical_or(has_succeeded, current_success)
                    x_adv_best[newly_succeeded_mask] = x_adv[newly_succeeded_mask]

        tolerance = 1e-6
        final_result = x_adv_best if self.early_stop else x_adv
        assert torch.all(final_result >= 0. - tolerance) and torch.all(final_result <= 1. + tolerance)
        assert torch.all(torch.abs(final_result - x_orig) <= self.eps + tolerance)

        return final_result, n_queries_per_sample.cpu()


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
        for model in self.models:
            model.eval()

        x_orig = x.clone().detach()
        x_adv = x.clone().detach()
        device = x.device
        batch_size = x.size(0)

        if self.rand_init:
            noise = torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv + noise, 0., 1.)
            x_adv = torch.max(torch.min(x_adv, x_orig + self.eps), x_orig - self.eps)
            x_adv = x_adv.detach()

        has_succeeded_all = torch.zeros(batch_size, dtype=torch.bool, device=device)
        x_adv_best = x_adv.clone().detach()

        for i in range(self.n):
            active_mask = ~has_succeeded_all
            if self.early_stop and not active_mask.any():
                break

            x_adv_iter = x_adv.clone().detach().requires_grad_(True)
            accumulated_grad = torch.zeros_like(x_adv_iter, device=device)

            # Accumulate gradients from all models
            for model in self.models:
                model.zero_grad()
                outputs = model(x_adv_iter)
                loss_per_sample = self.loss_func(outputs, y)
                scalar_loss = loss_per_sample.mean()
                scalar_loss.backward()
                if x_adv_iter.grad is not None:
                    accumulated_grad += x_adv_iter.grad.detach()
                    x_adv_iter.grad.zero_()

            avg_grad = accumulated_grad / len(self.models)
            grad_sign = avg_grad.sign()
            x_adv = x_adv.detach()

            if targeted:
                update = -self.alpha * grad_sign
            else:
                update = self.alpha * grad_sign

            x_adv[active_mask] = x_adv[active_mask] + update[active_mask]
            eta = x_adv - x_orig
            eta = torch.clamp(eta, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x_orig + eta, min=0., max=1.)
            x_adv = x_adv.detach()

            # Early stopping check: success against all models required
            if self.early_stop:
                 with torch.no_grad():
                    current_success_all_models = torch.ones(batch_size, dtype=torch.bool, device=device)
                    for model in self.models:
                        outputs_check = model(x_adv)
                        predicted_check = torch.argmax(outputs_check, dim=1)
                        success_this_model = (predicted_check == y) if targeted else (predicted_check != y)
                        current_success_all_models &= success_this_model
                    newly_succeeded_mask = current_success_all_models & (~has_succeeded_all) # again, cool oparation
                    has_succeeded_all = torch.logical_or(has_succeeded_all, current_success_all_models)
                    x_adv_best[newly_succeeded_mask] = x_adv[newly_succeeded_mask]

        tolerance = 1e-6
        final_result = x_adv_best if self.early_stop else x_adv
        assert torch.all(final_result >= 0. - tolerance) and torch.all(final_result <= 1. + tolerance)
        assert torch.all(torch.abs(final_result - x_orig) <= self.eps + tolerance)

        return final_result
