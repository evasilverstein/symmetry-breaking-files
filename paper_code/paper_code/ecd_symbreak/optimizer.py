"""
ECD_q1_scaled Optimizer

Energy Conserving Descent optimizer with q=1 scaling.
Physics-inspired optimizer based on Hamiltonian dynamics with energy conservation.

Key hyperparameters:
- lr: Learning rate (rescaled by sqrt(eta) internally)
- eta: Concentration parameter (controls result distribution)
- F0: Loss offset (typically F0 = Fmin - 1)
- nu: Bounce amplitude (adds chaos for ergodicity)
- consEn: Energy conservation mode
"""

from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch


class ECD_q1_scaled(Optimizer):
    """
    Energy Conserving Descent optimizer with q=1 scaling.

    Implements Hamiltonian dynamics with energy conservation for neural network
    optimization. The Liouville measure concentrates results toward low loss:
    p(Theta) ~ (F(Theta) - F0)^(-eta*d/2)

    Args:
        params: Model parameters to optimize
        lr: Learning rate (will be rescaled by 1/sqrt(eta))
        F0: Loss offset, should be < minimum achievable loss (default: -1)
        eps1: Numerical stability epsilon for normalization (default: 1e-10)
        eps2: Numerical stability epsilon for loss check (default: 1e-40)
        nu: Bounce amplitude for adding chaos (default: 0.1, rescaled by 1/sqrt(d))
        weight_decay: L2 regularization coefficient (default: 0)
        eta: Concentration parameter (default: required)
        consEn: Whether to use energy conservation (default: True)

    Memory: Uses 2N variables (Theta, momentum) vs 3N for Adam
    """

    def __init__(self, params, lr=required, F0=-1, eps1=1e-10, eps2=1e-40,
                 nu=0.1, weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, nu=nu,
                       weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.consEn = consEn
        self.eta = eta
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.Finit = 0.
        self.dim = 0
        super(ECD_q1_scaled, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step.

        Args:
            closure: A callable that re-evaluates the model and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute q^2 for L^2 weight decay
        self.q2 = torch.tensor(0., device=self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))

        # Initialization on first iteration
        if self.iteration == 0:
            # Define random number generator and set seed
            self.generator = torch.Generator(device=self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())

            # Initial value of the loss
            self.Finit = loss
            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0

            # Initialize momenta along (minus) the gradient
            # Note: by momenta we mean velocities
            p2init = torch.tensor(0., device=self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q
                        param_state["momenta"] = p
                    p2init.add_(torch.norm(p)**2)

            # Normalize initial momenta such that |p(0)| = 1
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(1/p2init))

            self.p2 = torch.tensor(1.0)

            # Rescale hyperparameters
            self.lr = self.lr / np.sqrt(self.eta)
            self.nu = self.nu / np.sqrt(self.dim)

        if loss + 0.5 * self.weight_decay * self.q2 - self.F0 > self.eps2:
            # Scaling factor for energy conservation
            if self.consEn == True:
                p2true = 1
                if torch.abs(p2true - self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)

            # Update momenta and compute p^2 needed for position update
            self.p2 = torch.tensor(0., device=self.param_groups[0]["params"][0].device)

            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data
                    p = param_state["momenta"]

                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        prefactor = -0.5 * self.lr * self.eta * self.dim / (self.dim - 1)
                        dotp = torch.dot(p.view(-1), (d_q + self.weight_decay * q).view(-1))
                        denom = loss + 0.5 * self.weight_decay * self.q2 - self.F0
                        p.add_((prefactor / denom) * ((d_q + self.weight_decay * q) - p * dotp))
                        self.p2.add_(torch.norm(p)**2)

            # Update positions and add tiny rotation of momenta
            p2new = torch.tensor(0., device=self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)

            for group in self.param_groups:
                for q in group["params"]:
                    # Update position
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * param_state["momenta"])

                    # Add noise to momenta
                    z = torch.randn(p.size(), device=p.device, generator=self.generator)
                    param_state["momenta"] = p / pnorm + self.nu * z
                    p2new += torch.dot(param_state["momenta"].view(-1),
                                       param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p / torch.sqrt(p2new)

            self.p2 = pnorm**2
            self.iteration += 1

        return loss
