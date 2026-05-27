import torch
from torch.optim import Optimizer


class Adam_LBFGS(Optimizer):
    def __init__(
        self,
        params,
        switch_epoch=10000,
        adam_lr=1e-3,  # New: Separate Adam LR
        adam_betas=(0.9, 0.999),  # Extracted from adam_param
        lbfgs_lr=1.0,  # New: Separate L-BFGS LR
        lbfgs_max_iter=20,  # Extract other params for flexibility
        lbfgs_tolerance_grad=1e-7,  # New: Expose tolerances
        lbfgs_tolerance_change=1e-9,
        lbfgs_history_size=100,
        lbfgs_line_search_fn="strong_wolfe",  # New: Expose line search
    ):
        self.params = list(params)
        self.switch_epoch = switch_epoch

        # Adam setup with separate LR
        self.adam = torch.optim.Adam(self.params, lr=adam_lr, betas=adam_betas)

        # L-BFGS setup with separate LR and params
        self.lbfgs = torch.optim.LBFGS(
            self.params,
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            tolerance_grad=lbfgs_tolerance_grad,
            tolerance_change=lbfgs_tolerance_change,
            history_size=lbfgs_history_size,
            line_search_fn=lbfgs_line_search_fn,
        )

        super().__init__(self.params, defaults={})

        self.state["current_step"] = 0

    def step(self, closure=None):
        self.state["current_step"] += 1

        if self.state["current_step"] < self.switch_epoch:
            self.adam.step(closure)
        else:
            self.lbfgs.step(closure)
            if self.state["current_step"] == self.switch_epoch:
                print(f"Switch to LBFGS at epoch {self.switch_epoch}")
