import logging
import math
from typing import List

import torch
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class ParamScheduler:
    """
    A scheduler for hyperparameters of the optimizer, such as learning rate, betas, and group weights.
    This allows for dynamic adjustment of these parameters during training.
    """

    def __init__(
        self,
        epochs=20000,
        lr_scheduler=None,
        betas_scheduler=None,
        group_weights_scheduler=None,
        default_lr=1e-3,
        default_betas=(0.99, 0.99),
        default_group_weights=(0.5, 0.5),
    ):
        """
        Initializes the parameter scheduler.

        Args:
            epochs (int): The total number of training epochs.
            lr_scheduler (callable, optional): A function that computes the learning rate for the current epoch.
                It should take (current_epoch, max_epochs, grouped_losses) as input.
            betas_scheduler (callable, optional): A function that computes the betas for the current epoch.
                It should take (current_epoch, max_epochs, grouped_losses) as input.
            group_weights_scheduler (callable, optional): A function that computes the group weights for the current epoch.
                It should take (current_epoch, max_epochs, grouped_losses) as input.
            default_lr (float): The default learning rate if no scheduler is provided.
            default_betas (tuple): The default betas if no scheduler is provided.
            default_group_weights (tuple): The default group weights if no scheduler is provided.
        """
        self.max_epochs = epochs
        self.epochs = 0
        self.lr_scheduler = lr_scheduler
        self.betas_scheduler = betas_scheduler
        self.group_weights_scheduler = group_weights_scheduler
        self.default_lr = default_lr
        self.default_betas = default_betas
        self.default_group_weights = default_group_weights

    def lr(self):
        """Returns the current learning rate."""
        if self.lr_scheduler is not None:
            return self.lr_scheduler(self.epochs, self.max_epochs, self.grouped_losses)
        return self.default_lr

    def betas(self):
        """Returns the current beta values."""
        if self.betas_scheduler is not None:
            return self.betas_scheduler(self.epochs, self.max_epochs, self.grouped_losses)
        return self.default_betas

    def group_weights(self):
        """Returns the current group weights."""
        if self.group_weights_scheduler is not None:
            return torch.tensor(self.group_weights_scheduler(self.epochs, self.max_epochs, self.grouped_losses))
        return self.default_group_weights

    def step(self, losses, grouped_losses):
        """
        Advances the scheduler by one step. This should be called at each training step.
        
        Args:
            losses (list): A list of all individual losses.
            grouped_losses (list): A list of losses, grouped by specified indices.
        """
        self.epochs += 1
        self.losses = losses
        self.grouped_losses = grouped_losses


def sadam(
    params: List[Tensor], grads: List[List[Tensor]], exp_avgs: List[List[Tensor]], exp_avg_sqs: List[List[Tensor]],
    max_exp_avg_sqs: List[List[Tensor]], agg_exp_avg: List[Tensor], agg_exp_avg_sqs: List[Tensor], state_steps: List[int], *, amsgrad: bool,
    beta1: float, beta2: float, lr: float, weight_decay: float, eps: float, maximize: bool, group_weights: Tensor, agg_momentum: bool,
    agg_beta1: float, agg_beta2: float
):
    r"""Functional API that performs the MultiAdam algorithm computation.
    This function is a modification of the Adam optimizer to handle multiple groups of losses.
    Instead of a single gradient for each parameter, it takes a list of gradients, one for each loss group.
    These gradients are then used to compute individual momentum and velocity terms, which are finally
    aggregated using `group_weights` to produce a single update for each parameter.

    Args:
        params (List[Tensor]): list of parameters to optimize.
        grads (List[List[Tensor]]): list of gradients for each parameter, for each loss group.
        exp_avgs (List[List[Tensor]]): list of exponential moving averages of gradients.
        exp_avg_sqs (List[List[Tensor]]): list of exponential moving averages of squared gradients.
        max_exp_avg_sqs (List[List[Tensor]]): list of maximum exponential moving averages of squared gradients (for AMSGrad).
        agg_exp_avg (List[Tensor]): list of exponential moving averages of aggregated updates.
        agg_exp_avg_sqs (List[Tensor]): list of exponential moving averages of squared aggregated updates.
        state_steps (List[int]): list of steps for each parameter.
        amsgrad (bool): whether to use the AMSGrad variant of this algorithm.
        beta1 (float): coefficient for the first moment estimates.
        beta2 (float): coefficient for the second moment estimates.
        lr (float): learning rate.
        weight_decay (float): weight decay (L2 penalty).
        eps (float): term added to the denominator to improve numerical stability.
        maximize (bool): whether to maximize the objective function.
        group_weights (Tensor): a tensor of weights for each loss group.
        agg_momentum (bool): whether to use aggregated momentum.
        agg_beta1 (float): coefficient for the first moment of aggregated updates.
        agg_beta2 (float): coefficient for the second moment of aggregated updates.
    """

    # n_group is num of different group_weights
    # n_params is the number of all params
    n_groups, n_params = len(grads), len(grads[0])
    grads_cat, exp_avgs_cat, exp_avg_sqs_cat, max_exp_avg_sqs_cat = [], [], [], []

    # Reorganize the state tensors to be parameter-major instead of group-major.
    # This makes it easier to process all loss-specific states for a single parameter together.
    for i in range(n_params):
        grads_cat.append(torch.stack([grads[j][i] for j in range(n_groups)]))
        exp_avgs_cat.append(torch.stack([exp_avgs[j][i] for j in range(n_groups)]))
        exp_avg_sqs_cat.append(torch.stack([exp_avg_sqs[j][i] for j in range(n_groups)]))
        if amsgrad:
            max_exp_avg_sqs_cat.append(torch.stack([max_exp_avg_sqs[j][i] for j in range(n_groups)]))

    for i, param in enumerate(params):

        grad = grads_cat[i] if not maximize else -grads_cat[i]  # Gradients for the current parameter from all loss groups
        exp_avg = exp_avgs_cat[i]
        exp_avg_sq = exp_avg_sqs_cat[i]
        step = state_steps[i]

        # Bias correction terms for first and second moments.
        # These are standard in Adam to account for the initialization of moments at zero.
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            grad = grad.add(param.unsqueeze(0), alpha=weight_decay)

        # Decay the first and second moment running average coefficient for each loss group.
        # This is the standard Adam update rule, applied independently to each loss group's gradient.
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        if amsgrad:
            # AMSGrad variant: maintain the maximum of all 2nd moment running averages.
            torch.maximum(max_exp_avg_sqs_cat[i], exp_avg_sq, out=max_exp_avg_sqs_cat[i])
            # Use the max for normalizing the running average of the gradient.
            denom = (max_exp_avg_sqs_cat[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            # Standard Adam: use the current 2nd moment for normalization.
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        # Calculate the raw update for each loss group.
        update_raw = exp_avg / denom
        # Aggregate the updates from all loss groups using the provided group weights.
        update = (update_raw * group_weights.view((-1, ) + (1, ) * (exp_avg.dim() - 1))).sum(dim=0)  # weighted sum for current param

        # Optional: apply a second layer of momentum on the aggregated update.
        if agg_momentum:
            bias_correction1_, bias_correction2_ = 1 - agg_beta1**step, 1 - agg_beta2**step
            # Update aggregated momentum and velocity.
            agg_exp_avg[i].mul_(agg_beta1).add_(update, alpha=1 - agg_beta1)
            agg_exp_avg_sqs[i].mul_(agg_beta2).addcmul_(update, update.conj(), value=1 - agg_beta2)
            # Normalize the aggregated momentum.
            denom = (agg_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2_)).add_(eps)
            update = (agg_exp_avg[i] / bias_correction1_ / denom)

        # Apply the final update to the parameter.
        param -= step_size * update

    # Update the state tensors in place.
    # The reorganized tensors (e.g., exp_avgs_cat) are copied back to the original state lists.
    for i in range(n_groups):
        for j in range(n_params):
            exp_avgs[i][j].copy_(exp_avgs_cat[j][i])
            exp_avg_sqs[i][j].copy_(exp_avg_sqs_cat[j][i])
            if amsgrad:
                max_exp_avg_sqs[i][j].copy_(max_exp_avg_sqs_cat[j][i])


class MultiAdam(Optimizer):
    """
    Implements the MultiAdam algorithm, an Adam-like optimizer designed to handle multiple loss components.
    This is particularly useful in scenarios like Physics-Informed Neural Networks (PINNs), where the total loss
    is a sum of different terms (e.g., PDE residual, boundary conditions, initial conditions).
    MultiAdam computes gradients for each loss group separately and then aggregates them using a weighted sum
    to update the model parameters.

    This optimizer allows for separate tracking of Adam's moments (mean and variance) for each loss group.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.99, 0.99),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        maximize=False,
        loss_group_idx=None,
        group_weights=None,
        agg_momentum=False,
        agg_betas=None,
        *,
        param_scheduler=None,
    ):
        """
        Initializes the MultiAdam optimizer.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): learning rate (default: 1e-3).
            betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.99, 0.99)).
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8).
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this algorithm (default: False).
            maximize (bool, optional): maximize the params based on the objective, instead of minimizing (default: False).
            loss_group_idx (list, optional): A list of indices that define how to group the losses.
                For example, if `losses` is `[l1, l2, l3, l4]` and `loss_group_idx` is `[2]`,
                then the losses will be grouped into `[l1, l2]` and `[l3, l4]`.
            group_weights (list, optional): The weights for each loss group. If not provided, uniform weights are used.
            agg_momentum (bool, optional): If True, applies a second layer of momentum on the aggregated update (default: False).
            agg_betas (Tuple[float, float], optional): Betas for the aggregated momentum if `agg_momentum` is True.
            param_scheduler (ParamScheduler, optional): A scheduler to dynamically adjust hyperparameters during training.
                If provided, it overrides `lr`, `betas`, and `group_weights`.
        """
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))

        if agg_momentum:
            if agg_betas is None:
                raise ValueError('agg_betas should be provided when agg_momentum is True')
            if not 0.0 <= agg_betas[0] < 1.0:
                raise ValueError('Invalid beta parameter at index 0: {}'.format(agg_betas[0]))
            if not 0.0 <= agg_betas[1] < 1.0:
                raise ValueError('Invalid beta parameter at index 1: {}'.format(agg_betas[1]))
        else:
            agg_betas = (0, 0)

        self.is_init_state = True
        if loss_group_idx is not None:
            self.loss_group_idx = loss_group_idx
        else:
            self.loss_group_idx = []
            logger.warning('loss_group_idx is not provided, all losses are treated as one group')

        self.n_groups = len(self.loss_group_idx) + 1
        self.group_weights = 1 / self.n_groups * torch.ones([self.n_groups]) if group_weights is None else torch.tensor(group_weights)

        if param_scheduler is not None:
            logger.warning('lr, betas and group_weights are ignored when using param_scheduler')
        else:
            param_scheduler = ParamScheduler(default_lr=lr, default_betas=betas, default_group_weights=self.group_weights)
        self.param_scheduler = param_scheduler

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            agg_momentum=agg_momentum,
            agg_betas=agg_betas,
        )
        super(MultiAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MultiAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('agg_momentum', False)

    def init_states(self):
        """
        Initializes the optimizer's state. This is called on the first `step`.
        For each parameter, it initializes the moving averages for each loss group.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                # Exponential moving average of gradient values for each loss group
                state['exp_avg'] = [torch.zeros_like(p, memory_format=torch.preserve_format) for _ in range(self.n_groups)]
                # Exponential moving average of squared gradient values for each loss group
                state['exp_avg_sq'] = [torch.zeros_like(p, memory_format=torch.preserve_format) for _ in range(self.n_groups)]
                # Maintains max of all exp. moving avg. of sq. grad. values (for AMSGrad)
                state['max_exp_avg_sq'] = [torch.zeros_like(p, memory_format=torch.preserve_format) for _ in range(self.n_groups)]
                # States for aggregated momentum
                state['agg_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['agg_exp_avg_sqs'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        self.is_init_state = False

    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and returns a list of losses.
                The closure should also set `self.losses`.
        """
        # The closure is expected to compute the losses and store them in `self.losses`.
        # `skip_backward=True` is a hint that the closure should not perform the backward pass.
        with torch.enable_grad():
            _ = closure(skip_backward=True)
            losses = self.losses

            # Group the losses based on the provided indices.
            loss_group_idx = [0] + self.loss_group_idx + [len(losses)]
            grouped_losses = []
            for i in range(len(loss_group_idx) - 1):
                grouped_losses.append(torch.sum(losses[loss_group_idx[i]:loss_group_idx[i + 1]]))

        assert len(grouped_losses) == self.n_groups
        self.zero_grad()
        # Update the hyperparameters using the scheduler.
        self.param_scheduler.step(losses=self.losses, grouped_losses=grouped_losses)

        params_with_grad = []
        grads_groups = []
        exp_avgs_groups = []
        exp_avg_sqs_groups = []
        max_exp_avg_sqs_groups = []

        agg_exp_avg = []
        agg_exp_avg_sqs = []

        if self.is_init_state:
            self.init_states()

        # Compute gradients for each loss group separately.
        for i, loss in enumerate(grouped_losses):
            # `retain_graph=True` is necessary because we are doing multiple backward passes.
            loss.backward(retain_graph=True)

            for group in self.param_groups:
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []

                # For each parameter, store its gradient and the corresponding optimizer state for the current loss group.
                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        grads.append(p.grad.clone())
                        # Zero out the gradient to prepare for the next backward pass.
                        p.grad.zero_()

                        state = self.state[p]

                        exp_avgs.append(state['exp_avg'][i])
                        exp_avg_sqs.append(state['exp_avg_sq'][i])

                        if group['amsgrad']:
                            max_exp_avg_sqs.append(state['max_exp_avg_sq'][i])

                        if group['agg_momentum']:
                            agg_exp_avg.append(state['agg_exp_avg'])
                            agg_exp_avg_sqs.append(state['agg_exp_avg_sqs'])

                grads_groups.append(grads)
                exp_avgs_groups.append(exp_avgs)
                exp_avg_sqs_groups.append(exp_avg_sqs)
                max_exp_avg_sqs_groups.append(max_exp_avg_sqs)

        # Perform the optimization step using the collected gradients and states.
        with torch.no_grad():
            for group in self.param_groups:
                params_with_grad = []
                state_steps = []
                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        # update the steps for each param group update
                        self.state[p]['step'] += 1
                        state_steps.append(self.state[p]['step'])

                beta1, beta2 = self.param_scheduler.betas()
                agg_beta1, agg_beta2 = group['agg_betas']
                # Call the functional API to perform the actual update.
                sadam(
                    params_with_grad,  # list of params(which has grad)
                    # list[list[Tensor]]: dim0 is different loss_group,
                    # dim1 is grads of every params for different losses
                    grads_groups,
                    exp_avgs_groups,
                    exp_avg_sqs_groups,
                    max_exp_avg_sqs_groups,
                    agg_exp_avg,
                    agg_exp_avg_sqs,
                    state_steps,
                    amsgrad=group['amsgrad'],
                    beta1=beta1,
                    beta2=beta2,
                    lr=self.param_scheduler.lr(),
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize'],
                    group_weights=self.param_scheduler.group_weights(),
                    agg_momentum=group['agg_momentum'],
                    agg_beta1=agg_beta1,
                    agg_beta2=agg_beta2,
                )

        return grouped_losses