import torch
from torch.optim import Optimizer


# Pytorch Port of a previous tensorflow implementation in `tensorflow_probability`:
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/g3doc/api_docs/python/tfp/optimizer/StochasticGradientLangevinDynamics.md
class SGLD(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in eaach dimension
        according to RMSProp.
    """
    def __init__(self,
                 params,
                 lr=1e-2,
                 precondition_decay_rate=0.95,
                 num_pseudo_batches=1,
                 num_burn_in_steps=3000,
                 diagonal_bias=1e-8,
                 noise_switch = True,
                 weight_decay = 0.0) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.95`
        num_pseudo_batches : int, optional
            Effective number of minibatches in the data set.
            Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `3000`.
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-8`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=1e-8,
            noise_switch = noise_switch,
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            noise_switch = group['noise_switch']
            weight_decay = group['weight_decay']
            diagonal_bias = group["diagonal_bias"]
            num_pseudo_batches = group["num_pseudo_batches"]
            precondition_decay_rate = group["precondition_decay_rate"]
            num_burn_in_steps = group["num_burn_in_steps"]
            for parameter in group["params"]:
                
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                gradient = parameter.grad.data

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                #  }}} State initialization #

                state["iteration"] += 1
                momentum = state["momentum"]

                #  Momentum update {{{ #
                
                momentum.add_((1.0 - precondition_decay_rate) * ((gradient ** 2) - momentum))
                #  }}} Momentum update #

                if state["iteration"] > num_burn_in_steps:
                    sigma = torch.ones_like(parameter)
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = (
                    1. / torch.sqrt(momentum + diagonal_bias)
                )
                
                weight_decay_gradient = torch.zeros_like(parameter).add_(weight_decay, parameter.data)
                
                scaled_grad = (
                    0.5 * lr * preconditioner * gradient +
                    0.5 * lr / num_pseudo_batches * preconditioner * weight_decay_gradient)
                    
                if noise_switch :
                    scaled_grad += sigma * torch.sqrt(lr*preconditioner/num_pseudo_batches) * torch.normal(mean=torch.zeros_like(gradient),std=torch.ones_like(gradient))
                
                parameter.data.add_(- scaled_grad)

        return loss
