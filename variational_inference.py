from abc import ABC
import torch
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood


class _ApproximateMarginalLogLikelihood(MarginalLogLikelihood, ABC):
    """
    Approximate Marginal Log Likelihood class.

    This class represents the approximate marginal log likelihood used in variational inference.
    It inherits from the `MarginalLogLikelihood` class and the `ABC` class.

    Args:
        likelihood: The likelihood function.
        model: The model used for inference.
        num_data: The number of data points.
        beta: The scaling factor for the KL divergence term (default: 1.0).
    """

    def __init__(self, likelihood, model, num_data, beta=1.0):
        """
        Initializes the ApproximateMarginalLogLikelihood class.
        
        Args:
            likelihood: The likelihood function.
            model: The model used for inference.
            num_data: The number of data points.
            beta: The scaling factor for the KL divergence term (default: 1.0).
        """
        super().__init__(likelihood, model)
        self.num_data = num_data
        self.beta = beta


    def forward(self, **kwargs):
        """
        Compute the forward pass of the approximate marginal log likelihood.

        This method calculates the approximate marginal log likelihood by combining the KL divergence term
        and the log prior term.

        Returns:
            The approximate marginal log likelihood.
        """

        # Get KL term
        kl_divergence = self.model.variational_strategy.kl_divergence().div(
            self.num_data / self.beta
        )

        # Log prior term
        log_prior = torch.zeros_like(kl_divergence)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        return -kl_divergence + log_prior


class VariationalKL(_ApproximateMarginalLogLikelihood):
    """
    This class represents the VariationalKL object, which is a subclass of _ApproximateMarginalLogLikelihood.
    It is used to compute the forward pass of the VariationalKL model.

    Args:
        **kwargs: Additional keyword arguments to be passed to the parent class.

    Returns:
        The result of the forward pass.

    """
    
    def forward(self, **kwargs):
        return super().forward(**kwargs)
