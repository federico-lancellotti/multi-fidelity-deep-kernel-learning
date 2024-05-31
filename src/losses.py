import torch
import torch.nn.functional as F
import torch.distributions as td


def loss_bce(input, target):
    """
    Compute the binary cross entropy loss between the input and target tensors.

    Args:
        input (Tensor): The input tensor.
        target (Tensor): The target tensor.

    Returns:
        Tensor: The computed binary cross entropy loss.
    """
    
    BCE = F.binary_cross_entropy(input, target, reduction="sum") / target.size(0)
    
    return BCE


def kl_divergence_balance(mu_1, var_1, mu_2, var_2, alpha=0.8, dim=1):
    """
    Compute the balanced Kullback-Leibler divergence between two sets of normal distributions.

    Args:
        mu_1 (torch.Tensor): Mean of the first set of normal distributions.
        var_1 (torch.Tensor): Variance of the first set of normal distributions.
        mu_2 (torch.Tensor): Mean of the second set of normal distributions.
        var_2 (torch.Tensor): Variance of the second set of normal distributions.
        alpha (float, optional): Weighting factor for the first set of distributions. Defaults to 0.8.
        dim (int, optional): Dimension along which the distributions are independent. Defaults to 1.

    Returns:
        torch.Tensor: Balanced KL divergence between the two sets of distributions.
    """
    
    p = td.Independent(td.Normal(mu_1, torch.sqrt(var_1)), dim)
    p_stop_grad = td.Independent(td.Normal(mu_1.detach(), torch.sqrt(var_1.detach())), dim)
    
    q = td.Independent(td.Normal(mu_2, torch.sqrt(var_2)), dim)
    q_stop_grad = td.Independent(td.Normal(mu_2.detach(), torch.sqrt(var_2.detach())), dim)
    
    div = alpha * td.kl_divergence(p_stop_grad, q) + (1 - alpha) * td.kl_divergence(p, q_stop_grad)
    div = torch.max(div, div.new_full(div.size(), 3))
    
    return torch.mean(div)


def loss_negloglikelihood(mu, target, var, dim):
    """
    Calculates the negative log-likelihood loss between the predicted mean `mu` and the target `target`.

    Args:
        mu (torch.Tensor): The predicted mean values.
        target (torch.Tensor): The target values.
        var (torch.Tensor): The variance values.
        dim (int): The dimension along which the independent distributions are defined.

    Returns:
        torch.Tensor: The negative log-likelihood loss.
    """
    
    normal_dist = torch.distributions.Independent(
        torch.distributions.Normal(mu, var), dim
    )
    
    return -torch.mean(normal_dist.log_prob(target))
