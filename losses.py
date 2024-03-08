import torch
import torch.nn.functional as F
import torch.distributions as td


def loss_bce(input, target):
    BCE = F.binary_cross_entropy(input, target, reduction="sum") / target.size(0)
    return BCE


def kl_divergence_balance(mu_1, var_1, mu_2, var_2, alpha=0.8, dim=1):
    p = td.Independent(td.Normal(mu_1, torch.sqrt(var_1)), dim)
    p_stop_grad = td.Independent(
        td.Normal(mu_1.detach(), torch.sqrt(var_1.detach())), dim
    )
    q = td.Independent(td.Normal(mu_2, torch.sqrt(var_2)), dim)
    q_stop_grad = td.Independent(
        td.Normal(mu_2.detach(), torch.sqrt(var_2.detach())), dim
    )
    div = alpha * td.kl_divergence(p_stop_grad, q) + (1 - alpha) * td.kl_divergence(
        p, q_stop_grad
    )
    div = torch.max(div, div.new_full(div.size(), 3))
    return torch.mean(div)


def loss_negloglikelihood(mu, target, var, dim):
    normal_dist = torch.distributions.Independent(
        torch.distributions.Normal(mu, var), dim
    )
    return -torch.mean(normal_dist.log_prob(target))
