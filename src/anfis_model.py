import itertools
import torch
from torch import nn

class NeuroFuzzySepsis(nn.Module):
    """First-order Sugeno ANFIS-style binary classifier."""

    def __init__(self, n_features=7, n_mfs=2):
        super().__init__()
        self.n_features = n_features
        centers = torch.tensor([[-0.8, 0.8]] * n_features, dtype=torch.float32)
        self.centers = nn.Parameter(centers)
        self.log_sigmas = nn.Parameter(torch.log(torch.ones(n_features, n_mfs)*0.9))
        rules = list(itertools.product(range(n_mfs), repeat=n_features))
        self.register_buffer("rules", torch.tensor(rules, dtype=torch.long))
        self.consequents = nn.Parameter(torch.zeros(len(rules), n_features+1))

    def forward(self, x, return_details=False):
        sigma = torch.exp(self.log_sigmas).clamp_min(1e-3)
        mu = torch.exp(-0.5*((x.unsqueeze(-1)-self.centers)/sigma)**2)

        firing = torch.ones(x.size(0), len(self.rules), device=x.device)
        for i in range(self.n_features):
            firing *= mu[:, i, self.rules[:, i]]
        normalized = firing/(firing.sum(dim=1, keepdim=True)+1e-8)

        x_aug = torch.cat([torch.ones(x.size(0),1,device=x.device), x], dim=1)
        rule_output = x_aug @ self.consequents.T
        logit = (normalized*rule_output).sum(dim=1)
        probability = torch.sigmoid(logit)

        if return_details:
            return probability, {
                "membership": mu,
                "firing": firing,
                "normalized_firing": normalized,
                "rule_output": rule_output,
                "logit": logit
            }
        return probability

    def rule_text(self, feature_names):
        names=["LOW","HIGH"]
        return [
            (idx, " AND ".join(f"{feature_names[i]} IS {names[m]}" for i,m in enumerate(rule)))
            for idx,rule in enumerate(self.rules.tolist())
        ]
