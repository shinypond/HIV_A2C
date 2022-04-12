import torch
import torch.nn as nn

from .utils import get_norm, get_act


class LinearBlock(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super().__init__()
        normalization = get_norm(config.model.normalization)
        activation = get_act(config.model.activation)
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            normalization(out_dim),
            activation,
            nn.Linear(out_dim, out_dim),
            normalization(out_dim),
            activation,
        )

    def forward(self, x: torch.Tensor, is_residual: bool = False):
        if is_residual:
            return self.main(x) + x
        else:
            return self.main(x)
        

class A2C_MODEL(nn.Module):
    def __init__(self, config, state_size, action_size):
        super().__init__()
        nf = config.model.nf
        n_layers = config.model.n_layers
        
        self.main = nn.ModuleList([])

        for i in range(n_layers):
            in_dim = nf if i > 0 else state_size
            out_dim = nf
            self.main.append(LinearBlock(config, in_dim, out_dim))

        self.mu = nn.Sequential(
            nn.Linear(nf, action_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(nf, action_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(nf, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor):
        out = self.main[0](torch.log(x + 1e-10), is_residual=False)
        for i in range(1, len(self.main)):
            out = self.main[i](out, is_residual=True)
        return self.mu(out), self.var(out), self.value(out)
