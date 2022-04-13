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
        

class A2C_CONTI_MODEL(nn.Module):
    def __init__(self, config, state_size, action_size):
        super().__init__()
        nf = config.model.nf
        n_layers = config.model.n_layers
        self.avg_action = torch.Tensor([0.35, 0.15]).reshape(1, -1)
        self.avg_action.requires_grad_(False)
        
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
        avg_action = self.avg_action.repeat(x.shape[0], 1).to(x)
        return self.mu(out) + avg_action, self.var(out), self.value(out)


class A2C_DISCRETE_MODEL(nn.Module):
    def __init__(self, config, state_size, n_actions):
        super().__init__()
        nf = config.model.nf
        n_layers = config.model.n_layers
        self.input_normalizer = torch.Tensor(
            [1e+6, 3000, 400000, 2000, 1e+6, 10000]
        ).reshape(1, -1)
        self.input_normalizer.requires_grad_(False)
        
        self.main = nn.ModuleList([])

        for i in range(n_layers):
            in_dim = nf if i > 0 else state_size
            out_dim = nf
            self.main.append(LinearBlock(config, in_dim, out_dim))

        self.policy = nn.Linear(nf, n_actions)
        self.value = nn.Linear(nf, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def normalize(self, x: torch.Tensor):
        '''Not used now'''
        out = x / self.input_normalizer.to(x)
        return out
        
    def forward(self, x: torch.Tensor):
        out = self.main[0](torch.log(x + 1e-10), is_residual=False)
        # x = self.normalize(x)
        # out = self.main[0](x, is_residual=False)
        for i in range(1, len(self.main)):
            out = self.main[i](out, is_residual=True)
        return self.policy(out), self.value(out)
