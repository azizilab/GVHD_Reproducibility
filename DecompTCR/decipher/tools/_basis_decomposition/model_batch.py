import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
import torch.nn.functional as F



class RealFunction(pyro.nn.PyroModule):
    def __init__(self, *hidden_dims, inverse_length_scale, sigmoid_output=False):
        super(RealFunction, self).__init__()
        dimensions = [1, *hidden_dims, 1]
        self.sigmoid_output = sigmoid_output
        layers = []
        for in_features, out_features in zip(dimensions, dimensions[1:]):
            layer = pyro.nn.PyroModule[torch.nn.Linear](in_features, out_features)
            layer.weight = pyro.nn.PyroSample(
                dist.Normal(0.0, inverse_length_scale / in_features ** 0.5)
                .expand(torch.Size([out_features, in_features]))
                .to_event(2)
            )
            layer.bias = pyro.nn.PyroSample(
                dist.Normal(0.0, 1.0 / in_features ** 0.5)
                .expand(torch.Size([out_features]))
                .to_event(1)
            )
            layers.append(layer)

        self.layers = layers
        for i in range(len(layers)):
            setattr(self, f"layer_{i}", layers[i])

        self.activation = torch.nn.Tanh() #Maybe change the activation function relu or sigmoid

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")

        # Reshape x if it's a tensor
        if len(x.size()) == 1:
            x = x.view((-1, 1))

        # Iterate through the layers
        for i in range(len(self.layers) - 1):
            layer = getattr(self, f"layer_{i}")
            x = layer(x)
            x = self.activation(x)

        # Apply the final layer
        layer = getattr(self, f"layer_{len(self.layers) - 1}")
        x = layer(x)

        # Optionally apply sigmoid activation
        if self.sigmoid_output:
            x = torch.sigmoid(x)

        return x
    
class BasisDecomposition(torch.nn.Module):
    def __init__(self, K, n_clones, batch_size, beta_prior=1.0, inverse_length_scale=1.0,  normalized_mode=True):
        super().__init__()
        self.K = K
        self.normalized_mode = normalized_mode

        self.trajectory_basis = pyro.nn.PyroModule[torch.nn.ModuleList](
            [RealFunction(32, 32, inverse_length_scale = inverse_length_scale, sigmoid_output=self.normalized_mode) for _ in range(self.K)]
        )
        self.K = K
        self.beta_prior = beta_prior
        self.n_clones = n_clones
        self.batch_size = batch_size

        self.return_basis = True
        self._last_basis = None
        self._last_patterns = None

    @property
    def gene_scales(self):
        return torch.exp(pyro.param("gene_scale", torch.zeros(self.n_clones,1)))#, constraint = pyro.distributions.constraints.interval(1e-5,1000.0)))
    
    @property
    def std(self):
        return torch.exp(pyro.param("std", torch.zeros(self.n_clones,1)))
    

    def forward(self, times, data):#, batch_idx):
        basis = self.get_basis(times)#, self.time_offset)

        if not self.normalized_mode:
            betas = pyro.sample(
                "beta",
                dist.Exponential(self.beta_prior)
                .expand([self.n_clones, self.K])
                .to_event(2),
            )
            std = 0.1
        else:
            
            betas = pyro.sample(
                "beta",
                dist.Dirichlet(torch.ones(self.K) * self.beta_prior)
                .expand(torch.Size([self.n_clones])).to_event(1),
            )
            gene_scales = self.gene_scales
            if torch.sum(torch.isnan(gene_scales)) != 0:
                print(gene_scales)
                
                    
            betas = betas * gene_scales
            if torch.sum(torch.isnan(betas)) != 0:
                for s in betas.sum(dim = -1):
                    print(s)
                
            std = self.std * gene_scales

        # print("Gene_scales size = {}".format(self.gene_scales.shape))
        # print("betas size = {}".format(betas.shape))
        gene_patterns = torch.einsum("gk, tk -> gt", betas, basis)
        gene_batch = gene_patterns#[batch_idx,:]
        std_batch = std#[batch_idx]
        
        if torch.min(gene_batch) < 0:
            print("gene_batch min = {}", torch.min(gene_batch))
            print("gene_batch max = {}", torch.max(gene_batch))

        with  pyro.plate("t", len(times), dim = -1):
            with pyro.plate("g", size = data.shape[0], subsample_size = self.batch_size, dim = -2) as ind:
                pyro.sample("obs", dist.NanMaskedNormal(gene_batch.index_select(0,ind), std_batch.index_select(0,ind)), obs=data.index_select(0, ind))
    
        self._last_basis = basis
        self._last_patterns = gene_patterns

        if self.return_basis:
            return basis
        else:
            return gene_patterns

        
    def get_basis(self, times):
        # noinspection PyTypeChecker
        basis = [trajectory(times) for trajectory in self.trajectory_basis]
        basis = [b / (b.max() + 1e-6) for b in basis]
        basis = torch.cat(basis, dim=1)
        return basis

    def show_basis(self, times):
        # noinspection PyTypeChecker
        basis = np.array(
            [trajectory(times).detach().numpy().reshape(-1) for trajectory in self.trajectory_basis]
        ).T

        data = pd.DataFrame(basis)
        data.plot()
        
def handle_na_zeros(betas):
        mask = (torch.isnan(betas)+ betas <= 0)
        
        if torch.sum(mask) == 0:
            return betas
        
        beta_sum = torch.sum(betas[~mask])
        n_beta = torch.sum(~mask)
        n_zero = torch.sum(mask)
        
        sub_val = (n_zero * 1e-8)/n_beta
        betas = betas[~mask] - sub_val
        betas = betas[mask] + 1e-8
        
        return betas  
        
class BasisDecomposition_Evaluate(torch.nn.Module):
    def __init__(self, trajectories, n_clones, beta_prior, normalized_mode=True):
        super().__init__()
        
        self.normalized_mode = normalized_mode

        self.trajectory_basis = trajectories
        self.n_clones = n_clones
        self.beta_prior = beta_prior
        self.return_basis = True
        self._last_basis = None
        self._last_patterns = None

    @property
    def gene_scales(self):
        return torch.exp(pyro.param("gene_scale_val", torch.zeros(self.n_clones,1)))

    @property
    def std(self):
        return torch.exp(pyro.param("std_val", torch.zeros(self.n_clones,1)))
    

    def forward(self, times, data):
        basis = self.trajectory_basis

        if not self.normalized_mode:
            betas = pyro.sample(
                "beta_val",
                dist.Exponential(self.beta_prior)
                .expand([data.shape[0], self.trajectory_basis.shape[1]])
                .to_event(2),
            )
            std = 0.1
        else:
            betas = pyro.sample(
                "beta_val",
                dist.Dirichlet(torch.ones(self.trajectory_basis.shape[1]) * self.beta_prior)
                .expand(torch.Size([data.shape[0]])).to_event(1),
            )
            # print("Gene_scales size = {}".format(self.gene_scales.shape))
            # print("betas size = {}".format(betas.shape))
            gene_scales = self.gene_scales
            betas = betas * gene_scales
            std = self.std * gene_scales


        gene_patterns = torch.einsum("gk, tk -> gt", betas, basis)


        t_axis = pyro.plate("t_val", len(times), dim=-1)
        gene_axis = pyro.plate("g_val", data.shape[0], dim=-2)

        with  gene_axis, t_axis:
            pyro.sample("obs_val", dist.NanMaskedNormal(gene_patterns, std), obs=data)
        
    
        self._last_basis = basis
        self._last_patterns = gene_patterns

        if self.return_basis:
            return basis
        else:
            return gene_patterns
    