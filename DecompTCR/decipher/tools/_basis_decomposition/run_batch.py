import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.infer
import pyro.optim
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO, TraceTMC_ELBO
from tqdm import tqdm

from decipher.tools._basis_decomposition.inference import get_inference_guide
from decipher.tools._basis_decomposition.model_batch import BasisDecomposition, BasisDecomposition_Evaluate
from decipher.tools.utils import EarlyStopping



def compute_basis_decomposition(
    clone_patterns,
    inference_mode,
    n_basis=5,
    lr=1e-3,
    n_iter=10_000,
    beta_prior=1.0,
    inverse_length_scale = 1.0,
    weight_decay = .001,
    batch_size = 64,
    seed=1,
    normalized_mode=True,
    times=None,
    plot_every_k_epochs=-1,
    clone_patterns_val = None,
):
    pyro.set_rng_seed(seed)
    clone_patterns = torch.FloatTensor(clone_patterns)
    orig_clone_size = clone_patterns.shape[0]
    
    if clone_patterns_val is not None:
        clone_patterns_val = torch.FloatTensor(clone_patterns_val)
        
    
    # add_nan = batch_size - clone_patterns.shape[0]% batch_size
    # if add_nan == batch_size:
    #     add_nan = 0
        
    # nan_rows = torch.full((add_nan, clone_patterns.shape[1]), float('nan'))
    # clone_patterns = torch.cat((clone_patterns, nan_rows), dim=0)
        
    clone_patterns_mean = clone_patterns.nanmean(axis=(-1), keepdim=True)
    clone_patterns_raw = clone_patterns
    clone_patterns = clone_patterns_raw / clone_patterns_mean

  
    model = BasisDecomposition(
        n_basis,
        beta_prior=beta_prior,
        n_clones = clone_patterns.shape[0],
        batch_size = batch_size,
        inverse_length_scale=inverse_length_scale,
        normalized_mode=normalized_mode,
    )
    
    gamma = 0.1  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / n_iter)
    
    # model = pyro.poutine.scale(model, scale = 0.001) #scale the loss function
    guide = get_inference_guide(model, inference_mode)
    adam = pyro.optim.ClippedAdam({"lr": lr, "weight_decay": weight_decay, "lrd": lrd})#Adam({"lr": lr})
    adam_val = pyro.optim.ClippedAdam({"lr": .25})
    # pyro_scheduler = pyro.optim.CosineAnnealingLR({'optimizer': adam, "Tmax": 20, 'optim_args': {'lr': lr, "weight_decay":weight_decay, "lrd": lrd}})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, adam, loss=elbo)
    

    

    pyro.clear_param_store()
    num_iterations = n_iter
    if times is None:
        times = torch.FloatTensor(np.linspace(-10, 10, clone_patterns.shape[-1]))
        times_val = torch.FloatTensor(np.linspace(-10, 10, clone_patterns.shape[-1]))
    else:
        # TODO: ensure that the times are in [-5, 5]-ish, transparently to the user
        times = torch.FloatTensor(times)
        times_val = torch.FloatTensor(times)

    
    if clone_patterns_val is not None:
        clone_patterns_mean_val = clone_patterns_val.nanmean(axis=(1), keepdim=True)
        clone_patterns_raw_val = clone_patterns_val
        clone_patterns_val = clone_patterns_raw_val / clone_patterns_mean_val

    
    
    losses = []
    rel_loss = []
    val_loss = []
    
    reconstruction_val = 0
    loss = 0
    reconstruction = 0.0
    early_stopping = EarlyStopping(patience=1000)
    pbar = tqdm(range(num_iterations))
    for epoch in pbar:
        # indices = torch.randperm(clone_patterns.shape[0])
        # for i in range(int(clone_patterns.shape[0]/batch_size)):
            # calculate the loss and take a gradient step
            # batch_idx = indices[i*batch_size:(i+1)*batch_size]
            # clone_batch = clone_patterns[batch_idx,:]
            
            
            # loss = svi.step(times, clone_batch, batch_idx)
            # reconstruction = ((model._last_patterns[batch_idx] - clone_batch) ** 2).nanmean().item()
        # with pyro.poutine.scale(scale = 1/clone_patterns.shape[0]):    
        loss = svi.step(times, clone_patterns)# + custom_loss(model._last_basis.detach())#, None)
        reconstruction = ((model._last_patterns - clone_patterns) ** 2).nanmean().item()
        
        pbar.set_description(
        #"Iter: %d Batch_size = (%d, %d) Batch_end_idx = %d  Loss: %.1f - Relative Error: %.2f, - Val_loss: %.2f" % (i, clone_batch.shape[0], clone_batch.shape[1],(i+1)*batch_size, loss, reconstruction, loss_val)
        "Loss: %.1f - Relative Error: %.2f, - Val_loss: %.2f" % (loss, reconstruction, reconstruction_val)
        )
    

        
        # batch_idx += batch_size
            
        model_val = BasisDecomposition_Evaluate(
            model._last_basis.detach(),
            clone_patterns_val.shape[0],
            beta_prior=beta_prior,
            normalized_mode = True
            
        )
        guide_val = get_inference_guide(model_val, inference_mode)
        svi_val = SVI(model_val, guide_val, adam_val, loss=elbo)
            
        for _ in range(30):
            loss_val = svi_val.step(times, clone_patterns_val)
        reconstruction_val = ((model_val._last_patterns - clone_patterns_val) ** 2).nanmean().item()
        val_loss.append(reconstruction_val)
            
        clear_val_params()
        
        pbar.set_description(
            "Loss: %.1f - Relative Error: %.2f, - Val_loss: %.2f" % (loss, reconstruction, reconstruction_val)
        )
        losses.append(loss)
        rel_loss.append(reconstruction)
            

        if plot_every_k_epochs > 0 and epoch % plot_every_k_epochs == 0:
            from IPython.core import display

            basis = model._last_basis.detach().numpy()
            plt.figure(figsize=(5, 2.5))
            _plot_basis(basis)
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.close()
        
        

    # samples = []
    # clone_scales = []
    model.return_basis = False
    # indices = np.arange(clone_patterns.shape[0])
    # for i in range(int(clone_patterns.shape[0]/batch_size)):
    #     predictive = Predictive(
    #         model, guide=guide, num_samples=10, return_sites=("beta", "_RETURN", "obs")
    #     )
    #     batch_idx = indices[i*batch_size:(i+1)*batch_size]
    #     clone_batch = clone_patterns[batch_idx,:]
    #     batch_samples = predictive(times, clone_batch, batch_idx)
    #     print(batch_samples["_RETURN"].shape)
    #     batch_samples["_RETURN"][:,batch_idx] *= clone_patterns_mean[batch_idx]
    #     batch_clone_scales = model.gene_scales[batch_idx] * clone_patterns_mean[batch_idx].squeeze(-1)
    #     batch_samples = summary(batch_samples)
    #     clone_scales.append(batch_clone_scales)
    #     samples.append(batch_samples)
    
    predictive = Predictive(
        model, guide=guide, num_samples=100, return_sites=("beta", "_RETURN", "obs")
    )
    samples = predictive(times, clone_patterns)
    samples["_RETURN"] *= clone_patterns_mean
    # print(clone_patterns_mean.shape)
    # print(model.gene_scales.detach().shape)
    # print((model.gene_scales * clone_patterns_mean).shape)
    clone_scales = model.gene_scales * clone_patterns_mean
    samples = summary(samples)

    
    # clone_scales = torch.vstack(clone_scales)[:,:orig_clone_size]
    # samples = combine_samples(samples)
    # for key in samples.keys():
    #     samples[key] = samples[key][:,:orig_clone_size]

    return model, guide, times, samples, clone_scales, losses, rel_loss, val_loss

def combine_samples(samples):
    res = {}
    for sample in samples:
        for key, tensor in sample.items():
            if key not in res:
                res[key] = []
            res[key].append(tensor)
    
    for key, tensors in res.items():
        res[key] = torch.stack(tensors, dim=1)
    
    return res

def clear_val_params():
    keys_to_delete = keys_to_delete = [key for key in pyro.get_param_store().keys() if "val" in key]
    for key in keys_to_delete:
        del pyro.get_param_store()[key]



def get_basis(model, guide, clone_patterns, times):
    clone_patterns = torch.FloatTensor(clone_patterns)
    times = torch.FloatTensor(times)
    return_basis_value = model.return_basis
    model.return_basis = True
    predictive = Predictive(
        model, guide=guide, num_samples=10, return_sites=("beta", "_RETURN", "obs")
    )
    samples = predictive(times, clone_patterns)
    samples = summary(samples)
    bases = samples["_RETURN"]["mean"].detach().numpy()
    model.return_basis = return_basis_value
    return bases


def _plot_basis(bases, colors=None):
    for i in range(bases.shape[1]):
        plt.plot(
            bases[:, i],
            c=colors[i] if colors is not None else None,
            label="basis %d" % (i + 1),
            linewidth=3,
        )


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "values": v,
            "quantile05": torch.quantile(v, 0.05, dim = 0),
            "quantile95": torch.quantile(v, 0.95, dim = 0)
        }
    return site_stats
