import torch as th
import numpy as np
from typing import Callable, Iterator, Any
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.optim import Rprop, Optimizer
from torch.nn import functional as F
from torch.func import grad, vmap, functional_call, stack_module_state
from copy import deepcopy
from dataclasses import dataclass


_DEFAULT_NUMPY_TYPE = np.float32
_DEFAULT_TORCH_TYPE = th.float32


@dataclass
class NBISResult:
    X: np.ndarray
    predict_proba: Callable
    predict_entropy: Callable
    model: nn.Module
    contourf: Callable | None = None
    crange: tuple[float, float] | None = None
    device: Any = None


class MLP(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, width: int | None = None, depth: int = 32):
        super().__init__()
        hid_dims = width if width is not None else 3 * in_dims
        self.enc = nn.Linear(in_features=in_dims, out_features=hid_dims)
        self.hid = nn.ModuleList([nn.Linear(in_features=hid_dims, out_features=hid_dims) for _ in range(depth)])
        self.dec = nn.Linear(in_features=hid_dims, out_features=out_dims)
        self.act = nn.Mish()

    def forward(self, X: th.Tensor):
        Z = self.enc(X)
        for linear in self.hid:
            Z = self.act(linear(Z)) + Z
        Y = self.dec(Z)
        return Y
    

class Ensemble(nn.Module):
    def __init__(self, model_class, num_models: int = 15, device='cpu', dtype=_DEFAULT_TORCH_TYPE, **kwargs):
        super().__init__()

        models = [model_class(**kwargs).to(device=device, dtype=dtype) for _ in range(num_models)]
        self.base = deepcopy(models[0]).to(device='meta')

        params, buffers = stack_module_state(models)
        self._all_params = params
        self._all_buffers = buffers

        self.func = lambda params, buffers, x: functional_call(self.base, (params, buffers), (x,))

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return iter(self._all_params.values())
    
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        return iter(self._all_buffers.values())

    def forward(self, X: th.Tensor):
        return vmap(self.func, in_dims=(0, 0, None))(self._all_params, self._all_buffers, X)
    
    def predict_proba(self, X: th.Tensor):
        return F.softmax(input=self(X), dim=2).mean(dim=0)
    
    def predict_entropy(self, X: th.Tensor):
        p, _ = self.predict_proba(X=X).max(dim=1)
        return -(th.nan_to_num(p * th.log(p)) + th.nan_to_num((1 - p) * th.log(1 - p)))


def wrapper(model: Ensemble, dtype, device, otype: str):
    def wrapping(X: np.ndarray):
        X = np.atleast_2d(X)
        with th.no_grad():
            X = th.from_numpy(X).to(dtype=dtype, device=device)
            if otype == 'probability':
                return model.predict_proba(X=X).cpu().numpy()
            if otype == 'entropy':
                return model.predict_entropy(X=X).cpu().numpy()
    return wrapping
    

def train_model(X: th.Tensor, Y: th.Tensor, optimizer: th.optim.Optimizer, model: Ensemble, epochs: int = 1):
    X = X.detach()
    Y = Y.detach()
    _loss_fn = lambda pred: F.cross_entropy(input=pred, target=Y)
    loss_fn = lambda inputs: (vmap(_loss_fn)(model(inputs))).mean()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(X)
        loss.backward()
        optimizer.step()


def move_points(X: th.Tensor, model: Ensemble, steps: int = 1, step_size: float = 0.001):
    X = X.detach().requires_grad_(True)
    dedx = grad(lambda X: model.predict_entropy(X=X).mean())
    for _ in range(steps):
        X = X + step_size * th.sign(dedx(X))
    return X.detach()


def nbis_step(dynamic: tuple[th.Tensor, th.Tensor],
              optimizer: th.optim.Optimizer,
              model: Ensemble, 
              epochs: int = 1,
              steps: int = 1,
              step_size: float = 0.001,
              frozen: tuple[th.Tensor, th.Tensor] | None = None,
              sampled: tuple[th.Tensor, th.Tensor] | None = None):
    X_train, Y_train = X_dynamic, _ = dynamic
    if frozen is not None:
        X_frozen, Y_frozen = frozen
        X_train = th.concat([X_train, X_frozen], dim=0)
        Y_train = th.concat([Y_train, Y_frozen], dim=0)
    if sampled is not None:
        X_sampled, Y_sampled = sampled
        X_train = th.concat([X_train, X_sampled], dim=0)
        Y_train = th.concat([Y_train, Y_sampled], dim=0)
    train_model(X=X_train, Y=Y_train, optimizer=optimizer, model=model, epochs=epochs)
    return move_points(X=X_dynamic, model=model, steps=steps, step_size=step_size)


def nbis_bounds(X: np.ndarray, dtype=_DEFAULT_TORCH_TYPE, device='cpu'):
    X_min = th.from_numpy(X.min(axis=0)).to(dtype=dtype, device=device)
    X_max = th.from_numpy(X.max(axis=0)).to(dtype=dtype, device=device)
    return X_min, X_max


@th.no_grad()
def probe_aabb(X: th.Tensor, num_samples: int):
    X_min = X.min(dim=0).values
    X_max = X.max(dim=0).values
    aabb_size = X_max - X_min
    return X_min + aabb_size * th.rand(num_samples, X_min.size(0), device=X.device, dtype=X.dtype)


def nbis(X0: np.ndarray, 
         func: Callable, 
         device='cpu', 
         dtype=_DEFAULT_TORCH_TYPE,
         epochs: int = 1, 
         outer_steps: int = 50, 
         inner_steps: int = 1, 
         step_size: float = 0.1,
         depth: int = 7,
         factor: int = 2,
         keep_X0: bool = True,
         bounded: bool = True,
         sample_aabb: bool = True,
         sample_size: int | None = None,
         ensemble: int = 32,
         noise: float | None = 0.06,
         optimizer_class: Optimizer = Rprop,
         **kwargs):
    if bounded:
        X_min, X_max = nbis_bounds(X=X0, dtype=dtype, device=device)
    X_frozen = X_dynamic = th.from_numpy(X0).to(dtype=dtype, device=device)
    Y_frozen = Y_dynamic = th.from_numpy(func(X0)).to(dtype=dtype, device=device)
    model = Ensemble(model_class=MLP, 
                     num_models=ensemble, 
                     in_dims=X_frozen.shape[1], 
                     out_dims=Y_frozen.shape[1], 
                     width=factor * X_frozen.shape[1] + factor, 
                     depth=depth, 
                     device=device, 
                     dtype=dtype)
    sample_size = X_frozen.shape[0] if sample_size is None else sample_size
    opt = optimizer_class(params=model.parameters(), **kwargs)
    for _ in range(outer_steps):
        X_prev = X_dynamic
        sampled = None
        if sample_aabb:
            sampled_X = probe_aabb(X=X_dynamic, num_samples=sample_size)
            sampled_Y = th.from_numpy(func(sampled_X.detach().cpu().numpy()))
            sampled = (sampled_X, sampled_Y).to(dtype=dtype, device=device))
        X_dynamic = nbis_step(dynamic=(X_dynamic, Y_dynamic),
                              optimizer=opt, 
                              model=model, 
                              epochs=epochs, 
                              steps=inner_steps, 
                              step_size=step_size,
                              frozen=(X_frozen, Y_frozen) if keep_X0 else None,
                              sampled=sampled)
        with th.no_grad():
            if noise is not None:
                X_dynamic = X_dynamic + th.randn_like(X_dynamic) * noise
            if bounded:
                inside = th.bitwise_and(input=(X_min < X_dynamic), other=(X_dynamic < X_max))
                X_dynamic = th.where(condition=inside, input=X_dynamic, other=X_prev)
        X_output = X_dynamic.detach().cpu().numpy()
        yield NBISResult(X=X_output, 
                         predict_proba=wrapper(model=model, dtype=dtype, device=device, otype='probability'), 
                         predict_entropy=wrapper(model=model, dtype=dtype, device=device, otype='entropy'), 
                         model=model,
                         device=device)
        Y_dynamic = th.from_numpy(func(X_output)).to(dtype=dtype, device=device)
