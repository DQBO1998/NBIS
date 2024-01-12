import torch as th
import numpy as np
from typing import Callable, Iterator, Any
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.optim import Rprop, Optimizer
from torch.nn import functional as F
from torch.func import grad, vmap, functional_call, stack_module_state
from copy import deepcopy
from sklearn.decomposition import PCA
from dataclasses import dataclass, replace
from fast_tsp import find_tour


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


def wrapper(model: Ensemble, device, otype: str):
    def wrapping(X: np.ndarray):
        X = np.atleast_2d(X)
        with th.no_grad():
            X = th.from_numpy(X).to(dtype=_DEFAULT_TORCH_TYPE, device=device)
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
    X = X0
    if bounded:
        X_min, X_max = nbis_bounds(X=X0, dtype=_DEFAULT_TORCH_TYPE, device=device)
    X_frozen = X_dynamic = th.from_numpy(X).to(dtype=_DEFAULT_TORCH_TYPE, device=device)
    Y_frozen = Y_dynamic = th.from_numpy(func(X)).to(dtype=_DEFAULT_TORCH_TYPE, device=device)
    model = Ensemble(model_class=MLP, 
                     num_models=ensemble, 
                     in_dims=X.shape[1], 
                     out_dims=Y_frozen.shape[1], 
                     width=factor * X.shape[1] + factor, 
                     depth=depth, 
                     device=device, 
                     dtype=_DEFAULT_TORCH_TYPE)
    sample_size = X0.shape[0] if sample_size is None else sample_size
    opt = optimizer_class(params=model.parameters(), **kwargs)
    for _ in range(outer_steps):
        X_prev = X_dynamic
        sampled = None
        if sample_aabb:
            sampled_X = probe_aabb(X=X_dynamic, num_samples=sample_size)
            sampled = (sampled_X, th.from_numpy(func(sampled_X.detach().cpu().numpy())).to(dtype=Y_dynamic.dtype, device=device))
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
                X_dynamic = th.where(condition=th.bitwise_and(input=(X_min < X_dynamic), other=(X_dynamic < X_max)), 
                                    input=X_dynamic, 
                                    other=X_prev)
        X_output = X_dynamic.detach().cpu().numpy()
        yield NBISResult(X=X_output, 
                         predict_proba=wrapper(model=model, device=device, otype='probability'), 
                         predict_entropy=wrapper(model=model, device=device, otype='entropy'), 
                         model=model,
                         device=device)
        Y_dynamic = th.from_numpy(func(X_output)).to(dtype=Y_frozen.dtype, device=device)
    

def sample_line(a: th.Tensor, b: th.Tensor, n: int):
    steps = th.linspace(0, 1, n).unsqueeze(1)
    return a + steps * (b - a)


def find_cycle(X: th.Tensor,
               distance_func: Callable,
               entropy_func: Callable,
               seconds: float):
    distances = vmap(lambda a: vmap(lambda b: distance_func(a, b))(X))(X)
    entropies = vmap(lambda a: vmap(lambda b: entropy_func(a, b))(X))(X)
    costs = ((distances / entropies).cpu().numpy() * 100.).astype(int)
    return th.from_numpy(np.array(find_tour(dists=costs, duration_seconds=seconds))).to(device=X.device)


def make_contourf(X: th.Tensor, distance_func: Callable):
    X = th.concat([X, X[0].reshape(1, -1)], dim=0)
    windows = X.unfold(dimension=0, size=2, step=1)
    arr, brr = windows[:, 0], windows[:, 1]
    leaps = vmap(lambda a, b: distance_func(a, b))(arr, brr)
    domain = th.concat([th.zeros((1,), dtype=leaps.dtype, device=leaps.device), th.cumsum(input=leaps, dim=0)], dim=0)
    @th.no_grad()
    def _contourf(input: th.Tensor):
        xi, i = domain.masked_fill(mask=(input < domain), value=-th.inf).max(dim=0)
        xj, j = domain.masked_fill(mask=(domain <= input), value=th.inf).min(dim=0)
        yi = X.index_select(dim=0, index=i)[0]
        yj = X.index_select(dim=0, index=j)[0]
        w = (input - xi) / (xj - xi)
        return yi + w * (yj - yi)
    def contourf(inputs: np.ndarray):
        inputs = th.from_numpy(inputs).to(dtype=domain.dtype, device=domain.device)
        outputs = vmap(_contourf)(inputs)
        return outputs.cpu().numpy()
    return contourf, domain.min().item(), domain.max().item()


@th.no_grad()
def refine(nbis_res: NBISResult, db_size: int | None = None, sample_res: int = 100, maxsecs: float = 60.):
    db_size = nbis_res.X.shape[0] if db_size is None else db_size
    X = th.from_numpy(nbis_res.X).to(device=nbis_res.device)
    distance_func = lambda a, b: th.sqrt(th.sum((a - b) ** 2))
    entropy_func = lambda a, b: nbis_res.model.predict_entropy(X=sample_line(a=a, b=b, n=sample_res)).mean()
    indices = find_cycle(X=X, distance_func=distance_func, entropy_func=entropy_func, seconds=maxsecs)
    contourf, start, end = make_contourf(X=X[indices], distance_func=distance_func)
    ref_X = contourf(np.linspace(start=start, stop=end, num=db_size))
    return replace(deepcopy(nbis_res), contourf=contourf, crange=(start, end), X=ref_X)
    
