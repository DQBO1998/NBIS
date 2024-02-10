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
from scipy.optimize import milp
from scipy.sparse import dok_array as sparse_array
from scipy.sparse import vstack
from time import time
from itertools import product
from pathos.multiprocessing import Pool


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


def dilate(X: np.ndarray, noise: float = 0.1, num: int = 5):
    X = np.repeat(a=X, repeats=num, axis=0)
    return X + np.random.randn(*X.shape) * noise


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
            sampled_Y = th.from_numpy(func(sampled_X.detach().cpu().numpy())).to(dtype=dtype, device=device)
            sampled = (sampled_X, sampled_Y)
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
            X_output = X_dynamic.cpu().numpy()
            yield NBISResult(X=X_output, 
                            predict_proba=wrapper(model=model, dtype=dtype, device=device, otype='probability'), 
                            predict_entropy=wrapper(model=model, dtype=dtype, device=device, otype='entropy'), 
                            model=model,
                            device=device)
            Y_dynamic = th.from_numpy(func(X_output)).to(dtype=dtype, device=device)


def final(iter):
    for item in iter:
        pass
    return item


def tmarker():
    old = time()
    while True:
        new = time()
        yield new - old
        old = new


def _fit_aabb(x: np.ndarray, supp: np.ndarray, db: np.ndarray, func: Callable, eps: float = 1., verbose: bool = False, max_time: float | None = None, M: float = 1e3):
    if verbose:
        print(f'x: {x}')
        print(f'supp has NaN: {np.any(np.isnan(supp))}')
        print(f'db has NaN: {np.any(np.isnan(db))}')
    
    ref = func(np.atleast_2d(x))[0]
    X_in = supp[func(supp) == ref]
    X_out = np.concatenate([db[func(db) != ref], supp[func(supp) != ref]], axis=0)
    
    N_END = x.shape[0]
    P_END = N_END + x.shape[0]
    Y_END = P_END + X_in.shape[0]
    Z_END = Y_END + X_out.shape[0] * x.shape[0]
    H_END = Z_END + X_out.shape[0] * x.shape[0]

    n = lambda i: i
    p = lambda i: N_END + i
    y = lambda i: P_END + i
    Z = lambda j, i: Y_END + x.shape[0] * j + i
    H = lambda j, i: Z_END + x.shape[0] * j + i

    tm = tmarker(); next(tm)

    _A1 = sparse_array((x.shape[0], H_END), dtype=float); _ub1 = np.zeros((_A1.shape[0],), dtype=float)
    _A2 = sparse_array((x.shape[0], H_END), dtype=float); _ub2 = np.zeros((_A2.shape[0],), dtype=float)
    for i in range(x.shape[0]):
        _A1[i, n(i)] = 1.; _ub1[i] = x[i]
        _A2[i, p(i)] = -1.; _ub2[i] = -x[i]

    if verbose:
        print(f'_A1 and _A2: {next(tm)} sec')

    _A3 = sparse_array((X_in.shape[0] * x.shape[0], H_END), dtype=float); _ub3 = np.zeros((_A3.shape[0],), dtype=float)
    _A4 = sparse_array((X_in.shape[0] * x.shape[0], H_END), dtype=float); _ub4 = np.zeros((_A4.shape[0],), dtype=float)
    for k, (j, i) in enumerate(product(range(X_in.shape[0]), range(x.shape[0]))):
        _A3[k, [n(i), y(j)]] = 1., -M; _ub3[k] = X_in[j, i]
        _A4[k, [p(i), y(j)]] = -1., -M; _ub4[k] = -X_in[j, i]

    if verbose:
        print(f'_A3 and _A4: {next(tm)} sec')
    
    _A5 = sparse_array((X_out.shape[0] * x.shape[0], H_END), dtype=float); _ub5 = np.zeros((_A5.shape[0],), dtype=float)
    _A6 = sparse_array((X_out.shape[0] * x.shape[0], H_END), dtype=float); _ub6 = np.zeros((_A6.shape[0],), dtype=float)
    for k, (j, i) in enumerate(product(range(X_out.shape[0]), range(x.shape[0]))):
        _A5[k, [n(i), Z(j, i)]] = -1., -M; _ub5[k] = -X_out[j, i]
        _A6[k, [p(i), H(j, i)]] = 1., -M; _ub6[k] = X_out[j, i]

    if verbose:
        print(f'_A5 and _A6: {next(tm)} sec')
    
    _A7 = sparse_array((X_out.shape[0], H_END), dtype=float); _ub7 = np.zeros((_A7.shape[0],), dtype=float)
    for j in range(X_out.shape[0]):
        for i in range(x.shape[0]):
            _A7[j, [Z(j, i), H(j, i)]] = 1., 1.
        _ub7[j] = 2 * x.shape[0] - 1

    if verbose:
        print(f'_A7: {next(tm)} sec')
    
    A = vstack((_A1, _A2, _A3, _A4, _A5, _A6, _A7))
    ub = np.concatenate((_ub1, _ub2, _ub3, _ub4, _ub5, _ub6, _ub7), axis=0)
    
    min_bnd = np.zeros((H_END,)); max_bnd = np.ones((H_END,))
    min_bnd[:P_END] = -np.inf; max_bnd[:P_END] = np.inf

    ints = np.ones((H_END), dtype=int)
    ints[:P_END] = 0

    coeff = np.zeros((H_END,))
    coeff[P_END:Y_END] = 1.

    if verbose:
        print(f'solving...', end=''); next(tm)

    res = milp(c=coeff, integrality=ints, bounds=(min_bnd, max_bnd), constraints=(A, -np.inf, ub), options=None if max_time is None else {'time_limit': max_time})

    if verbose:
        print(f'solved in {next(tm)} sec, with status {res.status}\n')

    if res.status in (0, 3):
        neg = res.x[:N_END]
        pos = res.x[N_END:P_END]
        return True, neg, pos
    return False, None, None


def fit_aabb(x: np.ndarray, 
             supp: np.ndarray, 
             db: np.ndarray, 
             func: Callable, 
             eps: float = 1., 
             verbose: bool = False, 
             M = 1e3,
             heuristic: bool = False, 
             c_size: int = 100,
             s_size: int | None = None,
             n_jobs: int | None = 2,
             max_time: float | None = None):
    if heuristic:
        s_size = (max(supp.shape[0], db.shape[0]) // c_size) if s_size is None else s_size
        with Pool(processes=n_jobs) as pool:
            res_lt = pool.map(func=lambda supp_db: _fit_aabb(x=x, supp=supp_db[0], db=supp_db[1], func=func, eps=eps, verbose=False, M=M), 
                              iterable=((supp[np.random.randint(low=0, high=supp.shape[0], size=c_size, dtype=int)], 
                                         db[np.random.randint(low=0, high=db.shape[0], size=c_size, dtype=int)]) for _ in range(s_size)))
            res_lt = [res for res in res_lt if res[0]]
            if res_lt:
                _, neg_lt, pos_lt = zip(*res_lt)
                neg = np.mean(neg_lt, axis=0); pos = np.mean(pos_lt, axis=0)
                return True, neg, pos
            return False, None, None
    return _fit_aabb(x=x, supp=supp, db=db, func=func, eps=eps, verbose=verbose, max_time=max_time, M=M)
