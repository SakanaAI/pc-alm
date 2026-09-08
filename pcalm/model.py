from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp

Params = list[jax.Array]


def activation_fn(name: str) -> Callable[[jax.Array], jax.Array]:
    if name == "linear":
        return lambda x: x
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"unknown activation: {name}")


def model_scales(width: int, depth: int, input_dim: int) -> list[float]:
    if depth < 2:
        raise ValueError("depth must include at least one hidden layer and one output layer")
    return [1.0 / math.sqrt(input_dim)] + [1.0 / math.sqrt(width * depth)] * (depth - 2) + [1.0 / width]


def skip_mask(depth: int) -> tuple[bool, ...]:
    return tuple([False] + [True] * (depth - 2) + [False])


def init_params(
    key: jax.Array,
    *,
    depth: int,
    width: int,
    input_dim: int,
    output_dim: int,
    dtype=jnp.float32,
) -> Params:
    keys = jax.random.split(key, depth)
    layers: Params = []
    for layer_ix in range(depth):
        in_dim = input_dim if layer_ix == 0 else width
        out_dim = output_dim if layer_ix == depth - 1 else width
        layers.append(jax.random.normal(keys[layer_ix], (out_dim, in_dim), dtype=dtype))
    return layers


def block_pred(
    W: jax.Array,
    scale: float,
    skip: bool,
    z_prev: jax.Array,
    phi: Callable[[jax.Array], jax.Array],
    *,
    is_first: bool,
) -> jax.Array:
    inp = z_prev if is_first else phi(z_prev)
    pred = scale * (inp @ W.T)
    if skip:
        pred = pred + z_prev
    return pred


def forward(params: Params, scales: list[float], skips: tuple[bool, ...], x: jax.Array, phi) -> list[jax.Array]:
    acts = []
    z_prev = x
    for layer_ix, W in enumerate(params):
        z = block_pred(W, scales[layer_ix], skips[layer_ix], z_prev, phi, is_first=(layer_ix == 0))
        acts.append(z)
        z_prev = z
    return acts


def logits(params: Params, scales: list[float], skips: tuple[bool, ...], x: jax.Array, phi) -> jax.Array:
    return forward(params, scales, skips, x, phi)[-1]
