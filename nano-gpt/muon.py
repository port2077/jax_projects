import jax
from functools import partial
import jax.numpy as jnp

def newton_schulz(G: jax.Array, steps: int) -> jax.Array :

    assert jnp.ndim(G) == 2, 'G should be 2-dim matrix'
    a,b = (3/2,1/2)

    X = G.astype('bfloat16')
    X = X / (jnp.linalg.norm(X,keepdims=True) + 1e-7)

    for _ in range(steps):
        A = jnp.matmul(X,X.T)
        X = a * X - b * jnp.matmul(A,X)

    return X

def muon_update(grad: jax.Array, momentum: jax.Array, beta: int=0.95, steps: int=5, nesterov=True):

    momentum = beta * momentum + (1-beta) * grad
    update =  newton_schulz(momentum, steps)
    update *= (grad.shape[-2]/grad.shape[-1]) ** 0.5

    return update 



