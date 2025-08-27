from typing import NamedTuple, Literal
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import transform
from optax._src import combine


class MuonUpdateState(NamedTuple):

    momentum: base.Updates

def scale_by_muon(beta: float=0.95, steps: int=5, nesterov: bool = True,
                  polynomial: Literal['cubic','quintic','cursed_quintic'] = 'cubic') -> base.GradientTransformation:

    def newton_schulz(G: jax.Array, steps: int, 
                      polynomial: Literal['cubic','quintic','cursed_quintic'] = 'cubic') -> jax.Array :

        assert jnp.ndim(G) == 2, 'G should be 2-dim matrix'
        X = G.astype('bfloat16')
        X = X / (jnp.linalg.norm(X,keepdims=True) + 1e-7)

        if polynomial == 'cubic':
            X = cubic_polynomial(X,steps)
        elif polynomial == 'quintic':
            X = quintic_polynomial(X,steps)
        elif polynomial == 'cursed_quintic':
            X = quintic_polynomial(X,steps,cursed=True)

        return X
    
    def cubic_polynomial(X: jax.Array,steps: int)-> jax.Array :
        a,b = (3/2,-1/2)
        def step(x,_):
            A = jnp.matmul(x,x.T)
            x = a * x + b * jnp.matmul(A,x)
            return x, None
        X,_ = jax.lax.scan(step,init=X,length=steps)
        return X
    
    def quintic_polynomial(X: jax.Array,steps: int,cursed: bool = False)-> jax.Array :
        a,b,c = (3.445,-4.7750,2.0315) if cursed else (3,-16/5,6/5)
        def step(x,_):
            A = jnp.matmul(x,x.T)
            B = jnp.matmul(A,A)
            x = a * x + b * jnp.matmul(A,x) + c * jnp.matmul(B,x)
            return x, None
        X,_ = jax.lax.scan(step,init=X,length=steps)
        return X


    def init_fn(params):
        momentum = jax.tree.map(jnp.zeros_like,params)
        return MuonUpdateState(momentum=momentum)

    def update_fn(grad: base.Updates, state: MuonUpdateState, params):
        momentum = jax.tree.map(lambda m,g: beta * m + (1-beta)*g,state.momentum,grad)
        update = jax.tree.map(lambda m,g: beta * m + (1-beta)*g,momentum,grad) if nesterov else momentum 
        update = jax.tree.map(lambda m: newton_schulz(m,steps,polynomial),update)
        update = jax.tree.map(lambda x: x * (x.shape[-2]/x.shape[-1])**0.5, update)

        return update, MuonUpdateState(momentum=momentum)
    
    return base.GradientTransformation(init_fn, update_fn)


def muon(lr: base.ScalarOrSchedule, beta: float=0.95, steps: int=5,
         polynomial: Literal['cubic','quintic','cursed_quintic'] = 'cubic', nesterov: bool= True) -> base.GradientTransformation:
    return combine.chain(
        scale_by_muon(
            beta = beta,
            steps= steps,
            polynomial=polynomial,
            nesterov=nesterov
        ),
        transform.scale_by_learning_rate(lr),
    )
