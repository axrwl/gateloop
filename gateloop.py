import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
from flax.linen.initializers import normal

@jax.jit
def gate_loop_operator(k, v, q, a):
    def binary_operator(e_i, e_j):
        a_i, kv_i = e_i
        a_j, kv_j = e_j
        return a_j * a_i, a_j * kv_i + kv_j
    kv = k * v + 0.j
    _, y = lax.associative_scan(binary_operator, (a, kv), axis = 1)
    return q * jnp.real(y)

class GateLoop(nn.Module):
    dim: int

    def setup(self):
        self.norm = nn.RMSNorm()
        self.wq = self.param('wq', normal(), (self.dim, self.dim))
        self.wk = self.param('wk', normal(), (self.dim, self.dim))
        self.wv = self.param('wv', normal(), (self.dim, self.dim))
        self.wa = self.param('wa', normal(), (self.dim, self.dim * 2))
        self.wg = self.param('wg', normal(), (self.dim, self.dim))
        self.wo = self.param('wo', normal(), (self.dim, self.dim))

    def __call__(self, x):
        x = self.norm(x)
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        a = x @ self.wa
        g = x @ self.wg

        a_r, a_i = jnp.split(a, 2, axis = -1)
        a_c = lax.complex(a_r, a_i)
        magnitude, phase = jnp.abs(a_c), jnp.angle(a_c)
        magnitude = nn.sigmoid(magnitude)
        a_c = magnitude * jnp.exp(1j * phase)
        y = gate_loop_operator(k, v, q, a_c)
        y = y * nn.silu(g)
        o = y @ self.wo
        return o
    
class FeedForward(nn.Module):
    dim: int
    mult: int = 4

    def setup(self):
        self.norm = nn.RMSNorm()
        self.proj_in = nn.Dense(self.dim * self.mult)
        self.proj_out = nn.Dense(self.dim)

    def __call__(self, x):
        x = self.norm(x)
        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.proj_out(x)
        return x
    
class GateLoopTransformer(nn.Module):
    num_tokens: int
    dim: int
    depth: int
    ff_mult: int = 4

    def setup(self):
        self.embedding = self.param('embedding', normal(), (self.num_tokens, self.dim))
        layers = []
        for _ in range(self.depth):
            layers.append((GateLoop(self.dim), FeedForward(self.dim, self.ff_mult)))
        self.layers = layers
        self.norm = nn.RMSNorm()

    def __call__(self, x):
        x = self.embedding[x]
        for gateloop, ff in self.layers:
            x = gateloop(x) + x
            x = ff(x) + x
        x = self.norm(x)
        logits = x @ self.embedding.transpose()
        return logits