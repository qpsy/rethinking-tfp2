# Statistical Rethinking (2nd Edition) with Tensorflow Probability on JAX

This repository replicates Kapil Sachdeva's excellent [implementation] 
(https://github.com/ksachdeva/rethinking-tensorflow-probability) of Statistical Rethinking (2nd Edition) using
jax subtrate of Tensorflow Probability and 
'AutoBatched' JointDistribution (especially JointDistributionCoroutineAutoBatched).

## Note
- 'sample_distributions' method not working
```python
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def _generator():
  a = yield tfd.Normal(loc=0.0, scale=1.0, name="a")
  ...

model = tfd.JointDistributionCoroutineAutoBatched(_generator)
post = tfp.mcmc.sample_chain(...)
# ValueError: Attempt to convert a value with an unsupported type (<class 'object'>) to a Tensor.
model.sample_distributions(value=post)
```
A tenatative remedy
```python
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
import tensorflow_probability as tfpx
tfdx = tfpx.distributions

def _generator():
  a = yield tfd.Normal(loc=0.0, scale=1.0, name="a")
  ...

model = tfd.JointDistributionCoroutineAutoBatched(_generator)
post = tfp.mcmc.sample_chain(...)

# the same model using the original tfp
def _generator_x():
  a = yield tfdx.Normal(loc=0.0, scale=1.0, name="a")
  ...

model_x = tfd.JointDistributionCoroutineAutoBatched(_generator_x)
model_x.sample_distributions(value=post)
```
For details, please see Chapter 7 and 8.