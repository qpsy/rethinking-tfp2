# Statistical Rethinking (2nd Edition) with Tensorflow Probability on JAX

This repository replicates Kapil Sachdeva's excellent [implementation] 
(https://github.com/ksachdeva/rethinking-tensorflow-probability) using
jax substrate of Tensorflow Probability and 
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

model_x = tfdx.JointDistributionCoroutineAutoBatched(_generator_x)
model_x.sample_distributions(value=post)
```
For details, please see Chapter 7 and 8.


- tfd.Poission treats observations (through experimental_pin) as batch in tfd.JointDistribution[...]AutoBatched

A remedy: use tfd.Sample
```python
def model():
  def _generator():
    ...
    yield tfd.Sample(tfd.Poisson(rate=rate), sample_shape=sample_shape)
  return tfd.JointDistributionCoroutineAutoBatched(_generator)
```
See Ch11, C12 for details.


- confusing parameters in tfd.NegativeBinomial 

The [official document] 
(https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/NegativeBinomial) says,

Given a Bernoulli trial with probability p of success, the NegativeBinomial distribution represents the distribution over the number of successes s that occur until we observe f failures, where total_count = f, probs = p.

However, results show that 'total_count = s, probs = 1-p' are the correct parameters. e.g.,
```python
tfd.NegativeBinomial(total_count=1., probs=.3).prob([1., 2., 3.])
# == dnbinom(c(1, 2, 3), size=1, prob=.7)
```