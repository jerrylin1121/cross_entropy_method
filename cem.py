import numpy as np
import random

class CEM():
  def __init__(self, func, d, maxits=500, N=100, Ne=10, argmin=True, v_min=None, v_max=None, init_scale=1, sampleMethod='Gaussian'):
    self.func = func                  # target function
    self.d = d                        # dimension of function input X
    self.maxits = maxits              # maximum iteration
    self.N = N                        # sample N examples each iteration
    self.Ne = Ne                      # using better Ne examples to update mu and sigma
    self.reverse = not argmin         # try to maximum or minimum the target function
    self.v_min = v_min                # the value minimum
    self.v_max = v_max                # the value maximum
    self.init_coef = init_scale       # sigma initial value

    assert sampleMethod=='Gaussian' or sampleMethod=='Uniform'
    self.sampleMethod = sampleMethod  # which sample method gaussian or uniform, default to gaussian

  def eval(self, instr):
    """evalution and return the solution"""
    if self.sampleMethod == 'Gaussian':
      return self.evalGaussian(instr)
    elif self.sampleMethod == 'Uniform':
      return self.evalUniform(instr)

  def evalUniform(self, instr):
    # initial parameters
    t, _min, _max = self.__initUniformParams()

    # random sample all dimension each time
    while t < self.maxits:
      # sample N data and sort
      x = self.__uniformSampleData(_min, _max)
      s = self.__functionReward(instr, x)
      s = self.__sortSample(s)
      x = np.array([ s[i][0] for i in range(np.shape(s)[0]) ] )

      # update parameters
      _min, _max = self.__updateUniformParams(x)
      t += 1

    return (_min + _max) / 2.
    

  def evalGaussian(self, instr):
    # initial parameters
    t, mu, sigma = self.__initGaussianParams()

    # random sample all dimension each time
    while t < self.maxits:
      # sample N data and sort
      x = self.__gaussianSampleData(mu, sigma)
      s = self.__functionReward(instr, x)
      s = self.__sortSample(s)
      x = np.array([ s[i][0] for i in range(np.shape(s)[0]) ] )

      # update parameters
      mu, sigma = self.__updateGaussianParams(x)
      t += 1

    return mu

  def __initGaussianParams(self):
    """initial parameters t, mu, sigma"""
    t = 0
    mu = np.zeros(self.d)
    sigma = np.ones(self.d) * self.init_coef
    return t, mu, sigma

  def __updateGaussianParams(self, x):
    """update parameters mu, sigma"""
    mu = x[0:self.Ne,:].mean(axis=0)
    sigma = x[0:self.Ne,:].std(axis=0)
    return mu, sigma
    
  def __gaussianSampleData(self, mu, sigma):
    """sample N examples"""
    sample_matrix = np.zeros((self.N, self.d))
    for j in range(self.d):
      sample_matrix[:,j] = np.random.normal(loc=mu[j], scale=sigma[j]+1e-17, size=(self.N,))
      if self.v_min is not None and self.v_max is not None:
        sample_matrix[:,j] = np.clip(sample_matrix[:,j], self.v_min[j], self.v_max[j])
    return sample_matrix

  def __initUniformParams(self):
    """initial parameters t, mu, sigma"""
    t = 0
    _min = self.v_min if self.v_min else -np.ones(self.d)
    _max = self.v_max if self.v_max else  np.ones(self.d)
    return t, _min, _max

  def __updateUniformParams(self, x):
    """update parameters mu, sigma"""
    _min = np.amin(x[0:self.Ne,:], axis=0)
    _max = np.amax(x[0:self.Ne,:], axis=0)
    return _min, _max
    
  def __uniformSampleData(self, _min, _max):
    """sample N examples"""
    sample_matrix = np.zeros((self.N, self.d))
    for j in range(self.d):
      sample_matrix[:,j] = np.random.uniform(low=_min[j], high=_max[j], size=(self.N,))
    return sample_matrix

  def __functionReward(self, instr, x):
    bi = np.reshape(instr, [1, -1])
    bi = np.repeat(bi, self.N, axis=0)
    return zip(x, self.func(bi, x))

  def __sortSample(self, s):
    """sort data by function return"""
    s = sorted(s, key=lambda x: x[1], reverse=self.reverse)
    return s
    
def func(a1, a2):
  c = a1 - a2
  return [ _c[0]*_c[0] + _c[1]*_c[1] for _c in c ]

if __name__ == '__main__':
  cem = CEM(func, 2, sampleMethod='Uniform', v_min=[-5.,-5.], v_max=[5.,5.])
  t = np.array([1,2])
  v = cem.eval(t)
  print(v, func(t.reshape([-1, 2]), v.reshape([-1,2])))
