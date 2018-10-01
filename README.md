# Cross Entropy Method

## Introduction

>  The Cross Entropy Method (CEM) deleveloped by Reuven Rubinstein is a general Monte Corlo approach to combinatorial and continuous multi-extremal optimization and importance sampling. 
>
> -- from [Wikipedia Cross-entorpy method](https://en.wikipedia.org/wiki/Cross-entropy_method)

## Generic CE Algorithm

The idea is to random sample the data and iterate for `maxits` to approach the target function $f$.

1. choose initial parameters $v^{(0)},\ \mu^{(0)}$ and $\sigma^{(0)};$ set $t$ = 1

2. generate `N `samples $X_1, X_2, ..., X_n$ from Gaussian distribution base on $\mu^{(t)}, \sigma^{(t)}$

3. solve for $v^{(t)}$, where:

   $$v^{(t)} = \arg\min_{x\in X}\ f(x)$$

4.  select the best `Ne` samples to update $\mu^{(t)}, \sigma^{(t)}​$

5. If convergence is reached then **stop**; otherwise, increase $t$ by 1 and reiterate from step 2.

## My Implementation

1. Implement the original CE method.
2. Change CE method sample method to Uniform distribution

## Class Introduction

```python
class CEM():
    def __init__(self, func, d, maxits=500, N=100, Ne=10, argmin=True, v_min=None, v_max=None, init_scale=1, sampleMethod='Gaussian')
        self.func = func              # target function
        self.d = d                    # dimension of function input X
        self.maxits = maxits          # maximum iteration
        self.N = N                    # sample N examples each iteration
        self.Ne = Ne                  # using better Ne examples to update mu and sigma
        self.reverse = not argmin     # try to maximum or minimum the target function
        self.v_min = v_min            # the value minimum
        self.v_max = v_max            # the value maximum
        self.init_coef = init_scale   # sigma initial value
        
        self.sampleMethod = sampleMethod  # which sample method: gaussian or uniform, 
                                          # default to gaussian
```

## Usage 

### import class

``` python
from cem import CEM
```

### init class

```python
cem = CEM(my_func, 2)
```

### evalution

```python
cem.eval()
```

## Example

### function without another input

```python
from cem import CEM

def my_func(x):
  return [ _x[0]*_x[0] + _x[1]*_x[1] for _x in x ]

if __name__ == '__main__':
    cem = CEM(func, 2)
    v = cem.eval()
    print(v, my_func(v.reshape(-1, 2)))
```

### function with other inputs

```python
from cem import CEM

def my_func(a1, a2):
	c = a1 - a2
	return [ _c[0]*_c[0] + _c[1]*_c[1] for _c in c ]

if __name__ == '__main__':
    cem = CEM(func, 2, v_min=[-5.,-5.], v_max=[5.,5.], sampleMethod='Uniform')
    t = np.array([1,2])
    v = cem.eval(t)
    print(v, my_func(t.reshape([-1, 2]), v.reshape([-1,2])))
```

