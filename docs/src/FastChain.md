# SlowChain

The `SlowChain` system is a Flux-like explicit parameter neural network
architecture system for less overhead in smaller neural networks. For neural
networks with layers of lengths >~200, these optimizations are overshadowed by
the cost of matrix multiplication. However, for smaller layer operations this
architecture can reduce a lot of the overhead traditionally seen in neural
network architectures and thus is recommended in a lot of scientific machine
learning usecaes.

## Basics

The basic is that `SlowChain` is a collection of functions of two values,
`(x,p)`, and chains these functions to call one after the next. Each layer in
this chain gets pre-defined amount of parameters sent to it. For example,

```julia
f = SlowChain((x,p) -> x.^3,
              SlowDense(2,50,tanh),
              SlowDense(50,2))
```

`SlowChain` here has a `2*50 + 50` length parameter `SlowDense(2,50,tanh)` function
and a `50*2 + 2` parameter function `SlowDense(50,2)`. The first function gets
the default number of parameters which is 0. Thus `f(x,p)` is equivalent to the
following code:

```julia
function f(x,p)
  tmp1 = x.^3
  len1 = paramlength(SlowDense(2,50,tanh))
  tmp2 = SlowDense(2,50,tanh)(tmp1,@view p[1:len1])
  tmp3 = SlowDense(50,2)(tmp2,@view p[len2:end])
end
```

`SlowChain` functions thus require that the vector of neural network parameters
is passed to it on each call, making the setup explicit in the passed parameters.

To get initial parameters for the optimization of a function defined by a
`SlowChain`, one simply calls `initial_params(f)` which returns the concatenation
of the initial parameters for each layer. Notice that since all parameters are
explicit, constructing and reconstructing chains/layers can be a memory-free
operation, since the only memory is the parameter vector itself which is handled
by the user.

### SlowChain Interface

The only requirement to be a layer in `SlowChain` is to be a 2-argument function
`l(x,p)` and define the following traits:

- `paramlength(::typeof(l))`: The number of parameters from the parameter vector
  to allocate to this layer. Defaults to zero.
- `initial_params(::typeof(l))`: The function for defining the initial parameters
  of the layer. Should output a vector of length matching `paramlength`. Defaults
  to `Float32[]`.

## SlowChain-Compatible Layers

The following pre-defined layers can be used with `SlowChain`:

```@docs
SlowDense
StaticDense
```
