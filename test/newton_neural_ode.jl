using DiffEqFlux, Flux, Optim, OrdinaryDiffEq

n = 5  # number of ODEs
tspan = (0.0, 1.0)

d = 5  # number of data pairs
x_data = rand(n, 5)
y_data = rand(n, 5)
training_data = Iterators.cycle([(x_data[:, i], y_data[:, i]) for i in 1:d])

NN = Chain(Dense(n, 10n, tanh),
           Dense(10n, n))

@info "ROCK4"
nODE = NeuralODE(NN, tspan, ROCK4(), reltol=1e-4, saveat=[tspan[end]])

loss_function(θ, x, y) = Flux.mse(y, nODE(x, θ))
res = DiffEqFlux.sciml_train(loss_function, nODE.p, LBFGS(), training_data)
res = DiffEqFlux.sciml_train(loss_function, nODE.p, NewtonTrustRegion(), training_data)

NN = SlowChain(SlowDense(n, 10n, tanh),
               SlowDense(10n, n))

@info "ROCK2"
nODE = NeuralODE(NN, tspan, ROCK2(), reltol=1e-4, saveat=[tspan[end]])

loss_function(θ, x, y) = Flux.mse(y, nODE(x, θ))
res = DiffEqFlux.sciml_train(loss_function, nODE.p, LBFGS(), training_data)
res = DiffEqFlux.sciml_train(loss_function, nODE.p, NewtonTrustRegion(), training_data)

@info "ROCK4"
nODE = NeuralODE(NN, tspan, ROCK4(), reltol=1e-4, saveat=[tspan[end]])

loss_function(θ, x, y) = Flux.mse(y, nODE(x, θ))
res = DiffEqFlux.sciml_train(loss_function, nODE.p, LBFGS(), training_data)
res = DiffEqFlux.sciml_train(loss_function, nODE.p, NewtonTrustRegion(), training_data)
