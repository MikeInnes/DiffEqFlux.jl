using DiffEqFlux, OrdinaryDiffEq, Optim, Flux, Zygote, Test

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

Slowdudt2 = SlowChain((x,p) -> x.^3,
             SlowDense(2,50,tanh),
             SlowDense(50,2))
Slow_n_ode = NeuralODE(Slowdudt2,tspan,Tsit5(),saveat=t)

function Slow_predict_n_ode(p)
  Slow_n_ode(u0,p)
end

function Slow_loss_n_ode(p)
    pred = Slow_predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

staticdudt2 = SlowChain((x,p) -> x.^3,
                        StaticDense(2,50,tanh),
                        StaticDense(50,2))
static_n_ode = NeuralODE(staticdudt2,tspan,Tsit5(),saveat=t)

function static_predict_n_ode(p)
  static_n_ode(u0,p)
end

function static_loss_n_ode(p)
    pred = static_predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

dudt2 = Chain((x) -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
n_ode = NeuralODE(dudt2,tspan,Tsit5(),saveat=t)

function predict_n_ode(p)
  n_ode(u0,p)
end

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

p = initial_params(Slowdudt2)
_p,re = Flux.destructure(dudt2)
@test Slowdudt2(ones(2),_p) ≈ dudt2(ones(2))
@test staticdudt2(ones(2),_p) ≈ dudt2(ones(2))
@test Slow_loss_n_ode(p)[1] ≈ loss_n_ode(p)[1]
@test static_loss_n_ode(p)[1] ≈ loss_n_ode(p)[1]
@test Zygote.gradient((p)->Slow_loss_n_ode(p)[1], p)[1] ≈ Zygote.gradient((p)->loss_n_ode(p)[1], p)[1] rtol=1e-3
@test Zygote.gradient((p)->static_loss_n_ode(p)[1], p)[1] ≈ Zygote.gradient((p)->loss_n_ode(p)[1], p)[1] rtol=1e-3

#=
using BenchmarkTools
@btime Zygote.gradient((p)->static_loss_n_ode(p)[1], p)
@btime Zygote.gradient((p)->Slow_loss_n_ode(p)[1], p)
@btime Zygote.gradient((p)->loss_n_ode(p)[1], p)
=#
