using OrdinaryDiffEq, Flux, DiffEqFlux, DiffEqSensitivity, Zygote, RecursiveArrayTools

u0 = Float32[0.; 2.]
du0 = Float32[0.; 0.]
tspan = (0.0f0, 1.0f0)
t = range(tspan[1], tspan[2], length=20)

model = SlowChain(SlowDense(2, 50, tanh), SlowDense(50, 2))
p = initial_params(model)
ff(du,u,p,t) = model(u,p)
prob = SecondOrderODEProblem{false}(ff, du0, u0, tspan, p)

function predict(p)
    Array(concrete_solve(prob, Tsit5(), ArrayPartition(du0,u0), p, saveat=t, sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())))
end

correct_pos = Float32.(transpose(hcat(collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end])))

function loss_n_ode(p)
    pred = predict(p)
    sum(abs2, correct_pos .- pred[1:2, :]), pred
end

data = Iterators.repeated((), 1000)
opt = ADAM(0.01)

l1 = loss_n_ode(p)

cb = function (p,l,pred)
    println(l)
    l < 0.01 && Flux.stop()
end

res = DiffEqFlux.sciml_train(loss_n_ode, p, opt, cb=cb, maxiters = 100)
l2 = loss_n_ode(res.minimizer)
@test l2 < l1

function predict(p)
    Array(concrete_solve(prob, Tsit5(), ArrayPartition(du0,u0), p, saveat=t, sensealg = QuadratureAdjoint(autojacvec=ZygoteVJP())))
end

correct_pos = Float32.(transpose(hcat(collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end])))

function loss_n_ode(p)
    pred = predict(p)
    sum(abs2, correct_pos .- pred[1:2, :]), pred
end

data = Iterators.repeated((), 1000)
opt = ADAM(0.01)

loss_n_ode(p)

cb = function (p,l,pred)
    println(l)
    l < 0.01 && Flux.stop()
end

res = DiffEqFlux.sciml_train(loss_n_ode, p, opt, cb=cb, maxiters = 100)
l2 = loss_n_ode(res.minimizer)
@test l2 < l1

function predict(p)
    Array(concrete_solve(prob, Tsit5(), ArrayPartition(du0,u0), p, saveat=t, sensealg = BacksolveAdjoint(autojacvec=ZygoteVJP())))
end

correct_pos = Float32.(transpose(hcat(collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end])))

function loss_n_ode(p)
    pred = predict(p)
    sum(abs2, correct_pos .- pred[1:2, :]), pred
end

data = Iterators.repeated((), 1000)
opt = ADAM(0.01)

loss_n_ode(p)

cb = function (p,l,pred)
    println(l)
    l < 0.01 && Flux.stop()
end

res = DiffEqFlux.sciml_train(loss_n_ode, p, opt, cb=cb, maxiters = 100)
l2 = loss_n_ode(res.minimizer)
@test l2 < l1
