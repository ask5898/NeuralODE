using ComponentArrays
using DiffEqFlux
using Lux
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using OrdinaryDiffEq
using Plots
using Random

rng = Random.default_rng()
q0 = Float32[0.6]
p0 = Float32[2.0,0.2,1.0, 2.0]
tspan = (0.0f0, 5.0f0)
datasize = 30
tstep = range(tspan[1], tspan[2], length=datasize)
loss_all = Vector{Float32}()

function LangMuir!(dq, q, p ,t)
    ka,kd,qm,c = p
    dq .= -ka*c*(q.- qm) .- q.*kd
end

ode = ODEProblem(LangMuir!, q0, tspan, p0)

ode_data = Array(solve(ode, Tsit5(); saveat = tstep))
#plt = scatter(tstep, ode_data'; label = "data"())
plt = plot(ode_data', xaxis = "t (time)", yaxis="q(Adsorbed amount)", fmt = :png)
display(plt)
#savefig(plt, "underlying.png")


### Define the Neural ODE
dudt2 = Lux.Chain(Lux.Dense(1, 50, tanh),Lux.Dense(50, 1))
p, st = Lux.setup(rng, dudt2)


prob_neuralode = NeuralODE(dudt2, tspan,  Tsit5(); saveat = tstep)

predict_neuralode(p) = Array(prob_neuralode(q0, p, st)[1])



### Define loss function as the difference between actual ground truth data and Neural ODE prediction
function loss_neuralode(p)
    pred = predict_neuralode(p)
    #print(size(ode_data))
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation

# Users should change doplot=true to see the plots callbacks
callback = function (state, l; doplot = true)
    # plot current prediction against data
    println(l)
    push!(loss_all,l[1])
    if doplot
        pred = predict_neuralode(state.u)
        plt = scatter(tstep, ode_data'; label = "data")
        scatter!(plt, tstep, pred'; label = "prediction")
        display(plot(plt))
        #savefig(plt, "BFGS.png")
    end
    return false
end

pinit = ComponentArray(p)
#print(size(pinit))

callback((; u = pinit), loss_neuralode(pinit); doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoForwardDiff()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x)[1], adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback = callback, maxiters = 200)

callback(result_neuralode, loss_neuralode(result_neuralode.u); doplot = true)

result_neuralode2 = Optimization.solve(optprob2,Optim.BFGS(); callback,allow_f_increases = false)
#ll = plot(loss_all, label="loss")
#savefig(ll, "loss.png")
