```
using POMDPs,POMDPModelTools
# state are real demand
Base.@kwdef struct firmgame <: POMDP{Float64, Symbol, Tuple{Float64,Int64, Int64}} # POMDP{State, Action, Observation
    discount_factor::Float64 = 0.95 # discount
end
m = firmgame()

POMDPs.actions(pomdp::firmgame) = [:np2, :np1, :p0, :p1, :p2, :invest, :borrow, :payback]

using Distributions

function POMDPs.gen(m::firmgame, s::Float64, a::Symbol, rng)
    s1  = s +  rand(Normal(0, 1))
    # transition model
    if a==:np2
        sp = Deterministic(s1)
    elseif a==:np1
        sp =  Deterministic(s1)
    elseif a==:p0
        sp =  Deterministic(s1)
    elseif a==:p1 
        sp = Deterministic(s1)
    elseif a==:p2
        sp = Deterministic(s1)
    elseif a==:invest
        sp = Deterministic(s1)
    elseif a==:borrow
        sp = Deterministic(s1)
    else 
        sp = Deterministic(s1)
    end

    o1 = sp + rand(Normal(0, 1))
    # observation model
    if a == :np2  
        o =  Deterministic(o1,1,1,1)
    elseif a== :np1
        o = Deterministic(o1,1,1,1)
    elseif a== :p0 
        o = Deterministic(o1,1,1,1)
    elseif a ==:p1 
        o = Deterministic(o1,1,1,1)
    elseif a== :p2
        o = Deterministic(o1,1,1,1)
    elseif a== :invest 
        o = Deterministic(o1,1,1,1)
    elseif a== :borrow
        o = Deterministic(o1,1,1,1)
    elseif a== :payback
        o = Deterministic(o1,1,1,1)
    else 
        o = Deterministic(o1,1,1,1)
    end
    
    # reward model
    if a == :pay
        r = 10
    elseif a ==:borrow
        r = 100
    else 
       r = -190
    end 
    # create and return a NamedTuple
    return (sp=sp, o=o, r=r)
end

POMDPs.discount(pomdp::firmgame) = pomdp.discount_factor
POMDPs.initialstate(pomdp::firmgame) = Deterministic(34.0)
POMDPs.initialobs(m::firmgame, s::Float64) = Deterministic((s,1.0,1.0,1.0))
m = firmgame()



using DeepQLearning
using Flux
using POMDPPolicies


model = Chain(Dense(2, 32), Dense(32, length(actions(m))))

exploration = EpsGreedyPolicy(m, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2))

solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, 
                             exploration_policy = exploration,
                             learning_rate=0.005,log_freq=500,
                             recurrence=true,double_q=true, prioritized_replay=true)
policy = solve(solver, m)


using POMDPSimulators

rsum = 0.0
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=100)
    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o r: $r")
    global rsum += r
end
println("Undiscounted reward was $rsum.")
using Distributions
Categorical([.10, .20, .40, .20, .10], 5)
```
Error:
```
ERROR: MethodError: no method matching +(::Int64, ::Array{Float64,1})
For element-wise addition, use broadcasting with dot syntax: scalar .+ array
Closest candidates are:
  +(::Any, ::Any, ::Any, ::Any...) at operators.jl:538
  +(::T, ::T) where T<:Union{Int128, Int16, Int32, Int64, Int8, UInt128, UInt16, UInt32, UInt64, UInt8} at int.jl:86
  +(::Integer, ::Integer) at int.jl:918
```
