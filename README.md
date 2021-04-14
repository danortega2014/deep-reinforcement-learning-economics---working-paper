# v3-

```

using POMDPs,POMDPModelTools
# state are real demand
Base.@kwdef struct firmgame <: POMDP{Int64, Symbol, Tuple{Int64,Int64, Int64}} # POMDP{State, Action, Observation
    discount_factor::Float64 = 0.95 # discount
end
m = firmgame()

POMDPs.actions(pomdp::firmgame) = [:np2, :np1, :p0, :p1, :p2, :invest, :borrow, :payback]

function POMDPs.gen(m::firmgame, s, a, rng)
    # transition model
    if a==:np2
        sp = SparseCat([-2+s,-1+s, 0+s, 1+s, 2+s], [.10, .20, .40, .20,.10]) #add s
    elseif a==:np1
        sp = SparseCat([-2+s,-1+s, 0+s, 1+s, 2+s], [.10, .20, .40, .20,.10]) #add s
    elseif a==:p0
        sp = SparseCat([-2+s,-1+s, 0+s, 1+s, 2+s], [.10, .20, .40, .20,.10]) #add s
    elseif a==:p1 
        sp = SparseCat([-2+s,-1+s, 0+s, 1+s, 2+s], [.10, .20, .40, .20,.10]) #add supply shocks
    elseif a==:p2
        sp = SparseCat([-2+s,-1+s, 0+s, 1+s, 2+s], [.10, .20, .40, .20,.10])  
    elseif a==:invest
        sp = SparseCat([-2+s,-1+s, 0+s, 1+s, 2+s], [.10, .20, .40, .20,.10])  
    elseif a==:borrow
        sp = SparseCat([-2+s,-1+s, 0+s, 1+s, 2+s], [.10, .20, .40, .20,.10]) 
    else 
        sp = SparseCat([-2+s,-1+s, 0+s, 1+s, 2+s], [.10, .20, .40, .20,.10]) 
    end
    # observation model
    if a == :np2  
        o = SparseCat([sp+1,sp+2,sp+3,sp+4, sp+5], [.40,.25,.20, .10, .5]), 1,1,1
    elseif a== :np1
        o = SparseCat([1,2,3,4,5,6,7,8,9], [.40,.25,.20, .10, .5]), 1,1,1
    elseif a== :p0 
        o = SparseCat([1,2,3,4,5,6,7,8,9], [.60,.05,.05,.05,.05,.05,.05,.05,.05]),1,1,1
    elseif a ==:p1 
        o = SparseCat([1,2,3,4,5,6,7,8,9], [.60,.05,.05,.05,.05,.05,.05,.05,.05]), 1,1,1
    elseif a== :p2
        o = SparseCat([1,2,3,4,5,6,7,8,9], [.60,.05,.05,.05,.05,.05,.05,.05,.05]),1,1,1
    elseif a== :invest 
        o = SparseCat([1,2,3,4,5,6,7,8,9], [.60,.05,.05,.05,.05,.05,.05,.05,.05]), 1,1,1
    elseif a== :borrow
        o = SparseCat([1,2,3,4,5,6,7,8,9], [.60,.05,.05,.05,.05,.05,.05,.05,.05]),1,1,1
    elseif a== :payback
        o = SparseCat([1,2,3,4,5,6,7,8,9], [.60,.05,.05,.05,.05,.05,.05,.05,.05]),1,1,1
    else 
        o = SparseCat([1,2,3,4,5,6,7,8,9], [.60,.05,.05,.05,.05,.05,.05,.05,.05]), 1,1,1
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
POMDPs.initialstate(pomdp::firmgame) = 34
m = firmgame()

using DeepQLearning
using Flux
using POMDPPolicies


model = Chain(Dense(2, 32), Dense(32, length(actions(m))))

exploration = EpsGreedyPolicy(m, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2))

solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, 
                             exploration_policy = exploration,
                             learning_rate=0.005,log_freq=500,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
policy = solve(solver, m)


using POMDPSimulators

rsum = 0.0
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=100)
    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o r: $r")
    global rsum += r
end
println("Undiscounted reward was $rsum.")


Error:
````
julia> policy = solve(solver, m)
ERROR: MethodError: Cannot `convert` an object of type Array{Float64,1} to an object of type Int64
Closest candidates are:
  convert(::Type{Int64}, ::Type{CUDAnative.AS.Generic}) at C:\Users\danor\.julia\packages\CUDAnative\C91oY\src\device\pointer.jl:23
  convert(::Type{Int64}, ::Type{CUDAnative.AS.Global}) at C:\Users\danor\.julia\packages\CUDAnative\C91oY\src\device\pointer.jl:24
  convert(::Type{Int64}, ::Type{CUDAnative.AS.Shared}) at C:\Users\danor\.julia\packages\CUDAnative\C91oY\src\device\pointer.jl:25
  ...
```
