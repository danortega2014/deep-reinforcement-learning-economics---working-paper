```
using POMDPs,POMDPModelTools

Base.@kwdef struct firmgame <: POMDP{Float64, Symbol, Tuple{Float64,Float64,Float64,Float64}} # POMDP{State, Action, Observation
    discount_factor::Float64 = 0.95 # discount
end
m = firmgame()

POMDPs.actions(pomdp::firmgame) = [:np2, :np1, :p0, :p1, :p2, :invest, :borrow, :payback]

using Distributions

function POMDPs.gen(m::firmgame, s::Float64, a::Symbol, rng)
    s1  = s +  rand(Normal(0.0, 1.0))
    # transition model
    if a==:np2
        sp = (s1,)
    elseif a==:np1
        sp = (s1,)
    elseif a==:p0
        sp = (s1,)
    elseif a==:p1 
        sp = (s1,)
    elseif a==:p2
        sp = (s1,)
    elseif a==:invest
        sp = (s1,)
    elseif a==:borrow
        sp = (s1,)
    else 
        sp = (s1,)
    end

    o1 = sp + rand(Normal(0.0, 1.0))
    # observation model
    if a == :np2  
        o = (o1,1.0,1.0,1.0,)
    elseif a== :np1
        o = (o1,1.0,1.0,1.0,)
    elseif a== :p0 
        o = (o1,1.0,1.0,1.0,)
    elseif a ==:p1 
        o = (o1,1.0,1.0,1.0,)
    elseif a== :p2
        o = (o1,1.0,1.0,1.0,)
    elseif a== :invest 
        o = (o1,1.0,1.0,1.0,)
    elseif a== :borrow
        o = (o1,1.0,1.0,1.0,)
    elseif a== :payback
        o = (o1,1.0,1.0,1.0,)
    else 
        o = (o1,1.0,1.0,1.0,)
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


function POMDPs.convert_o(::Type{A}, o::Tuple{Float64,Float64,Float64,Float64}, m::firmgame) where A<:AbstractArray
    o = convert(A, [o...])
    return o
end

m = firmgame()


using DeepQLearning
using Flux
using POMDPPolicies


model = Chain(Dense(2, 32), Dense(32, length(actions(m))))

exploration = EpsGreedyPolicy(m, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2))

solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, 
                             exploration_policy = exploration,
                             learning_rate=0.005,log_freq=500,
                             recurrence=false,double_q=true, prioritized_replay=true)
policy = solve(solver,m)
```
new error:

```
julia> policy = solve(solver,m)
ERROR: MethodError: no method matching +(::Tuple{Float64}, ::Float64)
Closest candidates are:
  +(::Any, ::Any, ::Any, ::Any...) at operators.jl:538
  +(::Bool, ::T) where T<:AbstractFloat at bool.jl:103
  +(::Missing, ::Number) at missing.jl:115
  ...
Stacktrace:
 [1] gen(::firmgame, ::Float64, ::Symbol, ::Random.MersenneTwister) at .\REPL[26]:21
 [2] macro expansion at C:\Users\danor\.julia\packages\POMDPs\agdPZ\src\gen_impl.jl:23 [inlined]
 [3] genout(::DDNOut{(:sp, :o, :r, :info)}, ::firmgame, ::Float64, ::Symbol, ::Random.MersenneTwister) at C:\Users\danor\.julia\packages\POMDPs\agdPZ\src\gen_impl.jl:19
 [4] (::RLInterface.var"#23#f#4")(::firmgame, ::Float64, ::Symbol, ::Random.MersenneTwister) at C:\Users\danor\.julia\packages\POMDPs\agdPZ\src\generative.jl:65
 [5] step!(::RLInterface.POMDPEnvironment{Array{Float32,1},firmgame,Float64,Random.MersenneTwister}, ::Symbol) at C:\Users\danor\.julia\packages\RLInterface\Vw7YT\src\RLInterface.jl:107
 [6] #populate_replay_buffer!#13 at C:\Users\danor\.julia\packages\DeepQLearning\bpMyG\src\prioritized_experience_replay.jl:116 [inlined]
 [7] initialize_replay_buffer(::DeepQLearningSolver{EpsGreedyPolicy{LinearDecaySchedule{Float64},Random._GLOBAL_RNG,Array{Symbol,1}}}, ::RLInterface.POMDPEnvironment{Array{Float32,1},firmgame,Float64,Random.MersenneTwister}, ::Dict{Symbol,Int64}) at C:\Users\danor\.julia\packages\DeepQLearning\bpMyG\src\solver.jl:183
 [8] solve(::DeepQLearningSolver{EpsGreedyPolicy{LinearDecaySchedule{Float64},Random._GLOBAL_RNG,Array{Symbol,1}}}, ::RLInterface.POMDPEnvironment{Array{Float32,1},firmgame,Float64,Random.MersenneTwister}) at C:\Users\danor\.julia\packages\DeepQLearning\bpMyG\src\solver.jl:48        
 [9] solve(::DeepQLearningSolver{EpsGreedyPolicy{LinearDecaySchedule{Float64},Random._GLOBAL_RNG,Array{Symbol,1}}}, ::firmgame) at C:\Users\danor\.julia\packages\DeepQLearning\bpMyG\src\solver.jl:37
 [10] top-level scope at REPL[38]:1

```
