```
using POMDPs,POMDPModelTools
# state are real demand
Base.@kwdef struct firmgame <: POMDP{Float64, Symbol, Tuple{Float64,Float64,Float64, Float64}} # POMDP{State, Action, Observation
    discount_factor::Float64 = 0.95 # discount
end
m = firmgame()

POMDPs.actions(pomdp::firmgame) = [:np2, :np1, :p0, :p1, :p2, :invest, :borrow, :payback]

using Distributions

function POMDPs.gen(m::firmgame, s::Float64, a::Symbol, rng)
    s1  = s +  rand(Normal(0.0, 1.0))
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

    o1 = sp + rand(Normal(0.0, 1.0))
    # observation model
    if a == :np2  
        o = Deterministic(o1,1.0,1.0,1.0)
    elseif a== :np1
        o = Deterministic(o1,1.0,1.0,1.0)
    elseif a== :p0 
        o = Deterministic(o1,1.0,1.0,1.0)
    elseif a ==:p1 
        o = Deterministic(o1,1.0,1.0,1.0)
    elseif a== :p2
        o = Deterministic(o1,1.0,1.0,1.0)
    elseif a== :invest 
        o = Deterministic(o1,1.0,1.0,1.0)
    elseif a== :borrow
        o = Deterministic(o1,1.0,1.0,1.0)
    elseif a== :payback
        o = Deterministic(o1,1.0,1.0,1.0)
    else 
        o = Deterministic(o1,1.0,1.0,1.0)
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

function POMDPs.convert_o(::Type{A}, o::Tuple{Float64,Float64,Float64,Float64}, m::firmgame) where A<:AbstractArray
    v = copyto!(A(undef, 4), o)
    return v
end
POMDPs.convert_o(::Type{Tuple{Float64,Float64,Float64,Float64}}, o::A, m::firmgame) where A<:AbstractArray = (o[1], o[2], o[3], o[4])

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

```

error:

```
ERROR: MethodError: no method matching convert_o(::Type{Array{Float32,1}}, ::NTuple{4,Float64}, ::firmgame)
Closest candidates are:
  convert_o(::Type{A1}, ::A2, ::Union{MDP, POMDP}) where {A1<:AbstractArray, A2<:AbstractArray} at C:\Users\danor\.julia\packages\POMDPs\agdPZ\src\pomdp.jl:181
  convert_o(::Type{A}, ::Number, ::Union{MDP, POMDP}) where A<:AbstractArray at C:\Users\danor\.julia\packages\POMDPs\agdPZ\src\pomdp.jl:183
  convert_o(::Type{V}, ::Any, ::FullyObservablePOMDP) where V<:AbstractArray at C:\Users\danor\.julia\packages\POMDPModelTools\N593Y\src\fully_observable_pomdp.jl:19
  
 ```
 error after adding convert_o, which I'm not sure if was implemented correctly:
 ```
 
julia> policy = solve(solver,m)
ERROR: MethodError: no method matching +(::Deterministic{Float64}, ::Float64)
Closest candidates are:
  +(::Any, ::Any, ::Any, ::Any...) at operators.jl:538
  +(::Bool, ::T) where T<:AbstractFloat at bool.jl:103
  +(::Missing, ::Number) at missing.jl:115
```
