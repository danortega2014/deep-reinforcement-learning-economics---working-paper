cd("C:\\Users\\danor\\Desktop\\models")

using POMDPs,POMDPModelTools
using Random, Distributions 

Random.seed!(1234)

Base.@kwdef mutable struct firmgame <: MDP{Tuple{Float64,Float64,Float64,Float64,Float64,Float64, Float64}, Symbol} # POMDP{State, Action, Observation
    discount_factor::Float64 = 0.98 # discount
    terminalx::Bool = 0
end 
#observations are nomprice, wealth, debt, capital, interest rate,labor
m = firmgame()
# actions  np2 to p2 represent what the agent expects the change in demand to be in the price, selling less when he expects low demand, and vice versa. 
POMDPs.actions(pomdp::firmgame) = [:np2, :np1, :p0, :p1, :p2, :smallinvest, :largeinvest, :smallborrow, :largeborrow, :smallpayback, :largepayback, :smallconsume, :largeconsume, :smallcraft, :largecraft] # 15 actions 

ms = 0.0 
z = 0.0  
totalnomprice = 0.09
steps = 0

function POMDPs.gen(m::firmgame, s::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}, a::Symbol, rng)
    global m_t =  round(rand(Normal(0.0, .6))) 
    global inflationshock =  round(rand(Normal(0.02, .55)))# -2:2
    nomprice, wealth, debt, capital, interest, labor, inventory  = s
    interest_ = round(max(0.0, interest - (inflationshock/130.0) + (debt/150000.0)), digits = 4)
    m.terminalx = 0 
    global ms += inflationshock  
    global z += m_t
    global steps += 1
    global totalnomprice =+ nomprice 
    priceindex = totalnomprice/steps
    demandindex = totalnomprice/steps 
    global y += 3.0 + m_t  #demand function 
    consumption_ = 0.0
    #transition function: wealth, debt, capital depends on action
    if a==:np2 && m_t == -2.0 && inventory >= 1.0  # supplied correctly and predicted change in real demand 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + nomprice*y - debt*interest/6.0)*(1.0 + interest/12.0) #agent sells the correct quantity which is equal to y (the demand)
        capital_ = capital*.95 
        inventory_ = inventory - 1.0
        labor_ = labor + 3.0

        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2), round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)
        

    elseif a ==:np2 && m_t > -2.0  && inventory >= 1.0 #predicted incorrectly --> undersupply --> supply shock to price, increase nom price by 1 
        nomprice_ = nomprice + m_t + inflationshock  
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth  + (nomprice*y) -debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory -1.0
        labor_ = labor + 3.0
 
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)
    elseif a ==:np1 && m_t == -1.0 && inventory >= 2.0 #supplied correctly and predicted change in real demand 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + nomprice*y - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 2.0
        labor_ = labor + 3.0
 
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)
    elseif a ==:np1 && m_t > -1.0 && inventory >= 2.0  #predicted incorrectly --> undersupply --> supply shock to price, increase nom price by 1 
        nomprice_ = nomprice + m_t + inflationshock  
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + nomprice*y - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 2.0
        labor_ = labor + 3.0

        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a == :np1 && m_t < -1.0 && inventory >= 2.0 #predicted incorrectly --> oversupply --> supply shock to price, increase nom price by 1 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth  + nomprice*y - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 2.0
        labor_ = labor + 3.0
 

        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a==:p0 && m_t == 0.0 && inventory >= 3.0 #supplied correctly and predicted change in real demand 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + nomprice*y - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 3.0
        labor_ = labor + 3.0
 

        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a==:p0 && m_t == -0.0 && inventory >= 3.0 #supplied correctly and predicted change in real demand 
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + nomprice*y - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 3.0
        labor_ = labor + 3.0

        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a==:p0 && m_t > 0.0  && inventory >= 3.0 #predicted incorrectly --> undersupply --> supply shock to price, increase nom price by 1 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + (nomprice*y)/2 - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 3.0
        labor_ = labor + 3.0
 

        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a==:p0 && m_t < 0.0 && inventory >= 3.0 #predicted incorrectly --> oversupply --> supply shock to price, increase nom price by 1 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + (nomprice*y)/2 - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 3.0
        labor_ = labor + 3.0

        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)
        
    elseif a==:p1 && m_t == 1.0 && inventory >= 4.0 #supplied correctly and predicted change in real demand 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + nomprice*y - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 4.0
        labor_ = labor + 3.0

        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)
        
    elseif a==:p1 && m_t > 1.0 && inventory >= 4.0  #predicted incorrectly --> undersupply --> supply shock to price, increase nom price by 1 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth  + (nomprice*y)/2 - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 4.0
        labor_ = labor + 3.0

         sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a==:p1 && m_t < 1.0  && inventory >= 4.0  #predicted incorrectly --> oversupply --> supply shock to price, increase nom price by 1 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + (nomprice*y)/2  - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory -4.0 
        labor_ = labor + 3.0
 
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

        
    elseif a==:p2  && m_t == 2.0 && inventory >= 5.0  #supplied correctly and predicted change in real demand 
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth + nomprice*y - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 5.0 
        labor_ = labor + 3.0
 
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a==:p2 && m_t < 2.0 && inventory >= 5.0  #predicted incorrectly --> oversupply --> supply shock to price, increase nom price by 1 
        nomprice_ = nomprice + m_t + inflationshock 
        debt_ = debt*(1+interest/6.0)
        wealth_ = (wealth  + (nomprice*y)/2 -  debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory - 5.0 
        labor_ = labor + 3.0
        
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)
        
    elseif a== :smallinvest && wealth >= (50.0 + debt*interest/6.0)
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1.0+interest/6.0)
        wealth_ = (wealth - debt*interest/6.0 -50.0)*(1.0 + interest/12.0)
        capital_ = capital*.95  + 5.0 
        inventory_ = inventory 
        labor_ = labor + 3.0
 
        
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a== :largeinvest && wealth >= (100.0 + debt*interest/6.0) 
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1.0 + interest/6.0)
        wealth_ =  (wealth - debt*interest/6.0 -100.0)*(1.0 + interest/12.0)
        capital_ = capital*.95  + 10.0 
        inventory_ = inventory 
        labor_ = labor + 3.0
 
        
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a== :smallborrow && wealth >= 0.0
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1.0 +interest/6.0) + 100.0
        wealth_ = (wealth - debt*interest/6.0 + 100.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory 
        labor_ = labor + 3.0
 
        
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)


    elseif a== :largeborrow && wealth >= 0.0
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1.0 +interest/6.0) + 200.0
        wealth_ = (wealth - debt*interest/6.0 + 200.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory 
        labor_ = labor + 3.0
        
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a == :smallpayback && debt >=  95.0 && wealth >= (100 + debt*interest/6.0)
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = max(0.0, debt*(1.0 +interest/6.0) - 100.0) 
        wealth_ = (wealth - debt*interest/6.0 -100.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory 
        labor_ = labor + 3.0
 
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a == :largepayback && debt >= 195.0  && wealth >= (200 + debt*interest/6.0)
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = max(0.0, debt*(1.0 +interest/6.0) - 200.0)
        wealth_ = (wealth - debt*interest/6.0 -200.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory 
        labor_ = labor + 3.0
 
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a==:smallcraft && labor >= 12.0  
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1.0+interest/6.0)
        wealth_ = (wealth - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory + ((12)^.5)*(capital)^.5
        labor_ = labor - 12.0 + 3.0
        
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)
    
    elseif a==:largecraft && labor >= 24.0  
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1.0+interest/6.0)
        wealth_ = (wealth - debt*interest/6.0)*(1.0 + interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory + ((24)^.5)*(capital)^.5
        labor_ = labor - 24.0 + 3.0 
        
        
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)

    elseif a ==:smallconsume && wealth >= (50.0+.10*wealth)   
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1.0+interest/6.0)
        wealth_ = (wealth - debt*interest/6.0 - 50.0 - .10*wealth)*(1.0+interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory 
        labor_ = labor + 3.0
        consumption_ = 200.0 +.10*wealth 
        
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)
    
    elseif a ==:largeconsume && wealth >= (100.0 +.2*wealth)  
       
        nomprice_ = nomprice + m_t + inflationshock
        debt_ = debt*(1.0+interest/6.0)
        wealth_ = (wealth - debt*interest/6.0 -100.0 - .10*wealth)*(1.0+interest/12.0)
        capital_ = capital*.95 
        inventory_ = inventory 
        labor_ = labor + 3.0
        consumption_ = 400.0  +.20*wealth  

        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)
    
    else
        m.terminalx = 1 
        nomprice_ = 15.0
        wealth_ = 105.0
        debt_ =  0.0
        capital_ = 5.0
        interest_ = 0.02
        labor_ = 12.0
        inventory_ = 0.0
        sp = max(0.0, nomprice_), round(max(0.0, wealth_), digits= 2), round(max(0.0,debt_), digits =2),  round(max(0.0, capital_), digits =3), interest_, min(24,labor_),  round(inventory_, digits = 2)  # Deterministic((15.0, 105.0, 0.0 , 5.0, 0.02, 15.0, 0.0))
    
    end
    #end of transition func
    
    wealth_ = (wealth - debt*interest/6.0 -100.0 - .10*wealth)*(1.0+interest/12.0)
    debt_ = debt*(1.0+interest/6.0)

    if wealth == 0.0 && m.terminalx == 0 
        r = - -debt_
    elseif m.terminalx == 0 
        r = (consumption_) + 4*(labor_ - labor) + (wealth_ - wealth)
    else 
        r = -50000.0
    end
    # create and return a NamedTupleSE
    return (sp=sp, r=r)
end

POMDPs.isterminal(m::firmgame, s::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}) = m.terminalx == 1 

POMDPs.discount(pomdp::firmgame) = pomdp.discount_factor
POMDPs.initialstate(pomdp::firmgame) = Deterministic((15.0, 105.0, 0.0 , 5.0, 0.02, 12.0, 0.0))  #initialprice_, wealth_, debt_, capital_, interest_, labor, inventory


function POMDPs.convert_s(::Type{A}, s::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}, m::firmgame) where A<:AbstractArray
    s = convert(A, [s...])
    return s
end


env = firmgame()


using DeepQLearning
using Flux
using POMDPPolicies
using CuArrays #load if you want to send weights to gpu in solve 
using BSON: @save, @load


# @load "qnetwork.bson" model_27
qnetwork2 = Chain(LSTM(7, 28), Dense(28, 15), softmax)

weights = Flux.params(model5)
# @save "mymodely.bson" weights
# Flux.reset!(model5)
stepsx = 500000
exploration2 = SoftmaxPolicy(m, LinearDecaySchedule(start = 5.0, stop = .001, steps = stepsx))
# exploration2 = EpsGreedyPolicy(env, LinearDecaySchedule(start=1.0, stop=0.1, steps=.9*stepsx))
solvrz= DeepQLearningSolver(qnetwork = qnetwork2, max_steps= stepsx, target_update_freq = 500,  batch_size = 50, train_freq = 4, 
                             exploration_policy = exploration2,
                             learning_rate=0.0005,log_freq=500, eval_freq = 100,  num_ep_eval = 15,
                             recurrence=true,double_q=true, dueling=false, prioritized_replay=false, prioritized_replay_alpha= .6, prioritized_replay_epsilon = .0000001,
                             prioritized_replay_beta = .6,  buffer_size = 300,  max_episode_length=500, train_start = 100 ,save_freq = stepsx/8, logdir = "C:\\Users\\danor\\Desktop\\models\\model")
policyzx = solve(solvrz,env)


# Flux.reset!(qnetw)
# resetstate!(policyzx)
#Flux.params(modelx)

using POMDPSimulators 
rsum = 0.0 
for (s,a,r) in stepthrough(env, policyzx, "s,a,r", max_steps=500)
    println("$m_t, $inflationshock, $s, $a, $r")
    global rsum += r
end  
println("Undiscounted reward was $rsum.") 