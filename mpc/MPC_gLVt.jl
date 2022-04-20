### Model Predictive Control of gLV-Temperature systems
# willsharpless@berkeley.edu or ucsd.edu
# July 28, 2021

using Polynomials, LinearAlgebra, Statistics, Combinatorics
using JuMP, Ipopt, GLPK, MosekTools
using ControlSystems, Graphs, GraphPlot #LightGraphs
using DataFrames, PyCall, Plots, LaTeXStrings, Dates, Colors
using JLD, Serialization
using Compose, Cairo, Fontconfig, ImageMagick

prepath = "/Users/willsharpless/Documents/Julia"
external_path = "/external/DriverSpecies-master/"
push!(LOAD_PATH, prepath * external_path)
using DriverSpeciesModule

np = pyimport("numpy")
colors = palette(:tab10);

if split(pwd(),'/')[2] == "Users" #local
    cd("/Users/willsharpless/Documents/")
end
to_arkin = pwd() * "/Julia/arkin/"

# future v: using SciML train to optimize nonlinear

#v4: save avg controllability of linearizations, iterates thru all possible input configs (& fp paths)
#v3: dLQR, clamped dLQR and Jump (questionable) working with linearized dynamics
# - often unable to drive to fixed point, despite possible with other xd (??)
# - complicated relationship between cost mats Q,R and trajectory (more input penalty -> more input ??)
# - I think linearizations are having trouble w T varying systems
# - currently iterating from and two all 4 n>=2 fp
# - added disturbance (no Tctrl vs Tctrl have same disturbance vec)
#v2: combining x and T into x̂ only to try and find violated constraints
#v1: JuMP model for linearized system

## useful functions

function fix_mat(A_JuMP, A_set)
    n = size(A_set)[1]
    for i=1:n, j=1:n; fix(A_JuMP[i,j], A_set[i,j]); end
end

function fix_vec(x_JuMP, x_set)
    n = size(x_set)[1]
    for i=1:n; fix(x_JuMP[i], x_set[i]; force=true); end
end

r_T(T; rθ=rθ) = sum([rθ[:,i]*(T^(i-1)) for i=1:n_deg])
∂r_∂T(T; rθ=rθ) = sum([rθ[:,i+1]*(T^(i-1)) for i=1:(n_deg-1)])
A_T(T; Aθ=Aθ) = sum([Aθ[:,:,i]*(T^(i-1)) for i=1:n_deg])
∂A_∂T(T; Aθ=Aθ) = sum([Aθ[:,:,i+1]*(T^(i-1)) for i=1:(n_deg-1)])

function gLVt_jac(x, T, rθ, Aθ)
    r, A = r_T(T; rθ=rθ), A_T(T; Aθ=Aθ)
    Dxf = Diagonal(r + A*x) + Diagonal(x)*A
    DTf = x.*(∂r_∂T(T; rθ=rθ) + ∂A_∂T(T; Aθ=Aθ)*x)
    DṪ = zeros((1, n+1))
    J = vcat(hcat(Dxf, DTf), DṪ)
    return J
end

function gLVt_ctrl(x, u, T, u_T, rθ, Aθ, B)
    r, A = r_T(T; rθ=rθ), A_T(T; Aθ=Aθ)
    dx = x.*(r + A*x) + B*u
    dT = u_T
    return dx, dT
end

function glvt_rk2(x, u, T, u_T, rθ, Aθ, B, Δt)
    dx, dT = gLVt_ctrl(x, u, T, u_T, rθ, Aθ, B)
    x_int, T_int = max.(x + 0.5*Δt*dx, 0), clamp(T + 0.5*Δt*dT, 25.0, 32.5) # intermediate integration
    dx_int, dT_int = gLVt_ctrl(x_int, u, T_int, u_T, rθ, Aθ, B) # constant input within integ window, but not params!
    x_next, T_next = max.(x + Δt*dx_int, 0), clamp(T + Δt*dT_int, 25.0, 32.5)
    return x_next, T_next
end

## Define Constants

# n = 7
n = 3
n_p = n + n^2
temps = 25:2.5:32.5

# import p_full_all, fit a polynomial and pull coefficients
n_deg = 4
p_file = n == 3 ? "p_full_all_rapkpa_bestish.jld" : "/p_full_all_bestish.jld"
p_full_all = load(to_arkin * p_file, "data")
pθ = zeros((n_p, n_deg))
for i=1:size(p_full_all)[1]
    # f = Polynomials.fit(temps, p_full_all[i,5:end]) #temperature
    f = Polynomials.fit(temps, p_full_all[i,1:4]) #glucose, to see if it makes a huge difference 
    if f[:] == [0.0]
        continue
    end #untrained param
    pθ[i,:] = f[:]
    # println(f[:])
end
rθ = pθ[1:n, :]
Aθ = zeros((n,n, n_deg))
for j=1:n_deg; for i=1:n; Aθ[i,:,j] = pθ[n*i + 1: n*(i+1), j]; end; end
# at some point should do a poly-regression check of predicted fp changes for addnl confirmation (no bifurcation etc.)

# Compute a dictionary of all equilibria
S = unique!(collect(permutations(BitArray([1, 1, 0]))))
fp_d = Dict()
for T ∈ temps
    fps = []
    r, A = r_T(T; rθ=rθ), A_T(T; Aθ=Aθ)
    push!(fps, -inv(A)*r) #full
    for s ∈ S
        fp_s = zeros(n)
        r_s, A_s = r[s], A[s,s]
        fp_s[s] = -inv(A_s)*r_s #sub fixed point
        push!(fps, fp_s)
    end
    get!(fp_d, T, fps)
end

## Define MPC

# n_hrz = 10 # must redefine mpc if changed
n_hrz = 5 #nl
Nonlinear = false

mpc = Model(Ipopt.Optimizer)
set_silent(mpc)
# how to increase ipopt iteration limit?

@variables mpc begin
    #bounds
    x_ub; T_ub; T_lb
    u_ub; u_lb; u_T_lb; u_T_ub

    #states
    X̂[1:(n+1), 1:n_hrz] #collection of states (x & T!)
    Û[1:(n+1), 1:(n_hrz-1)] #collection of inputs (u & u_T!)

    #dynamics
    A[1:(n+1), 1:(n+1)] # linearized gLVt params (x & T!)
    B[1:(n+1), 1:(n+1)] # controllable inputs (u & u_T!)

    #aux
    x̂_diff[1:(n+1), 1:n_hrz] #dummy variable because JuMP's AD is lacking
    û_diff[1:(n+1), 1:n_hrz] #dummy variable because JuMP's AD is lacking

    #resettable design variables
    P[1:(n+1), 1:(n+1)] # terminal state dev weights
    Q[1:(n+1), 1:(n+1)] # running state dev weights
    R[1:(n+1), 1:(n+1)] # running input penalty
    x̂d[1:(n+1)]
end

dead(x) = max(x, 0) # For 0 bounding dynamics (not optimizer constraint)
JuMP.register(mpc, :dead, 1, dead, autodiff=true)

if !Nonlinear
    ### linearized dynamics-FE constraints
    for i=1:n
        @NLconstraint(mpc, [j = 1:(n_hrz-1)], X̂[i,j+1] == dead(sum(A[i,k] * X̂[k,j] + B[i,k] * Û[k,j] for k=1:(n+1)))) #infeasible EVEN WITH u_T=0
    end
    @NLconstraint(mpc, [j = 1:(n_hrz-1)], X̂[n+1,j+1] == sum(A[n+1,k] * X̂[k,j] + B[n+1,k] * Û[k,j] for k=1:(n+1)))

else
    ### nonlinear variables
    @variables mpc begin
        rθ1[1:n]
        rθ2[1:n]
        rθ3[1:n]
        rθ4[1:n]
        Aθ1[1:n, 1:n]
        Aθ2[1:n, 1:n]
        Aθ3[1:n, 1:n]
        Aθ4[1:n, 1:n]
        Δt
    end

    ri(rθ1i, rθ2i, rθ3i, rθ4i, T) = rθ1i * T + rθ2i * T^2 + rθ3i * T^3 + rθ4i * T^4
    JuMP.register(mpc, :ri, 5, ri, autodiff=true)

    Aij(Aθ1ij, Aθ2ij, Aθ3ij, Aθ4ij, T) = Aθ1ij * T + Aθ2ij * T^2 + Aθ3ij * T^3 + Aθ4ij * T^4
    JuMP.register(mpc, :Aij, 5, Aij, autodiff=true)

    ### nonlinear dynamics-FE constraints
    for i=1:n
        @NLconstraint(mpc, [j = 1:(n_hrz-1)], X̂[i,j+1] == dead(X̂[i,j] + Δt *
            (X̂[i,j] * (ri(rθ1[i], rθ2[i], rθ3[i], rθ4[i], X̂[n+1,j]) + sum(Aij(Aθ1[i,k], Aθ2[i,k], Aθ3[i,k], Aθ4[i,k], X̂[n+1,j]) * X̂[k,j] for k=1:n))
                + sum(B[i,k] * Û[k,j] for k=1:(n+1))))) # THIS NEED TO BE Bᶜ!end
    end
end

### state constraints
# @constraint(mpc, X̂[1:n,:] .≥ x_lb) #implied by max-bounded dynamic integration
@constraint(mpc, X̂[1:n,:] .≤ x_ub)
@constraint(mpc, X̂[n+1,:] .≤ T_ub)
@constraint(mpc, X̂[n+1,:] .≥ T_lb)
@constraint(mpc, Û[1:n,:] .≤ u_ub)
@constraint(mpc, Û[1:n,:] .≥ u_lb)
@constraint(mpc, Û[n+1,:] .≤ u_T_ub)
@constraint(mpc, Û[n+1,:] .≥ u_T_lb)

### Objective
# absj(x) = abs(x)
# JuMP.register(mpc, :absj, 1, absj, autodiff=true)
# @constraint(mpc, [k = 1:n_hrz], x̂_diff[:,k] .== absj(X̂[:,k] .- x̂d))
# @constraint(mpc, [k = 1:(n_hrz-1)], û_diff[:,k] .== absj(Û[:,k]))
@constraint(mpc, [k = 1:n_hrz], x̂_diff[:,k] .== (X̂[:,k] .- x̂d).^2)
@constraint(mpc, [k = 1:(n_hrz-1)], û_diff[:,k] .== (Û[:,k]).^2) # input penalty
# @constraint(mpc, [k = 2:(n_hrz-1)], û_diff[:,k] .== (Û[:,k] - Û[:,k-1]).^2) # input change penalty
@NLobjective(mpc, Min, sum(x̂_diff[i,n_hrz] * sum(P[i,j] * x̂_diff[j,n_hrz] for j=1:(n+1)) for i=1:(n+1))
                     + sum(sum(x̂_diff[i,k] * sum(Q[i,j] * x̂_diff[j,k]     for j=1:(n+1)) for i=1:(n+1))
                         + sum(û_diff[i,k] * sum(R[i,j] * û_diff[j,k]     for j=1:(n+1)) for i=1:(n+1))
                           for k=1:(n_hrz-1)))

### Terminal constraints ???
# terminal_x = @constraint ...
# terminal_T = @constraint ...

## 

function run_mpc(mpc, gen, ini, trgt, dmats, bds; rθ=rθ, Aθ=Aθ, ricatti=false, Nonlinear=false, d_path = zeros(1,1))
    # println("Drivng gLV system... ")
    # tick()

    # Initialize
    (n, n_steps, Δtₒ) = gen
    (x0, T0) = ini
    (xd, Td) = trgt
    (Bᶜ, Qₒ, Pₒ, Rₒ) = dmats
    (x_ubₒ, T_ubₒ, T_lbₒ, u_lbₒ, u_ubₒ, u_T_lbₒ, u_T_ubₒ) = bds
    d_path = d_path == zeros(1,1) ? zeros(n, n_steps) : d_path

    # x_pty = 1e2
    # T_pty = 1e-4 * Bᶜ[n+1, n+1]
    # u_pty = 50
    # u_T_pty = 0.0
    # Qₒ = diagm(vcat([x_pty for i=1:n], T_pty))
    # Pₒ = Qₒ
    # Rₒ = diagm(vcat([u_pty for i=1:n], u_T_pty))
    Bᵈ = Δtₒ*Bᶜ;

    # make path arrays
    X̂_path, Û_path, ctrlb_path = zeros((n+1, n_steps)), zeros((n+1, n_steps-1)), zeros(n_steps-1)
    X̂_path[1:n,1], X̂_path[n+1,1] = x0, T0
    status = zeros(n_steps-1)

    # Set mpc design parameters
    x̂dₒ = vcat(xd, Td)
    bds_jump = (x_ub, T_ub, T_lb, u_lb, u_ub, u_T_lb, u_T_ub)

    # Set variables in JuMP model
    if !ricatti
        for (var_j, var) ∈ zip(bds_jump, bds); fix(var_j, var); end
        fix_vec(x̂d, x̂dₒ) # eventually switch Td to terminal constraint?
        fix_mat(P, Pₒ); fix_mat(Q, Qₒ); fix_mat(R, Rₒ)

        if !Nonlinear
            fix_mat(B, Bᵈ)
        else
            fix_vec(rθ1, rθ[:,1]); fix_vec(rθ2, rθ[:,2]); fix_vec(rθ3, rθ[:,3]); fix_vec(rθ4, rθ[:,4])
            fix_mat(Aθ1, Aθ[:,:,1]); fix_mat(Aθ2, Aθ[:,:,2]); fix_mat(Aθ3, Aθ[:,:,3]); fix_mat(Aθ4, Aθ[:,:,4])
            fix(Δt, Δtₒ)
            fix_mat(B, Bᶜ) #if nonlinear fix B to Bᶜ !!!
        end
    end

    # Iterate the MPC in simulation
    for i=1:(n_steps-1)
        # println(" Computing step $i")

        # Set initial conditions
        x̂_curr = X̂_path[:,i]
        if !ricatti; fix_vec(X̂[:,1], x̂_curr); end
        x_curr, T_curr = x̂_curr[1:n], x̂_curr[n+1]

        # Linearize around current point
        if !Nonlinear
            Aᶜ_lin = gLVt_jac(x_curr, T_curr, rθ, Aθ)
            # Aᶜ_lin = gLVt_jac(xd, T_curr, rθ, Aθ) #around xd eq (fails for lCFTOC)
            # Aᶜ_lin = gLVt_jac(xd, Td, rθ, Aθ) #around full eq
            Aᵈ_lin = I + Δtₒ*Aᶜ_lin #FE discretization
            if !ricatti; fix_mat(A, Aᵈ_lin); end

            ctrlb_path[i] = rank(ctrb(Aᶜ_lin, Bᶜ)) == n + Int(Bᶜ[n+1, n+1]) # care only about the gLV controllability when temp ctrl off
        end

        # Compute optimal input with soln to Ricatti, then clamp to u bounds
        if ricatti

            A, B, Q, R = Aᵈ_lin + I*1e-10, Bᵈ + I*1e-6, Qₒ, Rₒ + I*1e-10 #need PD mats, finicky LAPACK/Singular exceptions
            try
                K = dlqr(A, B, Q, R) #need PD mats
            catch e
                println("\nError produced: ", e)
                println("\n A"); for j=1:n+1; println(A[j,1:n+1]); end
                println("\n B"); for j=1:n+1; println(B[j,1:n+1]); end
                println("\n Q"); for j=1:n+1; println(Q[j,1:n+1]); end
                println("\n R"); for j=1:n+1; println(R[j,1:n+1]); end
            end
            K = dlqr(A, B, Q, R)

            # K = dlqr(Aᵈ_lin + I*1e-10, Bᵈ + I*1e-10, Qₒ, Rₒ + I*1e-10) #need PD mats
            û_opt_uc = -K*(x̂_curr .- x̂dₒ)

            # Bound inputs
            û_opt = copy(û_opt_uc)
            for (jx, j) ∈ enumerate(diag(Bᶜ))
                û_opt[jx] = j == 0.0 ? 0.0 : clamp(û_opt_uc[jx], u_lbₒ, u_ubₒ) # correct for numerical reqs of solving dLQR
            end
            û_opt[n+1] = clamp(û_opt_uc[n+1], u_T_lbₒ, u_T_ubₒ)
            # break

        # Compute optimal input by solving constrained finite time optimal control program
        else
            optimize!(mpc)
            # println(termination_status(mpc))
            if termination_status(mpc) ∈ [MOI.INVALID_MODEL]
                println("Fuck!")
                println(termination_status(mpc))
                break
            end
            û_opt = JuMP.value.(Û)[:,1]
            # break

        end

        # Store
        û_opt[abs.(û_opt) .< 1e-5] .= 0 # correct numerical errors
        Û_path[:,i] = û_opt
        u_opt, u_T_opt = û_opt[1:n], û_opt[n+1]

        # Progress true dynamics with u_opt, u_T_opt and disturbance
        x_next, T_next = glvt_rk2(x_curr, u_opt, T_curr, u_T_opt, rθ, Aθ, Bᶜ[1:n,1:n], Δtₒ)
        x_next = max.(x_next .+ d_path[:, i], 0)
        X̂_path[1:n,i+1], X̂_path[n+1,i+1] = x_next, T_next

        # break
    end

    # tock()
    return X̂_path, Û_path, ctrlb_path
end

## Define Controller Settings

# General
n_steps = 60
Δtₒ = 0.1

# Bounds
x_ubₒ = 0.4 # lb implicit in max-bounded dynamics
T_lbₒ, T_ubₒ = 25.0, 32.5
u_lbₒ, u_ubₒ = -0.05, 0.05
u_T_lbₒ, u_T_ubₒ = -10, 10 # corresponds to Δ2.5°/15 min

# Pick controllable species
ctrl_species = [1,2,3];
Bᶜ = zeros((n+1, n+1));
for i ∈ ctrl_species; Bᶜ[i, i] = 1.0; end

# Target
x0 = 0.1*ones(n)
T0 = temps[3] #27.5, 30, 32.5?
xd = [0.1; 0.1; 0.0]
Td = temps[3]

# Design Mats
x_pty = 1e2
u_pty = 50
u_T_pty = 0.0
T_ctrl = 1.0
T_pty = 1e-4 * T_ctrl
Qₒ = diagm(vcat([x_pty for i=1:n], T_pty))
Pₒ = Qₒ
Rₒ = diagm(vcat([u_pty for i=1:n], u_T_pty))


# Package a bunch of params
ini = (x0, T0)
gen = (n, n_steps, Δtₒ)
trgt = (xd, Td)
dmats = (Bᶜ, Qₒ, Pₒ, Rₒ)
bds = (x_ubₒ, T_ubₒ, T_lbₒ, u_lbₒ, u_ubₒ, u_T_lbₒ, u_T_ubₒ)

#test run
tick = Dates.now()
X̂_path, Û_path, ctrlb_path = run_mpc(mpc, gen, ini, trgt, dmats, bds; ricatti=true)
run_time = Dates.value(Dates.now() - tick)/1000 #seconds

## Run and Plot Species vs. Species & T control

# pal = [colors[i] for i in [1,4,6]];
# combs = Array{Plots.Plot{Plots.GRBackend},1}()
# for T_ctrl in [0.0, 1.0]
#
#     # Bᶜ[n+1,n+1] = T_ctrl # temp controllable?
#     Bᶜ[n+1,n+1] = T_ctrl # temp controllable?
#     u_T_lbₒ, u_T_ubₒ = -10*T_ctrl, 10*T_ctrl
#     bds = (x_ubₒ, T_ubₒ, T_lbₒ, u_lbₒ, u_ubₒ, u_T_lbₒ, u_T_ubₒ)
#
#     X̂_path, Û_path, ctrlb_path = run_mpc(mpc, gen, ini, trgt, dmats, bds; ricatti=true)
#     # X̂_path, Û_path, ctrlb_path = run_mpc(mpc, gen, ini, trgt, dmats, bds; Nonlinear=true) #ctrlb path doesn't make sense
#
#     # Plot species, temp, species input, temp input
#     pl_x = plot(Δtₒ:Δtₒ:n_steps*Δtₒ, X̂_path[1:n,:]',
#                 # ylims=[0, 0.25],
#                 palette=pal, legend=false,
#                 title=[L"Species\:Control", L"Species\:and\:Temperature\:Control"][Int(T_ctrl + 1)],
#                 titlefontsize=10,
#                 ytickfontsize=5,
#                 ylims=[0.0, 0.12],
#                 xticks=(collect(0:2.0:n_steps*Δtₒ), []), xaxis=false)
#     if T_ctrl != 1.0; ylabel!(L"x\:(Abundance)"); end
#
#     pl_T = plot(Δtₒ:Δtₒ:n_steps*Δtₒ, X̂_path[n+1,:],
#                 legend=false,
#                 # title=L"Temperature", titlefontsize=10,
#                 ytickfontsize=5,
#                 xticks=(collect(0:2.0:n_steps*Δtₒ), []), xaxis=false)
#     if T_ctrl != 1.0; ylabel!(L"T\:(C)"); end
#
#     pl_u = plot(Δtₒ:Δtₒ:(n_steps-1)*Δtₒ, Û_path[1:n,:]',
#                 legend=false, palette=pal, line=:stem,
#                 ytickfontsize=5,
#                 xticks=(collect(0:2.0:n_steps*Δtₒ), []), xaxis=false)
#     if T_ctrl != 1.0; ylabel!(L"u"); end
#
#     pl_u_T = plot(Δtₒ:Δtₒ:(n_steps-1)*Δtₒ, Û_path[n+1,:],
#                 legend=false, palette=pal, line=:stem,
#                 ytickfontsize=5,
#                 xticks=collect(0:2.0:n_steps*Δtₒ), xlabel=L"Time\:(hours)")
#     if T_ctrl != 1.0; ylabel!(L"u_{T}"); end
#
#     comb = plot(pl_x, pl_T, pl_u, pl_u_T, layout = @layout([A; B{0.2h}; C{0.1h}; D{0.1h}]), bottom_margin= -15Plots.px)
#
#     println("With "*["Species Control", "Species and Temperature Control"][Int(T_ctrl + 1)]*", Total Required Species Input:")
#     print(round(sum(abs, Û_path[1:n,:]), digits=3))
#     println()
#     push!(combs, comb)
# end
#
# title = plot(title = L"Driving\:gLVt's\:- Linearized\:dLQR\:(Ricatti)", grid = false, axis = false, bordercolor = "white", yticks=nothing, xticks=nothing, bottom_margin = -15Plots.px)
# # title = plot(title = L"Driving\:gLVt's\:- Linearized\:CFTOC", grid = false, axis = false, bordercolor = "white", yticks=nothing, xticks=nothing, bottom_margin = -15Plots.px)
# # title = plot(title = L"Driving\:gLVt's\:- Nonlinear\:CFTOC", grid = false, axis = false, bordercolor = "white", yticks=nothing, xticks=nothing, bottom_margin = -15Plots.px)
#
# spacer = plot(title = "", grid = false, axis = false, bordercolor = "white", yticks=nothing, xticks=nothing)
# plt = plot(combs..., layout=(1,2))
# plt_final = plot(title, plt, spacer, layout = @layout([A{0.025h}; B; C{0.025h}]), dpi=500)
# savefig(plt_final, pwd()*"/Thesis/Write/Paper/figs/MPC_Example_Figure.png")

## Simulate many drives and record data

d_lim = 0.025
n_rand = 100
rand_paths = cat(zeros(n, n_steps, 1), d_lim * 2 .*(rand(n, n_steps, n_rand) .- 0.5); dims=3) #1 no dist + n_rands

ctrl_species_list = collect(powerset([1, 2, 3]))

mpc_data = DataFrame(
    controller = String[],
    temperature = Float64[],
    initial = Array{Float64, 1}[],
    destination = Array{Float64, 1}[],
    temperaturecontrol = Float64[],
    ctrlspecies = Array{Int, 1}[],
    statepath = Array{Float64, 2}[],
    inputpath = Array{Float64, 2}[],
    ctrlbpath = Array{Float64, 1}[],
    dpath = Array{Float64, 2}[],
    runtime =  Float64[] #seconds
)

# 1 controller * 4 temps * 12 paths * 11 (randomizations + no dist) * 2 temp controls * 7 species-ctrl configs = 7392 its
# with ricatti runtime of ~ 0.001 seconds, LFTOC runtime of about ~4 sec

# ordered to minimize comp time over readability, sorry
println("Running Many Iterations of MPC from different Equilibria...")
for T_ctrl ∈ [0.0, 1.0]

    T_pty = 1e-4 * T_ctrl
    Qₒ = diagm(vcat([x_pty for i=1:n], T_pty))
    Pₒ = Qₒ
    Rₒ = diagm(vcat([u_pty for i=1:n], u_T_pty))
    u_T_lbₒ, u_T_ubₒ = -10*T_ctrl, 10*T_ctrl
    bds = (x_ubₒ, T_ubₒ, T_lbₒ, u_lbₒ, u_ubₒ, u_T_lbₒ, u_T_ubₒ)

    for ctrl_species ∈ ctrl_species_list[2:end]

        Bᶜ = zeros((n+1, n+1));
        for i ∈ ctrl_species; Bᶜ[i, i] = 1.0; end
        Bᶜ[n+1,n+1] = T_ctrl
        dmats = (Bᶜ, Qₒ, Pₒ, Rₒ)

        for rand = 1:size(rand_paths)[3]

            d_path = rand_paths[:, :, rand]

            for ctrlr ∈ ["Ricatti"] #["Ricatti", "LCFTOC"] # clamped ricatti works faster and much better

                ricatti = ctrlr == "Ricatti" ? true : false

                for T ∈ temps

                    for (x0, xd) in collect(permutations(fp_d[T], 2))

                        x0 = max.(x0, 0) #can't start at a negative equilibria
                        ini = (x0, T)
                        trgt = (xd, T)

                        tick = Dates.now()
                        X̂_path, Û_path, ctrlb_path = run_mpc(mpc, gen, ini, trgt, dmats, bds; ricatti=ricatti, d_path=d_path)
                        run_time = Dates.value(Dates.now() - tick)/1000

                        push!(mpc_data, (ctrlr, T, x0, xd, T_ctrl, ctrl_species, X̂_path, Û_path, ctrlb_path, d_path, run_time))

                        # break
                    end
                end
                # break
            end
            # break
        end
        # break
    end
end
println(" Finished. Making Plots now.")

# serialize("mpc_data_ricatti_allcfig_hrzn"*"$n_hrz"*"_rand100.jls", mpc_data)

## Comute Statistics
#
mpc_data = deserialize(to_arkin * "mpc_data_ricatti_allctrlconfig_hrzn"*"$n_hrz"*".jls")
#
#
# negative_state_test = select(mpc_data,
#             :statepath => ByRow(x -> any(x[:, 1] .< 0) ? 1 : 0) => :initial,
#             :statepath => ByRow(x -> any(x[:, 2:n_steps-1] .< 0) ? 1 : 0) => :intermediate,
#             :statepath => ByRow(x -> any(x[:, end] .< 0) ? 1 : 0) => :final
#             )
#
mpc_summary = select(mpc_data,
            :controller,
            :temperaturecontrol,
            :temperature,
            :initial,
            :destination,
            :ctrlspecies,
            :dpath => ByRow(x -> all(x .== 0) ? 0 : 1) => :disturbed,
            # AsTable([:destination, :statepath]) => ByRow(x,y -> sqrt(sum(abs, x .- y[1:n, end]))) => :finalspeciesdistance,
            # AsTable([:destination, :statepath]) => ByRow((x,y)-> sqrt(sum(abs, x .- y[1:n, end]))) => :finalspeciesdistance,
            [:destination, :statepath] => ((x,y) -> [sqrt(sum(abs2, x[i] .- y[i][1:n, end])) for i in axes(x,1)]) => :finalspeciesdistance,
            :inputpath => ByRow(x -> sum(abs, x[1:n, :])) => :totalspeciesinput,
            :inputpath => ByRow(x -> sum(abs, x[n+1, :])) => :totaltemperatureinput,
            :statepath => ByRow(x -> sum(x[n+1, :])/length(x[n+1, :])) => :meantemperature,
            :statepath => ByRow(x -> x[n+1, end]) => :finaltemperature,
            :ctrlbpath => ByRow(x -> [sum(x[i:i+10]) for i=1:length(x)-10]) => :ctrlbsma10
            )

grouped = groupby(mpc_summary, [:controller, :temperaturecontrol, :temperature, :initial, :destination, :disturbed, :ctrlspecies])
mpc_summary_stats = combine(grouped, [:finalspeciesdistance, :totalspeciesinput, :totaltemperatureinput, :meantemperature, :finaltemperature] .=> mean)

# serialize("mpc_summary_ricatti_allctrlconf_hrzn"*"$n_hrz"*"_rand100_bestish_G.jls", mpc_summary_stats)

## Plot Controllability of Linearized System

# disturbed = 0.
# for controller ∈ ["Ricatti"], tc ∈ [0., 1.], T ∈ temps
#     # c_plot = plot(size = (450,600))
#     println("\n\nTemp $T,"*" with tc? $tc")
#
#     for (csi, cs) ∈ enumerate(ctrl_species_list[2:end])
#         color = colors[csi]
#
#         df_temp = mpc_summary[(mpc_summary.controller .== controller) .&
#                                       (mpc_summary.temperature .== T)           .&
#                                       (mpc_summary.temperaturecontrol .== tc)   .&
#                                       (mpc_summary.ctrlspecies .== [cs])        .&
#                                       (mpc_summary.disturbed .== disturbed), :]
#
#         for (ir, row) in enumerate(eachrow(df_temp)) #iterate thru paths
#             label = ir == 1 ? "Ctrl of species $cs" : ""
#
#             if ir == 1;
#                 if sum(row.ctrlbsma10) == 49*11
#                     controllable = " always"
#                 elseif sum(row.ctrlbsma10) == 0
#                     controllable = " never"
#                 else
#                     controllable = " sometimes"
#                 end
#                 println("ctrl w species $cs is"*controllable*" controllable");
#             end
#
#             # plot!(1:length(row.ctrlbsma10), row.ctrlbsma10;
#             #                         # yerror=df_temp.finalspeciesdistance_std,
#             #                         label=label,
#             #                         color=color,
#             #                         # alpha=0.2,
#             #                         # linestyle=linestyle,
#             #                         # shape=:circle,
#             #                         linewidth=2)
#         end
#     end
#     # break
#     ylabel!("Controllability 1-hour SMA of Current Linearization")
#     xlabel!("Time points")
#     title!(" Temp $T, "*" with tc? $tc")
#     # display(c_plot)
# end

# plot!(legend=false)
# plot!(legend=:topleft)

## Make Summary Figures (Path-Averaged)

grouped_wop = groupby(mpc_summary, [:controller, :temperaturecontrol, :temperature, :ctrlspecies, :disturbed])
mpc_summary_wop_stats = combine(grouped_wop,
                                [:finalspeciesdistance, :totalspeciesinput, :totaltemperatureinput, :meantemperature, :finaltemperature] .=> mean,
                                [:finalspeciesdistance, :totalspeciesinput, :totaltemperatureinput, :meantemperature, :finaltemperature] .=> std)
disturbed = 1.
colors = palette(:default)

fd_pl = plot(size = (450,600))
for controller ∈ ["Ricatti"], (csi, cs) ∈ enumerate(ctrl_species_list[2:end]), tc ∈ [0., 1.]
    df_temp = mpc_summary_wop_stats[(mpc_summary_wop_stats.controller .== controller) .&
                                  (mpc_summary_wop_stats.temperaturecontrol .== tc)   .&
                                  (mpc_summary_wop_stats.ctrlspecies .== [cs])        .&
                                  (mpc_summary_wop_stats.disturbed .== disturbed), :]

    # label = "Ctrl of species $cs with " * ["no ΔT", "ΔT"][Int(tc)+1]
    cs_label = string((["RA", "PK", "PA"][cs] .* ", ")...)[1:end-2]
    label = ["Ctrl of species " * cs_label, ""][Int(tc)+1]
    linestyle = [:solid, :dot][Int(tc)+1]
    # color = controller == "Ricatti" ? colorant"indigo" : colorant"gold"
    color = colors[csi]
    plot!(df_temp.temperature, df_temp.finalspeciesdistance_mean;
                            # yerror=df_temp.finalspeciesdistance_std,
                            label=label, color=color,
                            linestyle=linestyle,
                            shape=:circle,
                            linewidth=2)
end
# xlabel!("Initial Temperature")
for i=1:2
    label = ["no ΔT", "ΔT"][Int(i)]
    linestyle = [:solid, :dot][Int(i)]
    plot!([], []; label=label, color=:black, linestyle=linestyle, shape=:circle, linewidth=2)
end
ylabel!("Final Distance to Target")
xticks!(temps)
plot!(legend=:topleft)

tsi_pl = plot(size = (450,600))
for controller ∈ ["Ricatti"], (csi, cs) ∈ enumerate(ctrl_species_list[2:end]), tc ∈ [0., 1.]
    df_temp = mpc_summary_wop_stats[(mpc_summary_wop_stats.controller .== controller) .&
                                  (mpc_summary_wop_stats.temperaturecontrol .== tc)   .&
                                  (mpc_summary_wop_stats.ctrlspecies .== [cs])        .&
                                  (mpc_summary_wop_stats.disturbed .== disturbed), :]

    label = "Ctrl of species $cs with " * ["no ΔT", "ΔT"][Int(tc)+1]
    linestyle = [:solid, :dot][Int(tc)+1]
    # color = controller == "Ricatti" ? colorant"indigo" : colorant"gold"
    color = colors[csi]
    plot!(df_temp.temperature, df_temp.totalspeciesinput_mean./length(cs);
                            label=label,
                            color=color,
                            linestyle=linestyle,
                            shape=:circle,
                            linewidth=2)
end
# xlabel!("Initial Temperature")
ylabel!("Normalized Total Species Input")
plot!(legend=false)

ft_pl = plot()
plot!(temps, temps, color=colorant"black", label="")
for controller ∈ ["Ricatti"], (csi, cs) ∈ enumerate(ctrl_species_list[2:end]), tc ∈ [1.]
    df_temp = mpc_summary_wop_stats[(mpc_summary_wop_stats.controller .== controller) .&
                                  (mpc_summary_wop_stats.temperaturecontrol .== tc)   .&
                                  (mpc_summary_wop_stats.ctrlspecies .== [cs])        .&
                                  (mpc_summary_wop_stats.disturbed .== disturbed), :]

    label = controller * " with " * ["no ΔT", "ΔT"][Int(tc)+1]
    linestyle = [:solid, :dot][Int(tc)+1]
    color = colors[csi]
    plot!(df_temp.temperature, df_temp.finaltemperature_mean; label=label, color=color, linestyle=linestyle, shape=:circle, linewidth=2)
end
xlabel!("Initial Temperature")
ylabel!("Final Temperature")
plot!(legend=false)

mt_pl = plot()
plot!(temps, temps, color=colorant"black", label="")
for controller ∈ ["Ricatti"], (csi, cs) ∈ enumerate(ctrl_species_list[2:end]), tc ∈ [1.]
    df_temp = mpc_summary_wop_stats[(mpc_summary_wop_stats.controller .== controller) .&
                                  (mpc_summary_wop_stats.temperaturecontrol .== tc)   .&
                                  (mpc_summary_wop_stats.ctrlspecies .== [cs])        .&
                                  (mpc_summary_wop_stats.disturbed .== disturbed), :]

    label = controller * " with " * ["no ΔT", "ΔT"][Int(tc)+1]
    linestyle = [:solid, :dot][Int(tc)+1]
    color = colors[csi]
    plot!(df_temp.temperature, df_temp.meantemperature_mean; label=label, color=color, linestyle=linestyle, shape=:circle, linewidth=2)
end
xlabel!("Initial Temperature")
ylabel!("Mean Temperature")
plot!(legend=false)

mpc_overall_fig = plot(fd_pl, tsi_pl, mt_pl, layout = @layout([A B C]), size = (1200, 400), bottom_margin = 20Plots.px, left_margin = 20Plots.px, top_margin = 10Plots.px, dpi=500)
xticks!(temps)

# mpc_overall_fig = plot(fd_pl, tsi_pl, layout = @layout([A B]), size = (850, 600), bottom_margin = 20Plots.px, left_margin = 20Plots.px, top_margin = 10Plots.px, dpi=500)
xlabel!("Initial Temperature")
# xticks!(temps)

# savefig(mpc_overall_fig, "~/Documents/Thesis/Write/Paper/figs/MPC_Overall_Figure_allconfig.png")


## Make Path Figures (Input-Averaged)

grouped_wop = groupby(mpc_summary, [:controller, :temperaturecontrol, :temperature, :initial, :destination, :disturbed])
mpc_summary_wop_stats = combine(grouped_wop,
                                [:finalspeciesdistance, :totalspeciesinput, :totaltemperatureinput, :meantemperature, :finaltemperature] .=> mean,
                                [:finalspeciesdistance, :totalspeciesinput, :totaltemperatureinput, :meantemperature, :finaltemperature] .=> std)
disturbed = 1.
colors = palette(:default)

fd_pl = plot(size = (450,600))
for controller ∈ ["Ricatti"], pthi = 1:12, tc ∈ [0., 1.]
    df_temp_all_path = mpc_summary_wop_stats[(mpc_summary_wop_stats.controller .== controller) .&
                                  (mpc_summary_wop_stats.temperaturecontrol .== tc)   .&
                                  (mpc_summary_wop_stats.disturbed .== disturbed), :]
    df_temp = df_temp_all_path[pthi:12:end, :]

    label = ["Path $pthi", ""][Int(tc)+1]
    linestyle = [:solid, :dot][Int(tc)+1]
    color = colors[pthi]

    plot!(df_temp.temperature, df_temp.finalspeciesdistance_mean;
                            # yerror=df_temp.finalspeciesdistance_std,
                            label=label, color=color,
                            linestyle=linestyle,
                            shape=:circle,
                            linewidth=2)
end
# xlabel!("Initial Temperature")
for i=1:2
    label = ["no ΔT", "ΔT"][Int(i)]
    linestyle = [:solid, :dot][Int(i)]
    plot!([], []; label=label, color=:black, linestyle=linestyle, shape=:circle, linewidth=2)
end
ylabel!("Final Distance to Target")
xticks!(temps)
plot!(legend=:topleft)

tsi_pl = plot(size = (450,600))
for controller ∈ ["Ricatti"], pthi = 1:12, tc ∈ [0., 1.]
    df_temp_all_path = mpc_summary_wop_stats[(mpc_summary_wop_stats.controller .== controller) .&
                                  (mpc_summary_wop_stats.temperaturecontrol .== tc)   .&
                                  (mpc_summary_wop_stats.disturbed .== disturbed), :]
    df_temp = df_temp_all_path[pthi:12:end, :]

    label = ["Path $pthi", ""][Int(tc)+1]
    linestyle = [:solid, :dot][Int(tc)+1]
    color = colors[pthi]

    plot!(df_temp.temperature, df_temp.totalspeciesinput_mean;
                            label=label,
                            color=color,
                            linestyle=linestyle,
                            shape=:circle,
                            linewidth=2)
end
# xlabel!("Initial Temperature")
ylabel!("Normalized Total Species Input")
plot!(legend=false)

ft_pl = plot()
plot!(temps, temps, color=colorant"black", label="")
for controller ∈ ["Ricatti"], pthi = 1:12, tc ∈ [1.]
    df_temp_all_path = mpc_summary_wop_stats[(mpc_summary_wop_stats.controller .== controller) .&
                                  (mpc_summary_wop_stats.temperaturecontrol .== tc)   .&
                                  (mpc_summary_wop_stats.disturbed .== disturbed), :]
    df_temp = df_temp_all_path[pthi:12:end, :]

    label = ["Path $pthi", ""][Int(tc)+1]
    linestyle = [:solid, :dot][Int(tc)+1]
    color = colors[pthi]
    plot!(df_temp.temperature, df_temp.finaltemperature_mean; label=label, color=color, linestyle=linestyle, shape=:circle, linewidth=2)
end
xlabel!("Initial Temperature")
ylabel!("Final Temperature")
plot!(legend=false)

mt_pl = plot()
plot!(temps, temps, color=colorant"black", label="")
for controller ∈ ["Ricatti"], pthi = 1:12, tc ∈ [1.]
    df_temp_all_path = mpc_summary_wop_stats[(mpc_summary_wop_stats.controller .== controller) .&
                                  (mpc_summary_wop_stats.temperaturecontrol .== tc)   .&
                                  (mpc_summary_wop_stats.disturbed .== disturbed), :]
    df_temp = df_temp_all_path[pthi:12:end, :]

    label = ["Path $pthi", ""][Int(tc)+1]
    linestyle = [:solid, :dot][Int(tc)+1]
    color = colors[pthi]
    plot!(df_temp.temperature, df_temp.meantemperature_mean; label=label, color=color, linestyle=linestyle, shape=:circle, linewidth=2)
end
xlabel!("Initial Temperature")
ylabel!("Mean Temperature")
plot!(legend=false)

mpc_overall_fig = plot(fd_pl, tsi_pl, mt_pl, layout = @layout([A B C]), size = (1200, 400), bottom_margin = 20Plots.px, left_margin = 20Plots.px, top_margin = 10Plots.px, dpi=500)
xticks!(temps)

# mpc_overall_fig = plot(fd_pl, tsi_pl, layout = @layout([A B]), size = (850, 600), bottom_margin = 20Plots.px, left_margin = 20Plots.px, top_margin = 10Plots.px, dpi=500)
xlabel!("Initial Temperature")
# xticks!(temps)

#
# savefig(mpc_overall_fig, "~/Documents/Thesis/Write/Paper/figs/MPC_Overall_Figure_allpaths.png")
#

## Make Network Figure

disturbed = 1.
combined_df = mpc_summary_stats[(mpc_summary_stats.controller .== "Ricatti") .&
                                (mpc_summary_stats.disturbed .== disturbed), :]
combined_fsd_mean = combined_df.finalspeciesdistance_mean
combined_tsi_mean = combined_df.totalspeciesinput_mean

for controller ∈ ["Ricatti"] #["Ricatti", "LCFTOC"]
    global combined_fsd_mean, combined_tsi_mean

    df = copy(mpc_summary_stats[(mpc_summary_stats.controller .== controller) .&
                                (mpc_summary_stats.disturbed .== disturbed), :])

    # histogram(df.finalspeciesdistance_mean)

    # over all temperatures performance
    # for controller ∈ ["Ricatti", "LCFTOC"]
    #     for disturbed ∈ [0, 1]
    #         df = copy(mpc_summary_stats[(mpc_summary_stats.controller .== controller) .&
    #                                     (mpc_summary_stats.disturbed .== disturbed), :])
    #         if controller == "Ricatti" && disturbed == 0
    #             histogram(df.finalspeciesdistance_mean, bins = 0.0:0.025:0.6, label=controller * [" not disturbed", " disturbed"][disturbed+1])
    #         else
    #             histogram!(df.finalspeciesdistance_mean, bins = 0.0:0.025:0.6, label=controller * [" not disturbed", " disturbed"][disturbed+1])
    #         end
    #     end
    # end

    orgs = ["Ra", "Pk", "Pa"]

    cmap = colormap("RdBu", 100)
    # cmap = range(HSL(colorant"red"), stop=HSL(colorant"green"), length=100)
    # cmap = cgrad(:thermal, 100)
    scale_ix(val, min, max; bins=100) = Int(floor(bins*(val - min)/(max + 0.001 - min))) + 1

    n_eq = length(fp_d[temps[1]])
    locs_x = [0; 2*√3; √3; √3]; locs_y = [0., 0., 1., 3.]
    G = DiGraph(n_eq) # one node per eq
    for i=1:n_eq, j=1:n_eq
        if i != j
            add_edge!(G, i, j)
        end
    end

    eqs_l = vcat([BitArray([1, 1, 1])], S)
    paths_l = collect(permutations(eqs_l, 2))
    eq_to_node = Dict([(eqs_l[2], 1), (eqs_l[4], 2), (eqs_l[1], 3), (eqs_l[3], 4)])
    paths_node = map(x -> [eq_to_node[x[1]], eq_to_node[x[2]]] , paths_l)

    function gplot_mpc_stats(df_temp, T)

         # total input used will be the weight of arrow
         # color of the arrow is finalspeciesdistance
        elw = []; ec = [];
        for i=1:n_eq, j=1:n_eq
            if i != j
                # add_edge!(G, i, j)
                path_ix = findall(x -> x[1] == i && x[2] == j, paths_node)
                path_data = df_temp[path_ix, :]
                push!(elw, path_data.totalspeciesinput_mean[1]) # pull total input from table
                # push!(ec, cmap[scale_ix(-path_data.finalspeciesdistance_mean[1], -maximum(df.finalspeciesdistance_mean), -minimum(df.finalspeciesdistance_mean))]) # pull distance from norm
                # push!(ec, cmap[scale_ix(-path_data.finalspeciesdistance_mean[1], -maximum(combined_fsd_mean), -minimum(combined_fsd_mean))]) # pull distance from norm
                push!(ec, cmap[scale_ix(-path_data.finalspeciesdistance_mean[1], -0.2, 0.0)]) # pull distance from norm
            end
        end

        if T != 32.5 # Labeling Stable Fixed Point
            nodestrokec = [colorant"black", colorant"black", colorant"green", colorant"black"]
        else
            nodestrokec = [colorant"black", colorant"black", colorant"black", colorant"green"]
        end

        graph_plot = gplot(G, locs_x, locs_y;
                        # layout=circular_layout,
                        # layout=layout,
                        nodelabel=["RaPk", "PkPa", "RaPkPa", "RaPa"],
                        # nodelabel=["RA-PK", "PK-PA", "RA-PK-PA", "RA-PA"],
                        NODELABELSIZE = 6.0,
                        nodestrokec = nodestrokec,
                        nodestrokelw = 0.01,
                        nodefillc = colorant"white",
                        NODESIZE=[0.18 for c=1:n_eq],
                        # NODESIZE=0.2*[0.1, 0.01, 0.01, 0.27, 0.01, 0.7, 0.01], #ra sized
                        # EDGELINEWIDTH=3*maximum(elw)/maximum(df.totalspeciesinput_mean), #max width
                        EDGELINEWIDTH=8*maximum(elw)/maximum(combined_tsi_mean), #max width
                        edgelinewidth=elw, #relative widths
                        edgestrokec=ec,
                        linetype="curve",
                        arrowlengthfrac=0.1,
                        outangle=0.08
                        )

        return graph_plot
    end

    # tc = 0.
    # T = 25.0
    #
    # gp = gplot_mpc_stats(tc, T)
    # display(gp)
    #
    # tc = 1.
    #
    # gp = gplot_mpc_stats(tc, T)
    # display(gp)

    for tc ∈ [0., 1.]
        for T ∈ temps
            df_temp = df[(df.temperaturecontrol .== tc) .&
                         (df.temperature .== T), :]
            gp = gplot_mpc_stats(df_temp, T)
            draw(PNG("mpc_graph_"*controller*"_"*["notc","tc"][Int(tc)+1]*"_$T"*"_bestish_G.png", 16cm, 16cm, dpi=1000), gp)
        end
    end

    T_plots = Array{Plots.Plot{Plots.GRBackend},1}();
    for tc ∈ [0., 1.], T ∈ temps
        fname = "mpc_graph_"*controller*"_"*["notc","tc"][Int(tc)+1]*"_$T"*"_bestish_G.png"
        pl = plot(load(fname), annotate=(5000,6000,(string(T)*"°",:left,10)), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing)
        # display(pl)
        push!(T_plots, pl)
    end

    s_ctrl_plot = plot(T_plots[1:4]..., layout=@layout([A B C D]), size = (1000, 600), dpi=500) #, bottom_margin = -100Plots.px, top_margin = -100Plots.px)
    st_ctrl_plot = plot(T_plots[5:8]..., layout=@layout([A B C D]), size = (1000, 600), dpi=500) #, bottom_margin = -100Plots.px, top_margin = -100Plots.px)
    s_tag = plot(annotate= (0, 0, (L"With\:Species\:Control\:Only", :left, 15)), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing) #, bottom_margin = -25Plots.px)
    st_tag = plot(annotate= (0, 0, (L"With\:Species\:&\:Temperature\:Control", :left, 15)), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing) #, bottom_margin = -25Plots.px)

    # title = plot(title=controller*[", not disturbed", ", disturbed"][Int(disturbed)+1], grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, bottom_margin = -25Plots.px)

    # h2 = scatter([0,0], [0,1], zcolor=[0,3], clims=(-0.2, 0.0), colorbar_ticks=[0.2, 0.0],
    #                     xlims=(1,1.1), label="", colorbar_title="Final Species Distance", c=cgrad(:RdBu), framestyle=:none) #, colorbar_title="Final Species Distance"
    #
    # @show h2[1][:xaxis][:showaxis]
    # @show h2[1][:yaxis][:showaxis]

    # scatter([0,0], [0,1], zcolor=[0,3], clims=clims,
    #          xlims=(1,1.1), xshowaxis=false, yshowaxis=false, label="", c=:viridis, colorbar_title="cbar", grid=false)

    # full_plt = plot(title, s_tag, s_ctrl_plot, st_tag, st_ctrl_plot, layout = @layout([A{0.025h}; B{0.025h}; C; D{0.025h}; E]), top_margin = 30Plots.px, dpi=500) #, bottom_margin = -25Plots.px)
    full_plt = plot(s_tag, s_ctrl_plot, st_tag, st_ctrl_plot, layout = @layout([A{0.025h}; B; C{0.025h}; D]), dpi=500) #, top_margin = 30Plots.px) #, bottom_margin = -25Plots.px)

    # full_plt_cbar = plot(full_plt, h2, layout = @layout([A B{0.035w}])) #, bottom_margin = -25Plots.px)
    savefig(full_plt, "~/Documents/Thesis/Write/Paper/figs/MPC_Paths_Figure_"*controller*"_hq_s.png")

    display(full_plt)
    # break
end
