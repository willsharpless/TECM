### Parameterization fo Parameter-Varying gLV Systems
# willsharpless@berkeley.edu or ucsd.edu
# Apr 28, 2021

print("\n\nParameterization of Environment-Varying gLV Systems\n")
print("willsharpless@berkeley.edu or ucsd.edu\n")
print("created: Apr 28, 2021\n")

### Version log
# 7 (08/23/21):
#   - Incorporating isogenic growth curves
#   - ¿ > max_abu inf penalty ?
#   - ¿ have to adjust loss somehow cuz more iso timepoints ==> higher penalty ? normalize ?
#       - only using 7 tp from isogenic experiments to balance, although all tp_iso ∈ [0,48]
#   - ¿ should I remove the subcomm weight ? which is encouraging focus on 3 > 2 > 1
#       - yes probably
# 6.4.2: abs(alpha) <= 2.0
# 6.4.1: bounded lower value of r to be 0.25 (min_iso_od * min_alpha_self = min_r)
# 6.2.1-4: wander weight variation
# 6.2 - Going back to single loss because of borg morea bugs
# 6.1 - changed a_ii (self) bounds [-8,8] -> [-8,0]
# 6 (06/17/21)
#   - MultiObjective Cost for all but first optim to encourage similar params
#   - Corrected bbo to only breed reduced parameter set with experiment relevance
#   - Removed unnecessary broadcasting in glv
# 5.2.2 (06/14/21)
#   - Added glv jacobian & changed solver to CVODE_BDF to improve integration
#   - turned bbo multithreading on (OFF - not working)
#   - added Suppresor to hide & capture warnings
# 5.2 - changed r bounds [-1,1] -> [0,1]
# 5.1 - weighted experiments in loss by (2^n_sub)
# 5 - sci_ml.train -> bboptimize !!!
# 4 - Serial training of experiments -> Parallel experiment scoring in mega loss()
# 3 - wrote pixs to find parameters for a given subcomm in the full comm

## Load & Precompile Modules

using Dates
print("run: " * split(string(Dates.now()),'T')[1] * ", " * split(string(Dates.now()),'T')[2] * "\n")
print("\nprecompiling packages...\n")

# Define paths based on execution environment
if split(pwd(),'/')[2] == "Users" # local cpu
    cd("/Users/willsharpless/")
    prepath = "/Documents/"
    od_path = pwd() * prepath * "Thesis/pairwise_second/od_readings/od_"
    ra_path = pwd() * prepath * "ipy/Arkinlab/SynComm_Analysis/RA_Data/"
    iso_path = pwd() * prepath * "Thesis/Data/Isogenic_curves/iso_"
    to_arkin = pwd() * prepath * "Julia/ECM/"
else # arkin server
    prepath = "/auto/sahara/namib/home/willsharpless/Julia/arkin/"
    od_path = prepath * "od_readings/od_"
    ra_path = prepath * "RA_Data/"
    iso_path = prepath * "Isogenic_curves/iso_"
    to_arkin = prepath
end

# General Math
using Plots, Random, StatsBase, Statistics, LinearAlgebra

# Diff Eq / Parameterization
using DifferentialEquations, Sundials
using DiffEqFlux, DiffEqSensitivity, Enzyme
using Optim, GalacticOptim
using BlackBoxOptim
using BlackBoxOptim: num_func_evals

# Network Analysis / Control
# using LightGraphs, Hungarian, Missings, MatrixNetworks, GraphPlot
# using ControlSystems

# Array Handling & IO
using SparseArrays, JLD, PyCall, ProgressMeter, Suppressor

# Custom modules
# external_path = "Julia/external/DriverSpecies-master/"
# my_modules_path = "Julia/modules/"
# push!(LOAD_PATH, pwd() * prepath * external_path)
# push!(LOAD_PATH, pwd() * prepath * my_modules_path)
# using DriverSpeciesModule # Angulo package
# using Glv: p_loader, pixs # My gLV-related helper functions

## Init

### Model Definition
n = 7
pair_interval = 24; iso_interval = 8 #hours
pair_tspan = (0.0, 120.0); iso_tspan = (0.0, 48.0);
psg_t = [24.0, 72.0] # only for pairs
max_abu = 2
x0 = zeros(n) # only for definition

# 0 bounding all states
dead(u,t,integrator) = any(u .< 1e-8) #|| any(u .> 10)
function die!(integrator); integrator.u[integrator.u .< 1e-12] .= 0; end
death_cb = ContinuousCallback(dead, die!, save_positions = (false,false))

# simulating passages
psg_time(u,t,integrator) = t ∈ psg_t
function psg!(integrator); integrator.u .*= 0.05; end # 1:20 culture dilution
psg_cb = DiscreteCallback(psg_time, psg!, save_positions = (false,false))
cbset = CallbackSet(death_cb, psg_cb)

# glv structure for handling subsets of parameters
struct p_loader{Float64, Bool}
    p_load :: Array{Float64,1}
    p_load_ix :: Array{Bool,1} #refers to subcomm load loc
    p_train_ix :: Array{Bool,1} #refers to p_full extraction loc
end

# glv evaluation
function (gLV_somep :: p_loader)(dx, x, p_full, t)
    # combine loaded and optimizing params
    n = length(x)
    p_r_A_flat = Array{eltype(x),1}(undef, n*(n+1))
    p_r_A_flat[gLV_somep.p_load_ix] = gLV_somep.p_load
    p_r_A_flat[.!gLV_somep.p_load_ix] = p_full[gLV_somep.p_train_ix]

    # unpack params
    r = p_r_A_flat[1:n];
    A = Array{eltype(x),2}(undef,n,n)
    for i=1:n; A[i,:] = p_r_A_flat[n*i + 1: n*(i+1)]; end

    dx .= x.*(r + A*x)
    if any(x .> max_abu); dx .= 0; end # no cc catch
end

# Direct jacobian aides integrators (efficiency + stiff-handling)
function gLV_jac(J, x, p, t)
    n = size(x)[1]
    r = p[1:n];
    A = Array{eltype(x),2}(undef,n,n)
    for i=1:n; A[i,:] = p[n*i + 1: n*(i+1)]; end
    J = Diagonal(r + A*x) + Diagonal(x)*A
    nothing
end

### Optimization Definition
function pixs(n_tot, dims_load, dims_train)
    # dims_load: list of lists of trained subcomms eg. [[1,3,5], [6,5,4]] or [[1,2]]
    # dims_train: list of orgs to add to combined subcomm

    udl = unique(vcat(dims_load...))
    dims_sub = unique(vcat(udl,dims_train))
    n_sub = length(dims_sub)

    # Outputs: flat logicals for finding params
    ptix_full = BitArray(undef, n_tot*(n_tot+1)); plix_full = BitArray(undef, n_tot*(n_tot+1));
    plix_subcomm = BitArray(undef, n_sub*(n_sub+1)) #(ptix_subcomm = .!plix_subcomm)

    # useful non-flat logicals
    A_full_log = BitArray(undef, n_tot, n_tot)
    A_sub_log = BitArray(undef, n_sub, n_sub)

    # Find Loaded Parameters in Full Community
    plix_full[udl] .= 1 # find r's
    for s in dims_load
        A_full_log[vec([CartesianIndex(i,j) for i in s for j in s])] .= 1 #find pairs in trained subcomms
    end
    for i=1:n_tot; plix_full[n_tot*i + 1: n_tot*(i+1)] = A_full_log[i,:]; end

    # Find Loaded Param in Subcommunity (make loaded dims go first)
    plix_subcomm[1:length(udl)] .= 1
    A_sub_log = A_full_log[vec(sort(dims_sub)), vec(sort(dims_sub))]
    for i=1:n_sub; plix_subcomm[n_sub*i + 1: n_sub*(i+1)] = A_sub_log[i,:]; end
    A_full_log .= 0 #reset

    # Find Training Parameters in Full Community
    ptix_full[dims_train] .= 1 # find r's
    A_full_log[vec([CartesianIndex(i,j) for i in dims_sub for j in dims_sub])] .=1
    for i=1:n_tot; ptix_full[n_tot*i + 1: n_tot*(i+1)] = A_full_log[i,:]; end
    ptix_full[plix_full] .= 0 # remove loaded pairs

    return plix_subcomm, plix_full, ptix_full
end

# Direct Global Loss: sum(Lp(time point error))
function sum_pnorm_error(predicted, actual, pn)
    return sum(sum(abs, (predicted - actual).^pn, dims=1).^(1/pn)) #+ γ*sum(abs, params)
end

# Direct Local Loss: sum(Lp(time point error))
function sum_χp_error(predicted, actual, pn)
    return sum(sum(abs.(predicted - actual).^pn ./ (actual.+1e-4), dims=2))
end

function fit_glv!(p_full_all, ptix_full, sc_dict, c, p_ranges; γ = 1e-4, pn = 2, ww = 0.1, strategy = "global", load_pop = nothing, gsa=false)
    # strategy == "global" => use bbo to opt w differential evolutionary algorithm
    # strategy == "local" => use DiffEqFlux to opt w quasi-newton descent (& normalzied loss)

    p_red = p_full_all[ptix_full, c] # reduced set for opt (don't have data for all params)
    p_full =  p_full_all[:, c] # for glv computations

    if c ∈ [1,5] #long optimization of initial fitting
        optopt = 1
    else #shorter opt with wander weight
        optopt = 2
        p_comp = p_full_all[:, c-1]
        p_red_comp = p_comp[ptix_full]
    end

    # Loss wrapper for training (simulate and score)
    function loss(p_red)
        loss = γ*sum(abs, p_red) #add lasso regularization
        p_full[ptix_full] = p_red

        # iterate thru subcommunities in training
        for (com, (glv_plant, data_all)) in sc_dict
            data = data_all[:,:,c]
            n_sub = Int(length(com)/2)
            n_tp = length(data[1,:])
            n_reps = Int(size(data,1)/n_sub) #error-catch, should always be 3

            sum_loss = 0; throw = 0

            # iterate thru replicates
            for i in 1:n_reps
                rep_data = data[(i-1)*n_sub+1 : i*n_sub, :]
                x0 = rep_data[:,1]

                # skip replicates that had to be thrown out (contamination or lost in sequencing)
                if rep_data == zeros(n_sub, size(rep_data)[2])
                    throw += 1
                    continue
                end

                # simulate replicate from first data point
                callback = n_sub == 1 ? death_cb : cbset
                saveat = n_sub == 1 ? iso_interval : pair_interval
                sol = DifferentialEquations.solve(glv_plant, CVODE_BDF(), u0=x0, p=p_full, saveat=saveat, callback=callback, tstops=psg_t)
                predicted = Array(sol)

                # filter unstable or unrealistic trajectories
                if size(predicted, 2) == n_tp && maximum(predicted) < max_abu

                    # ignore lost data points (tp lost in sequencing)
                    predicted = predicted[:, map(!iszero, eachcol(rep_data))]
                    actual = rep_data[:, map(!iszero, eachcol(rep_data))]

                    # score
                    sum_loss += sum_pnorm_error(predicted, actual, pn)

                    # if strategy == "global" #old definition
                    #     sum_loss += sum_pnorm_error(predicted, actual, pn)
                    # elseif strategy == "local"
                    #     sum_loss += sum_χp_error(predicted, actual, pn)
                    # end

                else
                    sum_loss += Inf
                end
            end

            # add subsz-weighted, tp-rep-mean loss for each subcommunity in training
            # loss += ((2^n_sub)*sum_loss)/(n_tp*(n_reps - throw))
            loss += sum_loss/(n_tp*(n_reps - throw))
        end

        if optopt == 1
            return loss
        else
            # add deviation penalty for c !∈ [1,5]
            loss += ww*sum_pnorm_error(p_red, p_red_comp, 1.0)
            return loss
        end
    end

    if gsa == true
        return loss(p_red)
    end

    red_ranges = p_ranges[ptix_full]
    lower_bounds, upper_bounds = [i for (i,j) in red_ranges], [j for (i,j) in red_ranges]

    fitness_progress_history = Array{Tuple{Int, Float64},1}()
    bbo_callback = oc -> push!(fitness_progress_history, (num_func_evals(oc), best_fitness(oc)))
    flux_callback = function (p, l)
        push!(fitness_progress_history, (0, l))
        false
    end

    if strategy == "global"

        mfe, cbs = optopt == 1 ? (mfe1, cps1) : (mfe2, cps2)
        result = bboptimize(loss,
                SearchRange = red_ranges,
                NumDimensions = length(p_red),
                MaxFuncEvals = mfe,
                # CallbackFunction = cbs, CallbackInterval = 0.0, #progress bar
                CallbackFunction = bbo_callback, CallbackInterval = 0.0,
                TraceMode = :silent)

        p_red_min, min_loss = best_candidate(result), best_fitness(result)

    elseif strategy == "local" #grad

        its = optopt == 1 ? its1 : its2
        result = DiffEqFlux.sciml_train(loss,
                    p_red, BFGS(initial_stepnorm=0.01);
                    lower_bounds, upper_bounds,
                    maxiters = its,
                    cb = flux_callback
                    )

        p_red_min, min_loss = result.minimizer, result.minimum

    elseif strategy == "local_evo"
        radius = 0.5 

        load_mat = zeros((length(p_red), length(load_pop)))
        for i=1:length(load_pop)
            load_mat[:, i] = load_pop[i][ptix_full, c]
        end
        load_ranges = [(minimum(load_mat[i,:]), maximum(load_mat[i,:])) for i=1:length(p_red)] #adapt range to load pop
        e_load_ranges = [(l - radius*abs(l), h + radius*abs(h)) for (l,h) in load_ranges] #extend by radius%
        bd_e_load_ranges = [(max(e_load_ranges[i][1], red_ranges[i][1]), min(e_load_ranges[i][2], red_ranges[i][2])) for i=1:length(p_red)] #bound to og
        bde_lr_lb, bde_lr_ub = [i for (i,j) in bd_e_load_ranges], [j for (i,j) in bd_e_load_ranges]
        load_mat = clamp.(load_mat, bde_lr_lb, bde_lr_ub) # shouldnt be nec arhgjgjgjl

        mfe, cbs = optopt == 1 ? (mfe1, cps1) : (mfe2, cps2)
        optr = bbsetup(loss;
                Method = :adaptive_de_rand_1_bin,
                PopulationSize=100,
                SearchRange = bd_e_load_ranges,
                NumDimensions = length(p_red),
                MaxFuncEvals = mfe,
                # CallbackFunction = cbs, CallbackInterval = 0.0, #progress bar
                CallbackFunction = bbo_callback, CallbackInterval = 0.0,
                TraceMode = :silent);
        for i=1:length(load_pop)
            optr.optimizer.population[i] = load_mat[:, i]
        end

        result = bboptimize(optr)
        p_red_min, min_loss = best_candidate(result), best_fitness(result)
    end

    # sciml_train sometimes return slightly out of bds final soln (?)
    p_red_min = clamp.(p_red_min, lower_bounds, upper_bounds)

    p_full_all[ptix_full, c] = p_red_min
    return min_loss, fitness_progress_history
end

## Import Relative Abundance and OD Data

# Import OD Data
np = pyimport("numpy")

od = Dict()
for name in ["data", "pair_order", "cond_order"]
    imp = np.load(od_path * name * ".npy", allow_pickle = true)
    if name != "data"
        imp = [get(imp, i) for i=0:length(imp)-1]
    end
    get!(od, name, imp)
end
conds = od["cond_order"]

# Import Relative Abundance Data
(root, dirs, ra_files) = first(walkdir(ra_path))

ra_data = Dict()
for file in ra_files
    imp = np.load(root * file)
    get!(ra_data, split(file, "_data")[1], imp)
end

# Match OD and RA data
comm_list = ["full", "pkpa", "rapk", "rapkpa", "skpkpa", "pkbmpa", "pkpafg", "mppkpa"]
id = ["Full Community", "FG - FG", "RA - PK", "RA - FG", "SK - FG", "BM - FG", "FG - FG", "MP - FG"]

matches = Dict()
for i = 1:length(comm_list); get!(matches, comm_list[i], id[i]); end

# Combine RA and OD Data into Absolute Data of pairs and triples
training_data = Dict()
n_tp = 7
for com in comm_list
    if com == "full"; n_sub = 7; else; n_sub = Int(length(com)/2); end

    comm_ra = cat(dims=3, ra_data[com*"_g"], ra_data[com*"_t"]) # 3 reps * n_sub x 7 tps x 8 conditions
    pair_col_ix = findfirst(occursin.(od["pair_order"], matches[com]))

    comm_od = od["data"][:, pair_col_ix, :] # 7 tps x 8 conditions matrix
    comm_od = reshape(comm_od, (1, n_tp, length(od["cond_order"]))) # 1 x 7 tps x 8 conditions matrix
    comm_od = repeat(comm_od, 3*n_sub, 1, 1) # repeat the od_data so it spans 3 reps * n_sub

    comm_train = comm_ra .* comm_od[:, 2:end, :] # 2:end because no time zero in ra

    get!(training_data, com, comm_train)
end

# Add OD (== Absolute Data) for isogenic cultures
orgs = ["ra", "sk", "mp", "pk", "bm", "pa", "fg"]
comm_list = cat(orgs, comm_list; dims=1); filter!(x->x≠"full",comm_list)
comm_list = cat("full", comm_list; dims=1) #rearrange

for com in comm_list[2:n+1]
    comm_train = np.load(iso_path * uppercase(com) * ".npy")
    get!(training_data, com, comm_train)
end

# make the dims_to_train
train_ix = [[findfirst(occursin.(orgs, i)) for i in [j[k:k+1] for k in 1:2:length(j)]] for j in comm_list if j != "full"]

# define subcomm dictionary of glv problems and corresponding data
sc_dict = Dict()
for (i, com) in enumerate(comm_list)
    if com == "full"; continue; end
    n_sub = Int(length(com)/2)
    plix_subcomm, plix_full, ptix_full = pixs(n, [[]], train_ix[i-1])
    p_load = [] # artificat with special case uses
    gLV_somep! = p_loader(p_load, collect(plix_subcomm), collect(ptix_full))
    tspan = n_sub == 1 ? iso_tspan : pair_tspan
    get!(sc_dict, com, (ODEProblem(ODEFunction(gLV_somep!; jac=gLV_jac), zeros(n_sub), tspan), training_data[com]))
end

# find logical for all parameters to be optimized
trained = Set()
for (i, com) in enumerate(comm_list)
    global trained
    if com == "full"; continue; end
    dims_train = train_ix[i-1]
    data_all = training_data[com]
    pairs_to_train = unique([Set([i, j]) for i in dims_train for j in dims_train if i != j])
    trained = union(trained, pairs_to_train)
end
dims_load = collect.(trained);
null, p_red_ix, null = pixs(n, dims_load, collect(1:7))

# export organized training data for figures
# # save("training_data.jld", "data", training_data)
# # save("comm_list.jld", "data", comm_list)
# # save("p_red_ix.jld", "data", p_red_ix)

## Optimization

strategy = "local_evo" # "global" (evo), "local" (gradient), or "local_evo" (adapted evo)

mfe1, mfe2 = strategy == "global" ? (500000, 100000) : (500000, 100000) # evo its CHECK THIS ON VI
its1, its2 = 100, 100 #grad its
# η = 1e-2 #grad ini step

# Optimization Progress Bar
const Prog1 = Progress(mfe1, 0.5, "Optimizing...")
const Prog2 = Progress(mfe2, 0.5, "Optimizing...")
function cps1(optcontroller)
    global Prog1
    ProgressMeter.update!(Prog1, num_func_evals(optcontroller))
end
function cps2(optcontroller)
    global Prog2
    ProgressMeter.update!(Prog2, num_func_evals(optcontroller))
end

# Initialize for Optimization (artifact)
r_init = zeros(n)
A_init = zeros(n,n)
p_full_all = vcat(r_init, A_init'...)
p_full_all = repeat(p_full_all, 1, length(od["cond_order"])) # 56 params x 8 conditions

# Need to adjust so that ratio of r/-a_ii ∈ range of isogenic od !!
aij_max = 5.0
r_ranges = [(0.05, 1.0) for i in 1:n]
a_ranges = [i%(n+1) != 1 ? (-aij_max, aij_max) : (-aij_max, -0.25) for i in 1:n^2] # a_ij : a_ii
p_ranges = vcat(r_ranges, a_ranges)

# Fit model for each condition
print("\n\n Beginning...")
warnings = "None!";

# for ww ∈ [1e-3, 1e-2], run = 1:3 #[1e-2, 1e-1, 1e0, 1e1]
# # for ww ∈ [0], run = 1:3
#     println("Wander Weight magnitude of $ww ==============================")

#     load_pop, fphs = nothing, []
#     p_full_all = load("p_full_all_v7_"*string(run)*"_ww"*string(ww)*"_LE8.jld", "data") #grad opt
#     # p_full_all = load("p_full_all_best.jld", "data") #evo opt

#     if strategy == "local_evo"
#         files = cd(readdir, prepath)
#         # p_files = filter(startswith("p_full_all_v8_run_ww0.5_lo_iso.jld"), files) #confining search range to isogenic parameterized (hot start)
#         # p_files = filter(startswith("p_full_all_best.jld"), files) #confining search range to pairwise parameterized (hot start)
#         p_files = filter(startswith("p_full_all_v7_2_ww0.01_LE8_10k.jld"), files)
#         load_pop = [load(prepath * p_file, "data") for p_file in p_files]

#         # optimization around isogenic trained only
#         # push!(load_pop, copy(load_pop[1])) # one will be for min, one will be for max
#         # p_int_ix = load(prepath * "p_int_ix.jld", "data")
#         # load_pop[1][p_int_ix,:] .= -aij_max
#         # load_pop[2][p_int_ix,:] .= aij_max
#     end

#     for c=1:length(conds)
#     # for c ∈ [1, 5]
#         global warnings

#         println(" At " * split(string(Dates.now()),'T')[2] * " started to train: " * od["cond_order"][c])

#         warnings = @capture_err begin
#             min_loss, fph = fit_glv!(p_full_all, p_red_ix, sc_dict, c, p_ranges, ww=ww, strategy=strategy, load_pop=load_pop) #, its, optopt, n_step
#             push!(fphs, fph)
#         end

#         # min_loss, fph = fit_glv!(p_full_all, p_red_ix, sc_dict, c, p_ranges, ww=ww, strategy=strategy, load_pop=load_pop) # modifies p_full_all!

#         # println("   Min loss: ", min_loss)
#         # print("\n At " * split(string(Dates.now()),'T')[2] * " finished with " * od["cond_order"][c])

#         # save("p_full_all_v7_"*string(run)*"_ww"*string(ww)*"_LE8_10k.jld", "data", p_full_all)
#         # save("p_full_all_v7_"*string(run)*"_ww"*string(ww)*"_LE8_10k_fphs.jld", "data", fphs) # fitness progress of each opt
#     end
# end

#  println()
# # println("Warnings that appeared during optimization:")
# # println(warnings)
# println()

# println("He terminado ", split(string(Dates.now()),'T')[2])
# println()


## Global Sensitivity Analysis

gsa_radius_p = 0.05
p_file = "p_full_all"
p_full_all = load(to_arkin * p_file * ".jld", "data")
red_ranges = p_ranges[p_red_ix]
n_pr = length(red_ranges)

p_labs = Array{String}(undef, size(p_full_all)[1])
for i = 1:n; p_labs[i] = "r"*string(i); end
c=n+1
for x=1:n;
    for y=1:n;
        p_labs[c] = "α"*string(x)*string(y);
        # p_labs[c] = " " * uppercase(orgs[y]) * " ⇒ " * uppercase(orgs[x]) * " "; #dont forget aij means j->i!
        c+=1;
    end;
end

Si_T = load(to_arkin * "/fitting/" * "Si_T_bestish.jld", "data")
Si_1 = load(to_arkin * "/fitting/" * "Si_1_bestish.jld", "data")
# Si_T, Si_1 = zeros(length(conds), n_pr), zeros(length(conds), n_pr)
for c=5:8 #c=1:length(conds)
    global warnings, Si_T, Si_1

    p_full = p_full_all[:, c]
    p_red_ranges_gsa = [(p - gsa_radius_p * (red_ranges[i][2] - red_ranges[i][1]), p + gsa_radius_p * (red_ranges[i][2] - red_ranges[i][1])) for (i, p) in enumerate(p_full[p_red_ix])] # define uncertainty as local box based on percentage of prior (using whole prior is bad idea in gLV (inf))
    p_red_ranges_gsa = [(max(p_red_ranges_gsa[i][1], red_ranges[i][1]), min(p_red_ranges_gsa[i][2], red_ranges[i][2])) for i=1:n_pr] #bound to og

    # p_ranges_gsa = [(0.0, 0.0) for i=1:length(p_full)]
    # p_ranges_gsa[p_red_ix] = bd_p_red_ranges_gsa #leave zero range at untrained p

    println(" At " * split(string(Dates.now()),'T')[2] * " started to analyze: " * od["cond_order"][c])

    function glv_gsa(p_red)
        p_full_all[p_red_ix, c] = p_red
        l = fit_glv!(p_full_all, p_red_ix, sc_dict, c, p_ranges, gsa=true)
        l
    end

    warnings = @capture_err begin
        m = gsa(glv_gsa, Sobol(), p_red_ranges_gsa, N = 2000)  
        Si_T[c,:], Si_1[c,:] = m.ST, m.S1
    end

        # min_loss, fph = fit_glv!(p_full_all, p_red_ix, sc_dict, c, p_ranges, ww=ww, strategy=strategy, load_pop=load_pop) # modifies p_full_all!

        # println("   Min loss: ", min_loss)
        # print("\n At " * split(string(Dates.now()),'T')[2] * " finished with " * od["cond_order"][c])

    save(to_arkin * p_file * "_Si_T.jld", "data", Si_T)
    save(to_arkin * p_file * "_Si_1.jld", "data", Si_1)
    println("Finished at ", split(string(Dates.now()),'T')[2])
    println()
end

# bar(p_labs[p_red_ix], m.ST, legend=false, xticks=(1:n_pr, p_labs[p_red_ix]), xrotation=45)
# plot!(title="Total, Sobol, N=1000")

# bar(p_labs[p_red_ix], m.S1, legend=false, xticks=(1:n_pr, p_labs[p_red_ix]), xrotation=45)
# plot!(title="1st Order, Sobol, N=1000")

## Random Leftover


## Assessing Subcomm Parameterization
# push!(LOAD_PATH, "/Users/willsharpless/Documents/Julia/arkin/")
# p_full_all = load("/Users/willsharpless/Documents/Julia/arkin/p_full_all_v6_2.jld", "data")
#
# pl = Array{Plots.Plot{Plots.GRBackend},1}()
#
# for i = 2:length(comm_list)
#     com = comm_list[i]
#     println(com)
#     dims_load = train_ix[i-1]
#     data_all = training_data[com]
#     pal = palette(:tab10)[dims_load]
#
#     dims_sub = unique(vcat(dims_load..., dims_load))
#     n_sub = length(dims_sub)
#
#     plix_subcomm, plix_full, ptix_full = pixs(n, dims_load, dims_load)
#     p_nonopt = zeros(sum(ptix_full))
#
#     println(com)
#     for c=1:length(conds)
#         data = data_all[:,:,c]
#         n_reps = Int(size(data,1)/n_sub);
#         p1 = plot()
#
#         # Loading the ODE with the pretrained params
#         p_full = p_full_all[:,c]
#         p_load = p_full[plix_full] # load the pretrained params from combined set
#         gLV_somep! = p_loader(p_load, collect(plix_subcomm), collect(ptix_full))
#         glv_plant = ODEProblem(gLV_somep!, x0, tspan)
#
#         for i in 1:n_reps
#             rep_data = data[(i-1)*n_sub+1 : i*n_sub, :]; x0 = rep_data[:,1]
#             sol = DifferentialEquations.solve(glv_plant, Tsit5(), u0=x0, p=p_full, saveat=0.5, callback=cbset, tstops=psg_t)
#             # sol = DifferentialEquations.solve(glv_plant, Tsit5(), u0=x0, p=p_nonopt, saveat=24, callback=cbset, tstops=psg_t)
#             plot!(0:24:120, rep_data', alpha = 0.5, palette = pal, legend=false)
#             plot!(sol, palette = pal)
#         end
#
#         if i == 2
#             title!(conds[c])
#         end
#
#         push!(pl, p1)
#     end
# end
#
# plt = plot(pl..., layout = (length(comm_list)-1, length(conds)), legend = false, size = (2000,2000))
# display(plt)

## Assessing Model on Full Community Cultures
#
# full_data_all = training_data["full"] #not comm_listly used to train
# dims_load = collect.(trained);
# plix_subcomm, plix_full, ptix_full = pixs(n, dims_load, collect(1:7))
# p_nonopt = zeros(sum(ptix_full)) # fill zeros for all untrained params
#
# # Initialize
# dims_sub = unique(vcat(dims_load..., collect(1:7)))
# n_sub = length(dims_sub)
# # n_reps = Int(size(data,1)/n_sub);
# n_reps = 3
#
# # Plotting
# pl = Array{Plots.Plot{Plots.GRBackend},1}()
# conds = ["0.31 mM", "1 mM", "3.1 mM", "10 mM", "25°C", "27.5°C", "30°C", "32.5°C"]
# colors = palette(:tab10)
# pal = [colors[i] for i=1:7]
#
# for c = 1:8
#
#     data = full_data_all[:,:,c]
#     p1 = plot()
#
#     # Loading the ODE with the pretrained params
#     p_full = p_full_all[:,c]
#     p_load = p_full[plix_full] # load the pretrained params from combined set
#     # p_load = p_full[ones(length(p_full))]
#     gLV_somep! = p_loader(p_load, collect(plix_subcomm), collect(ptix_full))
#     # gLV_somep! = p_loader(p_load, ones(length(p_full)))
#     glv_plant = ODEProblem(gLV_somep!, x0, tspan)
#
#     for i in 1:n_reps
#         rep_data = data[(i-1)*n_sub+1 : i*n_sub, :]; x0 = rep_data[:,1]
#         sol = DifferentialEquations.solve(glv_plant, Tsit5(), u0=x0, p=p_full, saveat=24, callback=cbset, tstops=psg_t)
#         # sol = DifferentialEquations.solve(glv_plant, Tsit5(), u0=x0, p=p_nonopt, saveat=24, callback=cbset, tstops=psg_t)
#         plot!(0:24:120, rep_data', alpha = 0.5, palette = pal, legend=false)
#         plot!(sol, palette = pal)
#     end
#
#     title!(conds[c])
#     push!(pl, p1)
# end
#
# plt = plot(pl..., layout = (2, 4), legend = false, size = (1000,900))
# display(plt)

## Print optimized glv parameters

# for c=5:8
#     p = p_full_all[:, c]
#     # print(ns[c], '\n')
#     print("mu:     A: \n")
#     for i = 1:n
#         print('|',round.(p[i],digits=2),"|     |",round.(p[n*i + 1: n*(i+1)],digits=1),"| \n")
#     end
#     print('\n')
# end
#
# stop
