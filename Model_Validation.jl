### Validation and Analysis of Model Parameterization
# willsharpless@berkeley.edu or ucsd.edu
# Apr 28, 2021

using Dates
print("run: " * split(string(Dates.now()),'T')[1] * ", " * split(string(Dates.now()),'T')[2] * "\n")
print("\nprecompiling packages...\n")

if split(pwd(),'/')[2] == "Users" #local
    cd("/Users/willsharpless/")
    prepath = "/Documents/"
    od_path = pwd() * prepath * "Thesis/pairwise_second/od_readings/"
    ra_path = pwd() * prepath * "ipy/Arkinlab/SynComm_Analysis/RA_Data/"
    iso_path = pwd() * prepath * "Thesis/Data/Isogenic_curves/"
    path_to_figs = pwd() * prepath * "Thesis/write/Paper/figs/"
else
    prepath = "/" # arkin server
    od_path = pwd() * prepath * "arkin/od_readings/"
    ra_path = pwd() * prepath * "arkin/RA_Data/"
    iso_path = prepath * "Isogenic_curves/"
end
to_arkin = pwd() * prepath * "Julia/arkin/"


# Plotting
using Plots, StatsPlots

# General Math
using Random, StatsBase, Statistics, LinearAlgebra, DifferentialEquations, DiffEqSensitivity, Sundials

# # Optimization
# using Optim, DiffEqFlux
#
# # Network Analysis / Control
# using LightGraphs, Hungarian, Missings, MatrixNetworks, GraphPlot
# using ControlSystems

# Array Handling & IO
using SparseArrays, JLD, PyCall, LaTeXStrings

# Custom modules
external_path = "Julia/external/DriverSpecies-master/"
my_modules_path = "Julia/modules/"
push!(LOAD_PATH, pwd() * prepath * external_path)
push!(LOAD_PATH, pwd() * prepath * my_modules_path)
# using DriverSpeciesModule # Angulo package
# using Glv: gLV_pflat!, pixs # My gLV-related helper functions
using Glv: pixs # My gLV-related helper functions

cd(to_arkin)
np, pd = pyimport("numpy"), pyimport("pandas")


###############################################################
## Import Params and Data
###############################################################


n = 7; n_t = 7; n_c = 8; n_reps_s = 3; n_reps_v = 11;
tsteps = 24; x0 = zeros(n)
orgs = ["ra", "sk", "mp", "pk", "bm", "pa", "fg"]
conds = [get(np.load(od_path * "od_cond_order.npy", allow_pickle = true), i-1) for i=1:n_c]
conds[1:4], conds[5:8] = [split(conds[i])[1] for i=1:4], [L"25^{\circ} C", L"27.5^{\circ} C", L"30^{\circ} C", L"32.5^{\circ} C"]
v_conds = [L"0.31 mM \rightarrow 3.1 mM \rightarrow 0.31 mM", L"25^{\circ} C \, \rightarrow \,  30^{\circ} C \, \rightarrow \,  25^{\circ} C"]

comm_list = ["full", "pkpa", "rapk", "rapkpa", "skpkpa", "pkbmpa", "pkpafg", "mppkpa"] #actual
comm_list = cat(orgs, comm_list; dims=1); filter!(x->x≠"full", comm_list)
comm_list = cat("full", comm_list; dims=1)

# comm_list = load("comm_list.jld", "data")
training_data = load("training_data.jld", "data")

for com in comm_list[2:n+1]
    comm_train = np.load(iso_path * "iso_" * uppercase(com) * ".npy")
    get!(training_data, com, comm_train)
end

train_ix = [[findfirst(occursin.(orgs, i)) for i in [j[k:k+1] for k in 1:2:length(j)]] for j in comm_list if j != "full"]

# Full Community Experiments
full_STAT = load(to_arkin * "Full_Comm_statenv_all.jld", "data") # static environment, all, abs abun

cols = ["R$i" for i=1:n_reps_v]
VGVT_od_temp = pd.read_csv(od_path * "validation/VGVT_combined.csv", names=cols).values
vg_od = vcat([repeat(VGVT_od_temp[1:n_t,i]', n, 1) for i=1:n_reps_v]...)
vt_od = vcat([repeat(VGVT_od_temp[n_t + 1:end,i]', n, 1) for i=1:n_reps_v]...)
VGVT_od = cat(vg_od, vt_od;dims=3)
VGVT_ra = np.load(ra_path * "VGVT_data.npy")
full_VGVT = 0.5 * VGVT_od .* VGVT_ra

# Plotting
colors = palette(:tab10);
pal = [colors[i] for i=1:n];
# title = Plots.scatter(ones(3), marker=0,markeralpha=0, annotations=(2, ones(3)[2], Plots.text(p_file)), axis=false, grid=false, leg=false, size=(200,100));

# 0 bounding all states
dead(u,t,integrator) = any(u .< 1e-8) #|| any(u .> 10)
function die!(integrator); integrator.u[integrator.u .< 1e-8] .= 0; end
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
    if any(x .> 10); dx .= 0; end # no cc catch
end

function gLV_pflat!(dx, x, p, t)
    n = length(x); r = p[1:n]; A = Array{eltype(x),2}(undef,n,n) # unpack params
    for i=1:n; A[i,:] = p[n*i + 1: n*(i+1)]; end

    dx .= x.*(r + A*x)
    if any(x .> 100); dx .= 0; end # no cc catch
end

function gLV_jac(J, x, p, t) # formatted for CVODE_BDF
    n = size(x)[1]
    r = p[1:n];
    A = Array{eltype(x),2}(undef,n,n)
    for i=1:n; A[i,:] = p[n*i + 1: n*(i+1)]; end
    J = Diagonal(r + A*x) + Diagonal(x)*A
    nothing
end

function gLV!(dx, x, r, A, t)
    n = length(x);
    dx .= x.*(r + A*x)
    if any(x .> 100); dx .= 0; end # no cc catch
end


###############################################################
## Full Community - Static and Fluctuating Environments (Model Validation)
###############################################################


tspan = (0.0, 120.0); psg_t = [24.0, 72.0]; all_tp = collect(0:24.0:120)
pair_tspan = (0.0, 120.0); iso_tspan = (0.0, 48.0);
pair_t = 0.0:24:120; iso_t = 0.0:8:48
ϵ = 1e-3; tspans = [(0.0, 48+ϵ), (48+ϵ, 96+ϵ), (96+ϵ, 144.0)]; switch_o = [1,3,1];

## For finding final+fixed points (turns off plotting)
find_fp = false # final
unpsgd = false # fixed (only if find_fp)
if unpsgd; 
    tspan = (0.0, 2000.); 
    all_tp = collect(0:24.0:2000.); 
end

gr(display_type=:inline)

files = cd(readdir, to_arkin)
p_files = filter(startswith("p_full_all_bestish"), files) # best fit (1 parameter set)
# p_files = filter(!contains("10"), filter(endswith("_LEbest.jld"), files)) # ensemble (several sets)

# p_files_1 = filter(startswith("p_full_all_v6.4.2"), files)
# p_files_2 = filter(startswith("p_full_all_v6.4.3"), files)
#
# p_files = filter(startswith("p_full_all_v8"), files)
# p_files = ["p_full_all_v7_1_ww0.01_LEbest.jld", "p_full_all_v7_1_ww0.01_LEbestgd.jld"]

# p_files = filter(endswith("_LE8_100k.jld"), files)

# p_files = filter(endswith("_LE8_10k.jld"), files)
# p_files = filter(contains("LE8"))
# p_files = ["p_full_all_v7_1_ww0.1_lo.jld", "p_full_all_v7_1_ww10.0.jld"] #best
# p_files = ["p_full_all_v7_pol_1.jld", "p_full_all_v7_pp_g2_2_ww0.1_l.jld","p_full_all_v8_n1init_ww0.5_lo.jld"] #best
# p_full_all[p_full_all .< -2.] .= -2.

# # p_files = filter(endswith("SAMIN.jld"), files)
# p_files = cat(p_files_1, p_files_2, dims=1)
# # p_files = ["p_full_all_v6.4.1_1_ww0.1.jld"]
# g_best = "p_full_all_v6.4.3_popopt_ww0.5_pn2.0__de_r1b_rl.jld"
# t_best = "p_full_all_v6.4.3_4_ww1.0.jld"
# p_full_all_g = load(to_arkin * g_best, "data")
# p_full_all_t = load(to_arkin * t_best, "data")
# p_full_all = hcat(p_full_all_g[:,1:4], p_full_all_t[:,5:8])

scored_G, scored_T = [], []
fp_orgs_pred = zeros(n_reps_s, n, n_c, length(p_files))
fp_orgs_data = zeros(n_reps_s, n, n_c)
p_full_all = load(to_arkin * p_files[1], "data")

for (pfi, p_file) in enumerate(p_files)
    global tspan, tspans, psg_t, all_tp, switch_o, p_full_all, fp_orgs_data, fp_orgs_pred
    for pn ∈ [2]

        p_full_all = load(to_arkin * p_file, "data")
        p_χ2 = []

        # Static Environments
        pl = Array{Plots.Plot{Plots.GRBackend},1}(); pal = [colors[i] for i=1:n];
        # tspan = (0.0, 120.0); psg_t = [24.0, 72.0]; all_tp = collect(0:24.0:120)
        for c = 1:n_c

            data = full_STAT[:,:,c]
            ρ = zeros(n, n_reps_s); χ2 = zeros(n, n_reps_s)
            p1 = plot()

            # Loading the ODE with the pretrained params
            p_full = p_full_all[:,c]
            # p_load = p_full[plix_full] # load the pretrained params from combined set
            # gLV_somep! = p_loader(p_load, collect(plix_subcomm), collect(ptix_full))
            glv_plant = ODEProblem(ODEFunction(gLV_pflat!; jac=gLV_jac), x0, tspan)

            for i in 1:n_reps_s

                rep_data = data[(i-1)*n+1 : i*n, :]; x0 = rep_data[:,1]
                sol = DifferentialEquations.solve(glv_plant, CVODE_BDF(), u0=x0, p=p_full, callback=cbset, tstops=all_tp)

                plot!(sol, palette = pal, linewidth=2, alpha=0.8)
                # if i == n_reps_s; vline!(psg_t; linecolor = :black); end
                scatter!(0:24:120, rep_data', palette = pal, legend=false, markersize = 5, markeralpha = 0.8, markerstrokewidth = 0.5, markerstrokealpha = 0.8)

                sol_tp = Array(sol)[:, findall(t->(t ∈ all_tp), sol.t)]
                if !find_fp
                    ρ[:, i] = diag(cor(rep_data', sol_tp'))
                    χ2[:, i] = sum((rep_data - sol_tp).^2 ./ (rep_data.+1e-4), dims=2)
                else
                    fp_orgs_pred[i, :, c, pfi] = Array(sol)[:, end]
                    # println("Orgs alive at fp", findall(x->x>0,Array(sol)[:, end]))
                    fp_orgs_data[i, :, c] = rep_data[:,end]
                    # println("Orgs alive at final_point", findall(x->x>0,rep_data[:,end]))
                end
            end

            ylims!((0.0001,0.27))
            # yaxis!(:log10)
            # if c ∉ [1,5]; plot!(yticks=false); end

            vline!(psg_t, linecolor = :black, alpha=0.5)
            plot!(grid=true)
            plot!(xticks=all_tp)
            if c >= 5; plot!(xlabel=L"t\:\:(hours)"); else; plot!(xlabel="");end
            if c == 1 || c == 5; plot!(ylabel=L"Abundance\:\:(OD)"); else; plot!(ylabel="");end
            title!(latexstring(conds[c]))

            if !find_fp
                # title!(conds[c]*"\nχ2 = "*string(round.(mean(χ2,dims=2),digits=2)); titlefontsize=10)
                mean_χ² = string(round.(mean(χ2),digits=2)); mean_ρ² = string(round.(mean(filter(!isnan, ρ.^2)),digits=2))
                annotate!(96, 0.27, text("χ̄² = $mean_χ²", 8))
                push!(p_χ2, mean(χ2))
            end

            push!(pl, p1)
        end 

        if find_fp
            push!(scored_G, (p_file, 0))
            push!(scored_T, (p_file, 0)) 
            break 
        end

        # Switched Environments
        ϵ = 1e-3; tspans = [(0.0, 48+ϵ), (48+ϵ, 96+ϵ), (96+ϵ, 144.0)];
        switch_o = [1,3,1]; psg_t = [48.0, 96.0]; all_tp = collect(0:24.0:144)
        for v = 1:2

            p1 = plot()
            data = full_VGVT[:,:,v]
            ρ = zeros(n, n_reps_v); χ2 = zeros(n, n_reps_v)

            # iterate through replicates
            for i in 1:n_reps_v

                rep_data = data[(i-1)*n+1 : i*n, :]; x0 = rep_data[:,1]
                sols = []
                sols_t = []

                # simulate switched system
                for j=1:3

                    glv_plant = ODEProblem(ODEFunction(gLV_pflat!; jac=gLV_jac), x0, tspans[j])
                    p_full = p_full_all[:, Int((v-1)*4) + switch_o[j]]
                    # print(p_full)
                    solj = DifferentialEquations.solve(glv_plant, Tsit5(), u0=x0, p=p_full, callback=cbset, tstops=all_tp)
                    x0 = Array(solj)[:,end]
                    push!(sols, Array(solj))
                    push!(sols_t, solj.t)
                end

                # combine switched solutions
                sol = hcat(sols...)
                sol_t = vcat(sols_t...)
                plot!(sol_t, sol', palette = pal, linewidth=2, alpha=0.6)
                scatter!(all_tp, rep_data', palette = pal, legend=false, markersize = 5, markeralpha = 0.8, markerstrokewidth = 0.5, markerstrokealpha = 0.8)

                # compute pearson for each species for replicate i
                sol_tp = sol[:, findall(t->(t ∈ all_tp), sol_t)]
                ρ[:, i] = diag(cor(rep_data', sol_tp'))
                χ2[:, i] = sum((rep_data - sol_tp).^2 ./ (rep_data.+1e-4), dims=2)
            end

            # annotate!("ρ = ")
            ylims!((0.0001,0.27))
            # yaxis!(:log10)
            vline!(psg_t, linecolor = :black, alpha=0.5)
            plot!(grid=true)
            # if v == 1; plot!(xticks=all_tp, xlabel=""); else; plot!(xticks=all_tp , xlabel=L"t"); end
            plot!(xticks=all_tp, xlabel="")
            # title!(v_conds[v] * ", ρ_med = " * string(median(mean(ρ, dims=2))))
            # title!(v_conds[v]*"\nχ2 = "*string(round.(mean(χ2,dims=2),digits=2)); titlefontsize=10)
            mean_χ² = string(round.(mean(χ2),digits=2)); mean_ρ² = string(round.(mean(filter(!isnan, ρ.^2)),digits=2))
            # title!(v_conds[v]*"\n χ̄² = $mean_χ² \n"; titlefontsize=10)
            annotate!(120, 0.27, text("χ̄² = $mean_χ²", 8))
            title = v_conds[v]
            # title!(L"text{%$title")
            title!(latexstring(v_conds[v]))
            insert!(pl, [5,10][v], p1)
            insert!(p_χ2, [5,10][v], mean(χ2))
            if v == 2; xlabel!(L"t\:\:(hours)"); end
            # println(ρ)
        end

        # laot = @layout [a b c d i; e f g h j]
        # plt = plot(pl..., layout = (2,5), legend = false, size = (2160,1200), bottom_margin = 20Plots.px);
        plt_upper = plot(pl[1:5]..., layout = (1,5), legend = false, size = (1600,700), top_margin = 20Plots.px, bottom_margin = 20Plots.px);
        plt_lower = plot(pl[6:10]..., layout = (1,5), legend = false, size = (1600,700), top_margin = 20Plots.px, bottom_margin = 30Plots.px);
        # title = plot(title = p_file * "_pn"* string(pn) * "_NMopt_v2,  1e-15 kill", grid = false, axis = false, bordercolor = "white", yticks=nothing, bottom_margin = -25Plots.px)
        title = plot(title="", grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, bottom_margin = -25Plots.px)
        spacer = plot(grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, bottom_margin = -25Plots.px)

        # plt_w_title = plot(title, plt, layout = @layout([A{0.025h}; B]))
        plt_w_spacer = plot(title, plt_upper, spacer, plt_lower, layout = @layout([A{0.025h}; B; C{0.025h}; D]), left_margin = 25Plots.px)
        savefig(plt_w_spacer, path_to_figs *"model_validation.pdf") 
        display(plt_w_spacer)
        push!(scored_G, (p_file, mean(p_χ2[1:5])))
        push!(scored_T, (p_file, mean(p_χ2[6:10])))
    end
    # break

    fp_orgs_pred[fp_orgs_pred .< 1e-5] .= 0
end

# save(to_arkin * "p_full_all_..._.jld", "data", p_full_all)


###############################################################
## Assessing Subcomm Parameterization (Model Fit)
###############################################################


plt_iso = plot()
plt = plot()

for p_file in p_files
    global p_full_all, plot_iso, plt
    p_full_all = load(to_arkin * p_file, "data")

    pl = Array{Plots.Plot{Plots.GRBackend},1}()
    g_plots = Array{Plots.Plot{Plots.GRBackend},1}();
    pal = [colors[i] for i=1:n];

    for i = 2:length(comm_list)
        
        # println(com)
        com = comm_list[i]
        dims_load = train_ix[i-1]
        data_all = training_data[com]
        pal = palette(:tab10)[dims_load]

        dims_sub = unique(vcat(dims_load..., dims_load))
        n_sub = length(dims_sub)
        tspan = n_sub == 1 ? iso_tspan : pair_tspan
        t_arr = n_sub == 1 ? iso_t : pair_t; all_tp = collect(t_arr)
        psg_t = n_sub == 1 ? [] : [24.0, 72.0]
        yub = n_sub == 1 ? 0.5 : 0.35
        yub_l = n_sub == 1 ? 0.47 : 0.34

        # plix_subcomm, plix_full, ptix_full = pixs(n, dims_load, dims_load)
        plix_subcomm, plix_full, ptix_full = pixs(n, [[]], train_ix[i-1])
        p_nonopt = zeros(sum(ptix_full))

        for c=1:length(conds)
            data = data_all[:,:,c]
            n_reps = Int(size(data,1)/n_sub);

            p1 = plot()
            χ2 = zeros(n_sub, n_reps)

            # Loading the ODE with the pretrained params
            # p_full_all = load(to_arkin * p_file, "data")
            p_full = p_full_all[:,c]
            p_load = [] # p_full[plix_full] # load the pretrained params from combined set
            gLV_somep! = p_loader(p_load, collect(plix_subcomm), collect(ptix_full))
            glv_plant = ODEProblem(gLV_somep!, x0, tspan)

            if com == "mppkpa" && c == 4
                test = (p_full, ptix_full, glv_plant, data, pal, n_sub, t_arr)
            end

            for i in 1:n_reps
                rep_data = data[(i-1)*n_sub+1 : i*n_sub, :]; x0 = rep_data[:,1]
                sol = DifferentialEquations.solve(glv_plant, CVODE_BDF(), u0=x0, p=p_full, callback=cbset, tstops=all_tp)
                # sol = DifferentialEquations.solve(glv_plant, Tsit5(), u0=x0, p=p_nonopt, saveat=24, callback=cbset, tstops=psg_t)
                # plot!(0:24:120, rep_data', alpha = 0.5, palette = pal, legend=false)
                scatter!(t_arr, rep_data', palette = pal, legend=false, markersize = 5, markeralpha = 0.8, markerstrokewidth = 0.5, markerstrokealpha = 0.8)
                plot!(sol, palette = pal)

                bug_fix = findall(t->(t ∈ all_tp), sol.t)
                if length(bug_fix) != length(all_tp) #DifferentialEquations shouldnt be returning 2 sol at same time?
                    deleteat!(bug_fix, findall(d -> d < 2, diff(bug_fix)))
                end
                sol_tp = sol[:, bug_fix]

                # println(size(rep_data), "rep data")
                # println(size(sol_tp), "sol tp")
                # println(all_tp, "all tp")
                # println(findall(t->(t ∈ all_tp), sol.t), "sol tp")
                χ2[:, i] = sum((rep_data - sol_tp).^2 ./ (rep_data.+1e-4), dims=2)
            end

            ylims!(0.0,yub)
            if i == 2 || i == 9
                title!(latexstring(conds[c]))
            end
            if i != 8 && i != 15
                xlabel!("")
            end
            if c == 1
                ylabel!(L"Abundance\:(OD)")
            end
            mean_χ² = string(round.(mean(χ2),digits=2));
            annotate!(t_arr[end-1], yub_l, text("χ̄² = $mean_χ²", 8))

            push!(pl, p1)
        end

        g_pl = plot(load("graphs/" * com * "_graph.png"), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing)
        push!(g_plots, g_pl)
    end

    g_plot_iso = plot(g_plots[1:n]..., layout=@layout([A; B; C; D; E; F; G]))
    g_plot = plot(g_plots[n+1:end]..., layout=@layout([A; B; C; D; E; F; G]))

    plt_iso = plot(pl[1:56]..., layout = (n, length(conds)), legend = false, size = (2000,2000))
    plt = plot(pl[57:end]..., layout = (length(comm_list)-1-n, length(conds)), legend = false, size = (2000,2000))

    spacer = plot(xticks=[], yticks=[], showaxis = false, grid = false)

    plt_iso = plot(g_plot_iso, spacer, plt_iso, layout = @layout([A{0.08w} C{0.01w} B]), legend = false, size = (2300,2000))
    plt = plot(g_plot, spacer, plt, layout = @layout([A{0.08w} C{0.01w} B]), legend = false, size = (2300,2000))

    savefig(plt_iso, path_to_figs *"model_fit_iso.pdf")
    savefig(plt, path_to_figs *"model_fit.pdf")
    display(plt_iso)
    display(plt)
end


###############################################################
## Interaction Change
###############################################################
# requires running an ensemble through the Model Validation loop above


conc = [0.31,1,3.1,10]; temp = [25, 27.5, 30, 32.5]
ww = [L"ww = 1e-2",L"ww = 1e-1",L"ww = 1"]
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

focus_orgs = [1,4,6] #collect(1:7) == all
n_f = length(focus_orgs)
_, pix_focus_iso, pix_focus_int = pixs(n, [[i] for i in focus_orgs], focus_orgs)
c2inc = [1,2,3,4]
min_Δ = 0 #0.0001

# ptypes = [
#     (L"Growth\:Rate\:(r)", collect(1:n), [0.25,1.0]),
#     (L"Self\:Interaction\:(\alpha_{ii})", collect(n+1:n+1:n^2+n), [-5.0,0.0]),
#     (L"Interaction\:(\alpha_{ij})", filter!(x->x∉collect(n+1:n+1:n^2+n), collect(n+1:n^2+n)), [-5.0,5.0])
# ]

### aij plot 

pl = Array{Plots.Plot{Plots.GRBackend},1}()
n_pf = sum(pix_focus_int)
pal = [palette(:rainbow,n_pf)[i] for i=1:n_pf];

for j = 1:2
    for jj=1:3
        if j==1; p1 = plot(xaxis=:log); else; p1 = plot();end
        xaxis!([L"Glucose\:(mM)", L"Temperature\:(^{\circ} C)"][j])
        if jj==1; yaxis!(L"Parameter\:Magnitude"); end
        scored = [scored_G, scored_T][j]

        for (p_file, score) in scored[jj:3:end]
            p_full_all = load(to_arkin * p_file, "data")
            p_full_all_focus_int = p_full_all[pix_focus_int, :] # only plotting parameters related to focus_orgs

            for i=1:n_pf

                # if i ≤ n_f || i % (n_f + 1) == 0; continue; end # only αij

                p_series = p_full_all_focus_int[i, [1:4, 5:8][j]]
                if mean(abs.(diff(p_series))) < min_Δ; continue; end

                # line color based on species which parameter directly affects
                # if i <= n; linecolor = pal[i]; else; linecolor = pal[Int(floor((i-1)/n))]; end

                plot!([conc, temp][j][c2inc], p_series[c2inc], 
                    linecolor=pal[i], xticks = [conc, temp][j], 
                    # alpha = clamp(1 - 2.5*score, 0.2, 1),
                    # linewidth = 3*(1 - 2*score), 
                    shape=:circ, markercolor=pal[i],
                    label="")
                xpos = 12.1; ypos = p_series[4]
                # annotate!(xpos, ypos, text(p_labs[i], :right, 15))
            end
        end
        if j==1; plot!(xticks=(conc,["0.31","1","3.1","10"])); end
        
        χ2 = mean([s[2] for s in scored[jj:3:end]])
        mean_χ² = string(round.(mean(χ2),digits=2));
        annotate!([5.6234, 31.25][j], 0.25, text("χ̄² = $mean_χ²", 8))

        title!(ww[jj], titlefontsize=12)
        ylims!(-4, 1)
        push!(pl, p1)
    end
end

# plt = plot(pl..., layout = (2,4), legend = false, size = (900,800))
leg = plot(zeros((1,6)), linewidth=5, xticks=[], yticks=[], showaxis = false, grid = false, legend=:outerright,
        label = permutedims(p_labs[pix_focus_int]), palette=pal, legendfontsize=12, thickness_scaling = 1)
spacer = plot(xticks=[], yticks=[], showaxis = false, grid = false)
plt = plot(spacer, pl..., leg, layout = @layout([A{.01w} [B C D; F G H] J{.1w}]), size = (1000,600), bottom_margin = 20Plots.px, dpi=500)
# savefig(plt, path_to_figs * "intrxns_over_wwfits.pdf")
display(plt)


### -ri/αii plot 
pl = Array{Plots.Plot{Plots.GRBackend},1}()
pal = [palette(:tab10)[i] for i ∈ focus_orgs];

for j = 1:2
    for jj=1:3
        if j==1; p1 = plot(xaxis=:log); else; p1 = plot(); end
        if jj==1; plot!(yaxis=(L"Absolute\:Abundance")); end
        xaxis!([L"Glucose\:(mM)", L"Temperature\:(^{\circ} C)"][j])
        scored = [scored_G, scored_T][j]

        for (p_file, score) in scored[jj:3:end]
            
            p_full_all = load(to_arkin * p_file, "data")
            p_full_all_focus_int = p_full_all[pix_focus_iso, :] # only plotting parameters related to focus_orgs

            for i=1:n_f

                p_series = -p_full_all_focus_int[i, [1:4, 5:8][j]] ./ p_full_all_focus_int[i+n_f, [1:4, 5:8][j]] #-ri/αii
                if mean(abs.(diff(p_series))) < min_Δ; continue; end

                # line color based on species which parameter directly affects
                # if i <= n; linecolor = pal[i]; else; linecolor = pal[Int(floor((i-1)/n))]; end

                plot!([conc, temp][j][c2inc], p_series[c2inc], 
                    linewidth = 2, linecolor=pal[i], xticks = [conc, temp][j], 
                    alpha = clamp(1 - score, 0.2, 1), 
                    shape=:circ, markercolor=pal[i],
                    label="")
                xpos = 12.1; ypos = p_series[4]
                # annotate!(xpos, ypos, text(p_labs[i], :right, 15))
            end
        end
    if j==1; plot!(xticks=(conc,["0.31","1","3.1","10"])); end
    title!(ww[jj], titlefontsize=12)
    ylims!(0.1, 0.4)
    push!(pl, p1)
    end
end

# plt = plot(pl..., layout = (2,4), legend = false, size = (900,800))
leg = plot(zeros((1,3)), linewidth=3, xticks=[], yticks=[], showaxis = false, grid = false, legend=:right, left_margin = -20Plots.px,
        label = permutedims(uppercase.(orgs[focus_orgs])), palette=pal, legendfontsize=12, thickness_scaling = 1)
spacer = plot(xticks=[], yticks=[], showaxis = false, grid = false)
plt = plot(spacer, pl..., leg, layout = @layout([A{.01w} [B C D; E F G] H{.1w}]), size = (1000,600), bottom_margin = 20Plots.px, dpi=500)
# savefig(plt, path_to_figs * "isoabun_over_wwfits.pdf")
display(plt)

## Final Point of Principal Orgs Plot
# before doing this run validation with find_fp, unpsgd = (true, false) and best p_file

fp_data_mean = reshape(mean(fp_orgs_data, dims=1)[:, focus_orgs, :], 3, 8) # over experimental reps
fp_pred_mean = reshape(mean(fp_orgs_pred, dims=1)[:, focus_orgs, :, :], 3, 8)  # over simulated reps

plg = plot()
plot!(fp_data_mean[1,1:4], fp_data_mean[2,1:4], fp_data_mean[3,1:4], lw = 2, label= "Glucose", color=:blue, shape=:circ)
plot!(fp_pred_mean[1,1:4], fp_pred_mean[2,1:4], fp_pred_mean[3,1:4], lw = 2, label= "Glucose Model", color=:blue, linestyle=:dash, shape=:circ)
plot!(xlabel=L"RA", ylabel=L"PK", zlabel=L"PA", gridstyle=:solid, gridlinewidth=3, legend=false, camera=(80,30))

plt = plot()
plot!(fp_data_mean[1,5:8], fp_data_mean[2,5:8], fp_data_mean[3,5:8], lw = 2, label= "Temperature", color=:orange, shape=:circ)
plot!(fp_pred_mean[1,5:8], fp_pred_mean[2,5:8], fp_pred_mean[3,5:8], lw = 2, label= "Temperature Model", color=:orange, linestyle=:dash, shape=:circ)
plot!(xlabel=L"RA", ylabel=L"PK", zlabel=L"PA", gridstyle=:solid, gridlinewidth=3, legend=false, camera=(80,30))

pl = plot(plg, plt)
display(pl)


###############################################################
## Rate of Convergence Plots
###############################################################
# before doing this run validation with find_fp, unpsgd .= true and entire ensemble p_files


function gLV_jac2(x, p)
    n = size(x)[1]
    r = p[1:n];
    A = Array{eltype(x),2}(undef,n,n)
    for i=1:n; A[i,:] = p[n*i + 1: n*(i+1)]; end
    J = Diagonal(r + A*x) + Diagonal(x)*A
    return J
end

# ww = ["best"]
n_w = length(ww); n_reps_e = 3

pl = Array{Plots.Plot{Plots.GRBackend},1}()
for j = 1:2

    if j==1; 
        p1 = plot(xaxis=:log);
        yaxis!(L"Maximum\:Eigenvalue");
    else; 
        p1 = plot(); 
    end
    xaxis!([L"Glucose\:(mM)", L"Temperature\:(^{\circ} C)"][j])
    scored = [scored_G, scored_T][j]

    mineig = zeros(n_w * n_reps_e, Int(n_c/2))
    for jj=1:n_w

        # mineig = zeros(n_reps_e, Int(n_c/2))
        for i=1:n_reps_e 

            p_file, score = scored[jj:n_w:end][i]
            p_full_all = load(to_arkin * p_file, "data")

            for c = 1:Int(n_c/2)

                # import converged sub comm mems
                pfi = findfirst(x -> x[1] == p_file, scored)
                dims_sub = findall(x->x>0, fp_orgs_pred[1, :, c + (j-1)*4, pfi])
                n_sub = length(dims_sub)

                # pull correct subcomm parameters
                _, pix_focus_iso, pix_focus_int = pixs(n, [[i] for i in dims_sub], dims_sub)
                plix_full = BitVector(pix_focus_iso + pix_focus_int)
                p = p_full_all[plix_full, c + (j-1)*4]
                
                # reshape into gLV structures
                r = p[1:n_sub];
                A = Array{eltype(p),2}(undef,n_sub,n_sub)
                for i=1:n_sub; A[i,:] = p[n_sub*i + 1: n_sub*(i+1)]; end

                # compute subcommunity fp and min(eig(Jac(∘))) of it
                J = gLV_jac2(-inv(A)*r, p)
                r_eigs = real.(eigvals(J))
                # println(r_eigs)

                # filter!(x->x<0, r_eigs)
                if any(r_eigs .> 0); 
                    println("\nConverged to an Unstable Point ???? Something is wrong"); 
                    println("   at " * p_file * " and c=" * string(c + (j-1)*4)); 
                    println("Real(eigs) = ", r_eigs)
                end

                # mineig[(jj-1)*n_w + i, c] = minimum(abs.(r_eigs)) # magnitude
                mineig[(jj-1)*n_w + i, c] = maximum(r_eigs)
            end
        end
        # plot!([conc, temp][j], permutedims(mean(mineig, dims=1)), label = ww[jj]) # averaging over ensemble
        # plot!([conc, temp][j], permutedims(mineig), label = ww[jj]) # averaging over ensemble
    end
    plot!([conc, temp][j], permutedims(mean(mineig, dims=1)), ribbon=permutedims(std(mineig, dims=1))/2, fillalpha=.25, label="") # averaging over ensemble

    if j==1; plot!(xticks=(conc,["0.31","1","3.1","10"])); end
    ylims!(-0.35, 0.005)
    push!(pl, p1)
end

roc_pl = plot(pl..., layout=@layout([A B]), size = (750,450), bottom_margin = 20Plots.px, dpi=500)
savefig(roc_pl, path_to_figs * "ROC_change.pdf")
display(roc_pl)


###############################################################
## Export the gLV subcommunity consisting of three most abundant members
###############################################################
# exporting the params in r, A formats for python HJI reachability script, plotting for validation


plix_subcomm_3, plix_full_3, ptix_full_3 = pixs(n, train_ix[3], train_ix[3])
p_file = p_files[1]

function plot_and_save_3_major(p_full_all)
    pl = Array{Plots.Plot{Plots.GRBackend},1}()

    i = 4 + 7
    com = comm_list[i]
    dims_train = train_ix[i-1]
    dims_load = []
    data_all = training_data[com]
    pal = palette(:tab10)[dims_load]

    dims_sub = unique(vcat(dims_load..., dims_train))
    n_sub = length(dims_sub)

    pairs_trained = unique([Set([i, j]) for i in dims_train for j in dims_train if i != j])
    dims_load = collect.(pairs_trained)

    # use pixs to compute the idx of all trained params
    plix_subcomm, plix_full, ptix_full = pixs(n, dims_load, dims_train)
    p_nonopt = zeros(sum(ptix_full))

    p_full_all_rapkpa = p_full_all[plix_full, :]
    # JLD.save(to_arkin * "p_full_all_rapkpa_bestish_1.jld", "data", p_full_all_rapkpa)

    # Plot
    tspan = (0.0, 120.0);
    pair_tspan = (0.0, 120.0); iso_tspan = (0.0, 48.0);
    pair_t = 0.0:24:120; iso_t = 0.0:8:48
    psg_t = [24.0, 72.0];
    pal = [colors[i] for i ∈ dims_sub];
    tspan = (0.0, 300.0); # psg_t = [100.0, 200.0]; # psg_t = [300.0]

    # println(com)
    for c=1:length(conds)
        data = data_all[:,:,c]
        n_reps = Int(size(data,1)/n_sub);
        p1 = plot()

        # Loading the ODE with the pretrained params
        p_full = p_full_all[:,c]
        p_load = p_full[plix_full] # load the pretrained params from combined set
        gLV_somep! = p_loader(p_load, collect(plix_subcomm), collect(ptix_full))
        glv_plant = ODEProblem(gLV_somep!, x0, tspan)

        for i in 1:n_reps
            rep_data = data[(i-1)*n_sub+1 : i*n_sub, :]; x0 = rep_data[:,1]
            sol = DifferentialEquations.solve(glv_plant, Tsit5(), u0=x0, p=p_full, callback=cbset, tstops=psg_t)
            plot!(0:24:120, rep_data', alpha = 0.5, palette = pal, legend=false)
            plot!(sol, palette = pal)
        end

        title!(conds[c])
        push!(pl, p1)

        # Reshape into r and A, and export
        n = n_sub
        p_r_A_flat = Array{Real,1}(undef, n*(n+1))
        p_r_A_flat[plix_subcomm] = p_load
        p_r_A_flat[.!collect(plix_subcomm)] = p_full[collect(ptix_full)]
        r = copy(p_r_A_flat[1:n]);
        A = Array{Real,2}(undef,n,n)
        for i=1:n; A[i,:] = p_r_A_flat[n*i + 1: n*(i+1)]; end

        # save_path = "/Users/willsharpless/Documents/Julia/arkin/r_A_bestish/"
        # np.save(save_path*"r_"*split(conds[c])[1], r)
        # np.save(save_path*"A_"*split(conds[c])[1], A)
    end

    plt = plot(pl..., layout = (1, length(conds)), legend = false, size = (2000,500))
    title = plot(title = p_file * ": only 3 major", grid = false, axis = false, bordercolor = "white", yticks=nothing, bottom_margin = -25Plots.px)
    plt_w_title = plot(title, plt, layout = @layout([A{0.025h}; B]))
    display(plt_w_title)

end

# plot_and_save_3_major(p_full_all)


###############################################################
## Interrogating Major Subcommunity Stability with and without Passaging
###############################################################


i = 11 # RA - PK - PA
com = comm_list[i]
data_all = training_data[com]
dims_sub = train_ix[i-1]; n_sub = length(dims_sub)

plix_subcomm_3, plix_full_3, ptix_full_3 = pixs(n, dims_sub, dims_sub)
p_file = p_files[1]
p_full_all_rapkpa = p_full_all[plix_full_3, :]

# Plot
pal = [colors[i] for i ∈ dims_sub];

psg_int, tf = 5*48., 576.
tspan = (0.0, tf);
psg_t = collect(0:psg_int:tspan[2])
psg_ints = [12*(2^i) for i=1:4]; reverse!(psg_ints)
psg_ints = cat(tf, psg_ints, dims=1)
pl_G = Array{Plots.Plot{Plots.GRBackend},1}()
pl_T = Array{Plots.Plot{Plots.GRBackend},1}()

# for psg_ts ∈ [[], collect(0:psg_int:tspan[2])], c=1:length(conds)
for (psgi, psg_int) ∈ enumerate(psg_ints), c=1:length(conds)
    global psg_t
    psg_t = collect(0:psg_int:tspan[2])
    data = data_all[:,:,c]
    n_reps = Int(size(data,1)/n_sub);
    p1 = plot()

    # Loading the ODE with the pretrained params
    p_full = p_full_all[:,c]
    p_load = p_full[plix_full_3] # load the pretrained params from combined set
    gLV_somep! = p_loader(p_load, collect(plix_subcomm_3), collect(ptix_full_3))
    glv_plant = ODEProblem(gLV_somep!, x0, tspan)

    for i in 1:n_reps
        rep_data = data[(i-1)*n_sub+1 : i*n_sub, :]; x0 = rep_data[:,1]
        sol = solve(glv_plant, CVODE_BDF(), u0=x0, p=p_full, callback=cbset, tstops=psg_t)
        # scatter!(0:24:120, rep_data', alpha = 0.5, palette = pal, legend=false)
        plot!(sol, palette = pal)
    end

    xlabel!("")
    if psg_int == tf
        plot!(title=latexstring(conds[c]), titlefontsize=20)
    elseif psg_int == psg_ints[end]
        xlabel!("t")
    end

    if c == 1 || c == 5
        ylabel=[L"None", L"\Delta_t=192", L"\Delta_t=96", L"\Delta_t=48", L"\Delta_t=24"][psgi]
        plot!(ylabel=latexstring(ylabel), yguidefontrotation=-90)
        annotate!(-5, 0.23, text("OD", 10))
    end

    if c == 4 || c == 8
        yub = 0.25
    else
        yub = 0.2
    end
    ylims!((0,yub))

    if c < 5
        push!(pl_G, p1)
    else
        push!(pl_T, p1)
    end
end

plt_G = plot(pl_G..., layout = (length(psg_ints), Int(length(conds)/2)), legend = false)
plt_T = plot(pl_T..., layout = (length(psg_ints), Int(length(conds)/2)), legend = false)
spacer = plot(grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing)
plt = plot(plt_G, spacer, plt_T, layout = @layout([A; B{0.001h}; C]), legend = false, size = 0.75.*(1700,2200))
plt_w_spacer = plot(spacer, plt, layout = @layout([A{0.02w} B]), bottom_margin = 15Plots.px, left_margin = 10Plots.px, right_margin = 10Plots.px)
# savefig(plt_w_spacer, path_to_figs * "EOPassaging_v.pdf")
# display(plt_w_spacer)


###############################################################
## Sensitivity Plots
###############################################################


p_red_ix = load(to_arkin * "p_red_ix.jld", "data")
# Si_T = load(to_arkin * p_file * "_Si_T.jld", "data")
# Si_1 = load(to_arkin * p_file * "_Si_1.jld", "data")
Si_T = load(to_arkin * "Si_T_bestish.jld", "data")
Si_1 = load(to_arkin * "Si_1_bestish.jld", "data")
n_pr = size(Si_T)[2]
Si = [abs.(Si_T), abs.(Si_1)] #bad
# Si = [Si_T, Si_1] 

conds = [L"0.31mM\:\:\:\:", L"1mM\:\:\:\:", L"3.1mM\:\:\:\:", L"10mM\:\:\:\:", L"25^{\circ} C\:\:\:\:", L"27.5^{\circ} C\:\:\:\:", L"30^{\circ} C\:\:\:\:", L"32.5^{\circ} C\:\:\:\:"]
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

Si_pl = Array{Plots.Plot{Plots.GRBackend},1}()
for c = 1:length(conds)

    if c == 1 || c == 5
        pl = Array{Plots.Plot{Plots.GRBackend},1}()
    end

    for o = 1:2
        p1 = plot()

        # bar!(p_labs[p_red_ix], Si[o][c, :], legend=false, xticks=(1:n_pr, p_labs[p_red_ix]), xrotation=45)

        bar!(1:n, Si[o][c, :][1:n], label="Growth Rates", color=colorant"blue")

        for i = n+1:n_pr
            if p_labs[p_red_ix][i][3] == p_labs[p_red_ix][i][4]
                label = i < 10 ? "Self-Interaction" : ""
                color = colorant"red"
            else                
                label = i < 10 ? "Interaction" : ""
                color = colorant"green"
            end
            bar!((i, Si[o][c, i]), label=label, color=color)
        end
        
        plot!(xticks=(1:n_pr, p_labs[p_red_ix]), xrotation=45, legend=false)

        if c == 1 || c == 5
            title!([L"Total", L"First\:\:Order"][o])
            if o == 2
                plot!(legend=true)
            end
        end

        if o == 1
            plot!(ylabel=latexstring(conds[c]), yguidefontrotation=-90)
            annotate!(-2.5, 0.63, text("Value", 8))
            ylims!(-0.025, 0.6)
        else
            # ylims!()
        end

        push!(pl, p1)
    end

    if c == 4 || c == 8
        si_plot = plot(pl..., layout=(Int(length(conds)/2), 2))
        push!(Si_pl, si_plot)
    end
end

spacer = plot(xticks=[], yticks=[], showaxis = false, grid = false)
# Si_plot = plot(Si_pl[1], spacer, Si_pl[2], layout=@layout([A B{0.025w} C]), size=(1000, 1000))

Si_G = plot(spacer, Si_pl[1], layout=@layout([A{0.025w} B]), size=(1000, 1000))
# savefig(Si_G, path_to_figs * "Si_G.pdf")
Si_T = plot(spacer, Si_pl[2], layout=@layout([A{0.025w} B]), size=(1000, 1000))
# savefig(Si_T, path_to_figs * "Si_T.pdf")


###############################################################
## Direct PCR Validation Plot
###############################################################


Orgs = latexstring.(uppercase.(orgs))
data_dpcr = [21.6, 20.1, 32.9, 21.9, 26.5, 18.1, 19.8] #copying from image of data fml
data_qiabt = [21.4, 20.6, 30.9, 19.7, 26.3, 17.4, 18.5]
data_dpcr_sd = [0.5, 0.5, 0.8, 0.4, 0.3, 0.5, 0.4]
data_qiabt_sd = [0.5, 1.8, 2.5, 0.9, 0.8, 0.3, 0.3]

dvq = groupedbar(Orgs, [data_dpcr data_qiabt], yerr = [data_dpcr_sd data_qiabt_sd], labels = ["Direct PCR" "Qiagen Blood & Tissue"], size=(400,300), legendfontsize=6, dpi=500)
ylabel!(L"Count", legendfontsize=6)
savefig(dvq, path_to_figs * "DvQ.png")


###############################################################
## Random Leftover
###############################################################


# for c=1:length(conds)
#     p_full = p_full_all[:,c]
#     p_load = p_full[plix_full] # load the pretrained params from combined set

#     n = length(x)
#     p_r_A_flat = Array{eltype(x),1}(undef, n*(n+1))
#     p_r_A_flat[gLV_somep.p_load_ix] = gLV_somep.p_load
#     p_r_A_flat[.!gLV_somep.p_load_ix] = p_full[gLV_somep.p_train_ix]

#     # unpack params
#     r = p_r_A_flat[1:n];
#     A = Array{eltype(x),2}(undef,n,n)
#     for i=1:n; A[i,:] = p_r_A_flat[n*i + 1: n*(i+1)]; end


##
#
# p_file = p_files[3]
# p_full_all = load(to_arkin * p_file, "data")
# p_int_ix = load(to_arkin * "p_int_ix.jld", "data")
# p_full_all[p_int_ix,1] .= -1.


(p_full, ptix_full, glv_plant, data, pal, n_sub, t_arr) = test

s = 1/10
cp_full = copy(p_full)
# p_full[findall(ptix_full .> 0)[1]] = 0.2957000037*s
# p_full[findall(ptix_full .> 0)[3]] = -1.0*s
# p_full[findall(ptix_full .> 0)[4]] = -2.634347181086732*s
# p_full[findall(ptix_full .> 0)[2]] = 0.3309000035*s
# p_full[findall(ptix_full .> 0)[5]] = -4.285279848381462*s
# p_full[findall(ptix_full .> 0)[6]] = -1.0*s

p1 = plot()
for i in 1:3
    rep_data = data[(i-1)*n_sub+1 : i*n_sub, :]; x0 = rep_data[:,1] #*0 .+ 0.2
    # sol = DifferentialEquations.solve(glv_plant, CVODE_BDF(), u0=x0, p=p_full, callback=cbset, tstops=psg_t)
    sol = DifferentialEquations.solve(glv_plant, CVODE_BDF(), u0=x0, p=cp_full, tstops=psg_t)
    # sol = DifferentialEquations.solve(glv_plant, Tsit5(), u0=x0, p=p_nonopt, saveat=24, callback=cbset, tstops=psg_t)
    # plot!(0:24:120, rep_data', alpha = 0.5, palette = pal, legend=false)
    scatter!(t_arr, rep_data', palette = pal, legend=false, markersize = 5, markeralpha = 0.8, markerstrokewidth = 0.5, markerstrokealpha = 0.8)
    plot!(sol, palette = pal)
end
display(p1)


# 1: ra
# 4: pk
# 6: pa
# 11: ra -> ra
# 13: pk -> ra
# 26: pa -> ra
# 29: ra -> pk
# 32: pk -> pk
# 34: pa -> pk
# 43: ra -> pa
# 46: pk -> pa
# 48: pa -> pa

# for i=1:2
#     p_full_all = load(to_arkin * "p_full_all_v7_pol_$i.jld", "data")
#
#     p_full_all[34, :] = 2*p_full_all[34, :]
#
# end

# # c =
# for i=2
#     p_full_all = load(to_arkin * "p_full_all_v7_pol_$i.jld", "data")
#
#     for c=1:8
#
#         # pk - pa
#         if c in [1,2,3,7]
#             p_full_all[34, c] = 1.3*p_full_all[34, c] # 34: pa -> pk
#             p_full_all[46, c] = 0.45*p_full_all[46, c] # 46: pk -> pa
#         elseif c in [5,6]
#             p_full_all[34, c] = 1.5*p_full_all[34, c] # 34: pa -> pk
#             p_full_all[46, c] = 0.4*p_full_all[46, c] # 46: pk -> pa
#         else
#             p_full_all[34, c] = 1.2*p_full_all[34, c] # 34: pa -> pk
#             p_full_all[46, c] = 0.48*p_full_all[46, c] # 46: pk -> pa
#         end
#
#         # ra - pk
#         # p_full_all[1, :] = p_full_all[4, :]
#
#         cu = [1.5, 1.25, 1.0, 1.0,
#             2.0, 2.0, 2.0, 1.5]
#         cd = [2.32005, 1.77155, 1.5375, 1.8356,
#             1.0909, 0.8, 0.82991, 0.5927]
#
#         p_full_all[13, c] = cu[c]*p_full_all[13, c] # 13: pa -> ra
#         p_full_all[29, c] = cd[c]*p_full_all[29, c] # 29: ra -> pk
#         p_full_all[11, 1:4] = 0.975*p_full_all[11, 1:4] # 29: pk -> ra
#         p_full_all[11, 5:8] = 0.9*p_full_all[11, 5:8] # 29: pk -> ra
#
#         # cj = [0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
#         # p_full_all[29, :] =cj[c]*p_full_all[29, :] # 29: pa -> ra
#
#     end
#     # p_full_all[6, :] = 2*p_full_all[6, :] # 6: pa
#     p_full_all[39, 4] = -1.0*p_full_all[39, 4] # 39: pk -> bm
#     p_full_all[33, 4] = -0.15*p_full_all[33, 4] # 33: bm -> pk
#     p_full_all[[53, 55], :] .= -1.0 # 53: pk -> fg, 55: pa -> fg
#     p_full_all[[25, 27, 31, 45], 5:8] = p_full_all[[25, 27, 31, 45], 1:4] #
#
#     save(to_arkin * "p_full_all_v7_pol2_$i.jld", "data", p_full_all)
# end
#
#
# test = copy(p_full_all)
# p_files = ["p_full_all_v7_pol_1.jld", "p_full_all_v7_pol_2.jld"]
# p_files = ["p_full_all_v7_pol2_1.jld", "p_full_all_v7_pol2_2.jld"]
# p_files = ["p_full_all_v7_pp_g2_2_ww0.1_l.jld"]
