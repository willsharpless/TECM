### Static vs Hybrid Volume Size Plot
# willsharpless@berkeley.edu or ucsd.edu
# Feb 30, 2022

using PyCall, Plots, LaTeXStrings, Colors
using Combinatorics

colors = palette(:default)
ctrl_species_set = collect(powerset([1, 2, 3]))
concs = [0.31, 1., 3.1, 10.]

np = pyimport("numpy")
opdm_path = "/Users/willsharpless/Documents/ipy/Tomlin/optimized_dp-master/"
od_path = "/Users/willsharpless/Documents/Thesis/pairwise_second/od_readings/"
cs_data_ar = np.load(opdm_path * "hybrid_BRT_glucose_bvs_cs_data.npy", allow_pickle=true)
cs_data_ar = py"cs_data_ar[()]"

plot()

for (csi, cs) in enumerate(ctrl_species_set[2:end])

    best_so, best_v, static_v = cs_data_ar[string(cs .- 1)]
    ratio_v = best_v./static_v
    
    # if cs==[2]
    #     plot!(concs, static_v, color=colors[csi], linestyle=:solid, label="(PK),(RA,PK),(RA,PA),(PK,PA) static")
    #     plot!(concs, best_v, color=colors[csi], linestyle=:dash, label="(PK),(RA,PK),(RA,PA),(PK,PA) static")
    # else
    cs_label = string((["RA", "PK", "PA"][cs] .* ", ")...)[1:end-2]
    plot!(concs, static_v .+ 1e-5 * csi, color=colors[csi], linestyle=:solid, label="Ctrl of species "*cs_label, shape=:circle, linewidth=2)
    plot!(concs, best_v  .+ 1e-5 * csi, color=colors[csi], linestyle=:dash, label="", shape=:circle, linewidth=2)
    # end
    
    # plt.plot(concs, ratio_v, label=str(cs))
end
for i=1:2
    label = ["Static", "Hybrid"][Int(i)]
    linestyle = [:solid, :dot][Int(i)]
    plot!([], []; label=label, color=:black, linestyle=linestyle, shape=:circle, linewidth=2)
end

ylabel!(L"Controllable\:\:State\:\:Volume\:\:(OD^3)")
xlabel!(L"Glucose\:\:(mM)")
plot!(legend=:bottom, xaxis=:log, yaxis=:log)
plot!(xticks=(concs,["0.31","1","3.1","10"]))
plot!(yticks=([10.0^-x for x=4:-1:1]))

savefig("/Users/willsharpless/Documents/Thesis/Write/Paper/figs/BRT_comp.png")

