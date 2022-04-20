### Network Fluctuation
# willsharpless@berkeley.edu or ucsd.edu
# Jul 25, 2021

prepath = "/Users/willsharpless/Documents/Julia"
external_path = "/external/DriverSpecies-master/"
my_modules_path = "/modules/"
push!(LOAD_PATH, prepath * external_path)
push!(LOAD_PATH, prepath * my_modules_path)

using JLD, Plots, LightGraphs, LaTeXStrings, GraphPlot, Compose, SparseArrays
using Cairo, Fontconfig, ImageMagick, Colors

using DriverSpeciesModule
# using Hungarian, Missings # the hungarian algorithm used to findall driver species
# using MatrixNetworks

using Glv: pixs

## Define Constants

n_full = 7
colors = palette(:tab10);
cmap = colormap("RdBu", 100)
scale_ix(val; min=-5.0, max=5.0, bins=98) = Int(ceil(bins*(val - min)/(max - min))) + 1

orgs = ["ra", "sk", "mp", "pk", "bm", "pa", "fg"]
actual = ["full", "pkpa", "rapk", "rapkpa", "skpkpa", "pkbmpa", "pkpafg", "mppkpa"]
train_ix = [[findfirst(occursin.(orgs, i)) for i in [j[k:k+1] for k in 1:2:length(j)]] for j in actual if j != "full"]
conds = [L"0.31\:mM", L"1\:mM", L"3.1\:mM", L"10\:mM", L"25^{\circ} C", L"27.5^{\circ} C", L"30^{\circ} C", L"32.5^{\circ} C"]
training_data = load("training_data.jld", "data")

to_arkin = "/Users/willsharpless/Documents/Julia/arkin/"
p_file = "p_full_all_best.jld"
p_full_all = load(to_arkin * p_file, "data")

path_to_figs = "/Users/willsharpless/Documents/Thesis/write/Paper/figs/"

## Make 3-member subcommunity

dims_train = train_ix[3]; dims_load = []
dims_sub = unique(vcat(dims_load..., dims_train))
n_sub = length(dims_sub)
pairs_trained = unique([Set([i, j]) for i in dims_train for j in dims_train if i != j])
dims_load = collect.(pairs_trained)
plix_subcomm, plix_full, ptix_full = pixs(n_full, dims_load, dims_train)
p_full_all_3 = p_full_all[plix_full, :]

## Colored Interaction Graph

function interaction_graph(n, p_set, dims_comm, c; self_loops=false, thr=0)

    p_full = p_set[:,c]
    A = Array{eltype(p_full_all),2}(undef,n,n)
    for i=1:n; A[i,:] = p_full[n*i + 1: n*(i+1)]; end

    G = DiGraph(n)
    elw = []; ec = [];
    for i=1:n, j=1:n
        if !self_loops
            if i != j && abs(A[i,j]) > thr; # omit self-loops and below-threshold
                add_edge!(G, j, i) # transpose due to nature of gLV
                push!(elw, abs(A[i,j]))
                push!(ec, cmap[scale_ix(A[i,j])])
            end
        else
            if abs(A[i,j]) > thr;
                add_edge!(G, i, j)
                push!(elw, abs(A[i,j]))
                push!(ec, cmap[scale_ix(A[i,j])])
            end
        end
    end

    graph_plot = gplot(G;
                    linetype="curve",
                    layout=circular_layout,
                    # layout=layout,
                    # nodelabel=titlecase.(orgs),
                    nodestrokec = colorant"white",
                    NODESIZE=[0.1 for c=1:n],
                    # NODESIZE=0.2*[0.1, 0.01, 0.01, 0.27, 0.01, 0.7, 0.01], #ra sized
                    EDGELINEWIDTH=2, #max width
                    edgestrokec=ec,
                    edgelinewidth=elw, #relative widths
                    arrowlengthfrac=0.1,
                    outangle=0.15,
                    nodefillc=[colors[i] for i ∈ dims_comm])

    # display(graph_plot)
    return G, elw, ec, graph_plot
end

## Network Fluctuation Figure

thresholds = [0.0, 0.5, 1.0]

# n = n_full; p_set = p_full_all; dims = collect(1:7)
Full = [n_full, p_full_all, thresholds, collect(1:7)]

# n = n_sub; p_set = p_full_all_3; dims = dims_sub
# Major_3 = [n_sub, p_full_all_3, thresholds, dims_sub]

function make_net_fluc_fig(n, pset, thresholds, dims)
    n_scc = zeros((length(conds), length(thresholds)))
    n_mds = zeros((length(conds), length(thresholds)))
    graph_plots = []

    # make graphs from params and compute scc & mds
    for (i, thr) ∈ enumerate(thresholds)
        # global n, pset, dims
        println(" With a threshold value of $thr ...")
        for c=1:length(conds)

            #full community
            # G, elq, ec, graph_plot = interaction_graph(n_full, p_full_all, collect(1:7), c; thr=thr)

            #3-major community
            # G, elq, ec, graph_plot = interaction_graph(n_sub, p_full_all_3, dims_sub, c; thr)

            G, elq, ec, graph_plot = interaction_graph(n, p_set, dims, c; thr=thr)

            if thr == 0.0
                push!(graph_plots, graph_plot)
                draw(PNG("fluc_graphs_$c.png", 16cm, 16cm, dpi=1000), graph_plot)
            end

            A = convert(SparseMatrixCSC{Float64,Int64}, adjacency_matrix(G))
            mds = DriverSpecies(A) #SparseMatrixCSC(A')
            scc = strongly_connected_components(G)

            n_scc[c, i], n_mds[c, i] = length(scc), length(mds)

            # println(mds)
            # println(conds[c], " with ")
            # println("MDS: Species ", mds)
            # println("SCC: ", scc)
            # display(graph_plot)
        end
    end

    pls = Array{Plots.Plot{Plots.GRBackend},1}()
    c_purples = range(colorant"white", stop=colorant"indigo", length=7);
    c_yellows = range(colorant"white", stop=colorant"gold", length=7);
    pals = [[c_purples[i] for i=3:3+length(thresholds)], [c_yellows[i] for i=3:3+length(thresholds)]]

    conc = [0.31, 1, 3.1, 10]; temp = [25, 27.5, 30, 32.5]
    titles = [L"\#\:of\:Strongly\:Connected\:Components", L"\#\:of\:Required\:Driver\:Species"]

    for i=1:2, j=1:2
        if i==1; pli = plot(xaxis=:log); title!(titles[j], titlefontsize=16); else; pli = plot(); end
        if j==1; xaxis!([L"Glucose\:(mM)", L"Temperature\:(C)"][i]); end

        pl_data = [n_scc, n_mds][j][((i-1)*4 + 1):i*4,:]
        # println(pl_data)
        # println(n_scc)
        pli = plot!([conc, temp][i], pl_data;
                        palette=pals[j],
                        # alpha=0.5,
                        yticks=collect(Int(minimum(pl_data)):Int(maximum(pl_data))),
                        xticks=([conc, temp][i], [conc, temp][i]),
                        labels=thresholds',
                        linewidth=2)

        push!(pls, pli)
    end

    scc_mds_plot = plot(pls..., layout=(2,2),
                        size=(1000,600),
                        # top_margin=-10Plots.px,
                        bottom_margin=1Plots.px);
    Btag = plot(annotate=(0, 0, ("B", :left, 20)), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, bottom_margin=10Plots.px, left_margin=-20Plots.px)
    # scc_mds_plot_B = plot(Btag, scc_mds_plot, layout=@layout([A{0.025h}; B]))

    graph_ims = Array{Plots.Plot{Plots.GRBackend},1}();
    for c=1:length(conds)
        plc = plot(load("fluc_graphs_$c.png"), annotate=(5000,6000,(conds[c],:left,10)), grid=false, axis=false, bordercolor="white", xticks=nothing, yticks=nothing)
        push!(graph_ims, plc)
    end

    Atag = plot(annotate= (0, 0, ("A", :left, 20)), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, left_margin=-20Plots.px)
    gluc_graph_plot = plot(graph_ims[1:4]..., layout=(1,4), size=(2000,400))
    # gluc_graph_plot_A = plot(Atag, gluc_graph_plot, layout=@layout([A{0.025h}; B]))

    Ctag = plot(annotate= (0, 0, ("C", :left, 20)), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, left_margin=-20Plots.px)
    temp_graph_plot = plot(graph_ims[5:8]..., layout=(1,4), size=(2000,400))
    # temp_graph_plot_C = plot(Ctag, temp_graph_plot, layout=@layout([A{0.025h}; B]))

    network_fluc = plot(Atag, gluc_graph_plot, Btag, scc_mds_plot, Ctag, temp_graph_plot;
                        layout = @layout([A{0.02h}; B{0.225h}; C{0.02h}; D; E{0.02h}; F{0.225h}]),
                        left_margin = -5Plots.px,
                        right_margin = -5Plots.px,
                        # top_margin = -10Plots.px,
                        # bottom_margin = 10Plots.px,
                        size=(1000,1000))

    return network_fluc
end

network_fluc_full = make_net_fluc_fig(n_full, p_full_all, thresholds, collect(1:7))
# network_fluc_3 = make_net_fluc_fig(Major_3...)
# # savefig(network_fluc_full, path_to_figs *"network_fluctuation_figure_full.pdf")
# # savefig(network_fluc_3, path_to_figs *"network_fluctuation_figure_3.pdf")


## make a gif for the internet lol

## Interaction Change Figure

n=n_full
pl = Array{Plots.Plot{Plots.GRBackend},1}()
pal = [colors[i] for i=1:n];
conc = [0.31,1.,3.1,10.]; temp = [25, 27.5, 30, 32.5]
labs = Array{String}(undef, size(p_full_all)[1])
for i = 1:4; labs[i] = "r"*string(i); end
c=n
for x=1:n;
    for y=1:n;
        labs[c] = "α"*string(x)*string(y);
        c+=1;
    end;
end

focus_orgs = [1,4,6] #collect(1:7) == all
# focus_orgs = collect(1:7)
c2inc = [1,2,3,4] #include all: [1,2,3,4], linear approx: [1,4]
min_Δp = 0.25

ptypes = [
    (L"Growth\:Rate\:(r)", collect(1:n), [0.25,1.0]),
    (L"Self\:Interaction\:(\alpha_{ii})", collect(n+1:n+1:n^2+n), [-5.0,0.0]),
    (L"Interaction\:(\alpha_{ij})", filter!(x->x∉collect(n+1:n+1:n^2+n), collect(n+1:n^2+n)), [-5.0,5.0])
]

for j = 1:2
    for pt in ptypes
        p_type, pix_range, p_bounds = pt[1], pt[2], pt[3]

        if j==1; p1 = plot(xaxis=:log); else; p1 = plot();end
        # p1 = plot()
        # xaxis!([L"Glucose\:(mM)", L"Temperature\:(C)"][j])
        if p_type == L"$Growth\:Rate\:(r)$"
            # yaxis!(L"Value");
            xaxis!([L"Glucose\:(mM)", L"Temperature\:(C)"][j])
        end
        if j == 1
            title!(p_type)
        end

        for i ∈ pix_range
            # only plotting parameters related to focus_orgs
            if i ∉ focus_orgs && Int(floor((i-1)/n)) ∉ focus_orgs; continue; end

            p_series = p_full_all[i, [1:4, 5:8][j]]
            if abs(p_series[1] - p_series[4])/abs(p_series[1]) > min_Δp; continue; end

            # line color based on species which parameter directly affects
            if i <= n; linecolor = pal[i]; else; linecolor = pal[Int(floor((i-1)/n))]; end

            plot!([conc, temp][j][c2inc], p_series[c2inc], linewidth = 2, linecolor=linecolor, xticks = ([conc, temp][j], [conc, temp][j]))

            # xpos = 12.1; ypos = p_series[4]
            # annotate!(xpos, ypos, text(labs[i], :right, 15))
        end

        ylims!(p_bounds...)
        push!(pl, p1)
    end
end

ΔpΔE_plt = plot(pl..., layout = (2,3), legend = false, size = (900,450), top_margin = 5Plots.px, bottom_margin = 5Plots.px)
# title = plot(title=p_file, grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, bottom_margin = -25Plots.px)
# spacer = plot(grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing)
# plt_w_spacer = plot(spacer, plt, layout = @layout([A{0.0001w} B]))

display(ΔpΔE_plt)
# savefig(ΔpΔE_plt, path_to_figs *"param_enviro_relationship_3major_mind1.0.pdf")
# # savefig(ΔpΔE_plt, path_to_figs *"param_enviro_relationship.pdf")

## mds

# function mds_glv(n, p, thr)
#     g = SimpleDiGraph(n);
#     c=1;d=1
#     for i=n+1:n^2 + n
#         # print('\n', "p[i]=",p[i],"   ")
#         if (abs(p[i]) >= thr) && ~(c == d);
#             # print(c,',',d);
#             add_edge!(g, c, d); end
#         d += 1
#         if d == 5; d = 1; c += 1; end
#     end
#     A = convert(SparseMatrixCSC{Float64,Int64}, adjacency_matrix(g))
#     mds = DriverSpecies(A)
#     return mds, g
# end
#
# thresh = 0.5
#
# mds_g0p31, gr_g0p31 = mds_glv(n, p_gluc_0p31mM, thresh)
# print("\n 0p31 mds: ", mds_g0p31,  "\n")
