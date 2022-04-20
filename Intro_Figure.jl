### Introduction Figure
# willsharpless@berkeley.edu or ucsd.edu
# Jul 25, 2021

using Graphs
using JLD, Plots, LaTeXStrings, GraphPlot, Compose
using Cairo, Fontconfig, ImageMagick
n=7;
colors = palette(:tab10); pal = [colors[i] for i=1:n];
orgs = ["ra", "sk", "mp", "pk", "bm", "pa", "fg"]
actual = ["full", "pkpa", "rapk", "rapkpa", "skpkpa", "pkbmpa", "pkpafg", "mppkpa"]
comm_list = cat(orgs, actual; dims=1); filter!(x->x≠"full", comm_list)
train_ix_iso = [[findfirst(occursin.(orgs, i))] for i in orgs]
train_ix = [[findfirst(occursin.(orgs, i)) for i in [j[k:k+1] for k in 1:2:length(j)]] for j in actual if j != "full"]
training_data = load("training_data.jld", "data")
cd("/Users/willsharpless/Documents/Julia/arkin/graphs")

## Make Graphs for Figures

### Make subcomm graphs
for (ci, com) in enumerate(cat(train_ix_iso, train_ix, dims=1))
# for (ci, com) in enumerate(train_ix)
    # p1 = plot()
    G = DiGraph(length(com))
    for c=1:length(com), d=1:length(com)
        if length(com) > 1
            if c != d; add_edge!(G, c, d); end #self-loops look bad
        else
            add_edge!(G, c, d) #self-loops required for single node
        end
    end
    nodelabel = titlecase.(orgs[com])
    nodesize = [0.15 for c=1:length(com)]
    edgestrokec = length(com) == 1 ? RGBA(0.0,0.0,0.0,0) : colorant"lightgray"
    layout = length(com) == 1 ? circular_layout : circular_layout
    
    sc_graph = gplot(G;
                    linetype="curve",
                    layout=layout,
                    # nodelabel=nodelabel,
                    # edgestrokec = :black,
                    NODESIZE=nodesize,
                    EDGELINEWIDTH=1.5,
                    arrowlengthfrac=0.3,
                    edgestrokec=edgestrokec,
                    outangle=0.15,
                    nodefillc=[colors[i] for i∈com])
    display(sc_graph)
    
    draw(PNG(comm_list[ci]*"_graph.png", 16cm, 16cm, dpi=1000), sc_graph)
end

### Make full graph with bolded edges where trained
G = DiGraph(n)
for com in train_ix; for c∈com, d∈com; if c != d; add_edge!(G, c, d); end; end; end
trained_edges = adjacency_matrix(G)
n_trainedp = sum(trained_edges) + n + n # trained_pairs + self + mu
trained_edges_ix = findall(!iszero, trained_edges)
untrained_edges_ix = findall(iszero, trained_edges)

H = DiGraph(n); Elw = zeros((n,n))
for pr in trained_edges_ix; add_edge!(H, pr[1], pr[2]); Elw[pr] = 0.7; end
for pr in untrained_edges_ix; if pr[1] != pr[2]; add_edge!(H, pr[1], pr[2]); Elw[pr] = 0.1; end; end

elw = [Elw[i,j] for i=1:n for j=1:n if i != j]
full_graph = gplot(H;
                linetype="curve",
                layout=circular_layout,
                # layout=layout,
                # nodelabel=titlecase.(orgs),
                # nodestrokec = :black,
                NODESIZE=[0.1 for c=1:n],
                # NODESIZE=0.2*[0.1, 0.01, 0.01, 0.27, 0.01, 0.7, 0.01], #ra sized
                # EDGELINEWIDTH=0.7,
                edgelinewidth=elw,
                arrowlengthfrac=0.1,
                outangle=0.15,
                nodefillc=[colors[i] for i=1:n])
# display(full_graph)
draw(PNG("full_graph.png", 16cm, 16cm, dpi=1000), full_graph)

## Make and Combine all Subplots

#### Combine subcomm graphs
pair_plots = Array{Plots.Plot{Plots.GRBackend},1}();
for com in actual[2:n+1]
    pl = plot(load(com*"_graph.png"), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing)
    display(pl)
    push!(pair_plots, pl)
end
pair_plot = plot(pair_plots..., layout=@layout([A B; C D; E F; G]), size = (800,1200))
Btag = plot(annotate= (0, 0, ("B", :left, 30)), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, bottom_margin = -25Plots.px)
pair_plot_B = plot(Btag, pair_plot, layout=@layout([A{0.025h}; B]))

### Join with full comm graph
full_graph_im = load("full_graph.png")
full_plot = plot(full_graph_im, grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, size = (800,1200))
Ctag = plot(annotate= (0, 0, ("C", :left, 30)), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, bottom_margin = -25Plots.px)
full_plot_C = plot(Ctag, full_plot, layout=@layout([A{0.025h}; B]))

pair_n_full_plot = plot(pair_plot_B, full_plot_C, layout = @layout([A B]), size = (1600,1200))

### Make legend plot
orgs_nice = [L"- \: Rhizobium \: pusense",
             L"- \: Shinella \: kummerowiae",
             L"- \: Microbacterium \: phyllosphaerae",
             L"- \: Pseudomonas \: koreensis",
             L"- \: Bacillus \: megaterium",
             L"- \: Pantoea \: agglomerans",
             L"- \: Flavobacterium \: ginsengiterrae"]

rzns = [LaTeXString("(Induces Nodulation, Fixes Nitrogen [R1])"),
        LaTeXString("(Induces Nodulation, Fixes Nitrogen [R2])"),
        LaTeXString("(Produces PGP Volatiles [R3])"),
        LaTeXString("(Produces Many Antagonistic Biosurfactants,"),
        LaTeXString("(Produces Antagonistic Volatiles, "),
        LaTeXString("(Fast Growing (Occaisonal Pathogen), Antifungal,"),
        LaTeXString("(Increases Resistance to Salt Stress,")]

rzns_sl = ["", "", "",
        LaTeXString("Induces Pathogen Resistance [R4])"),
        LaTeXString("Induces Pathogen Resistance [R5])"),
        LaTeXString("Produces PGP Phytohormones [R6])"),
        LaTeXString("Phosphate Solubilizing [R7])")]

pal = [colors[i] for i=1:n];
lx, ly = [], []
for j=2.5:-2:0.5, i=1:2:7; if i != 7 || j!=0.5; push!(lx, i); push!(ly, j); end; end;
leg_plot = plot(palette = pal, size=(1600,200), legend=false, grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing)
for i=1:n;
    scatter!((lx[i], ly[i]), markershape = :square, markersize = 20, markerstrokewidth = 1.5, markerstrokecolor = :black, markerstrokealpha = 0.2);
    annotate!((lx[i]+0.15, ly[i], (orgs_nice[i], :left, 15)))
    annotate!((lx[i]+0.15, ly[i]-0.4, (rzns[i], :left, 8)))
    annotate!((lx[i]+0.15, ly[i]-0.62, (rzns_sl[i], :left, 8)))
end
plot!(xlim=[0.9,8.6], ylim=[0,3])
Atag = plot(annotate= (0, 0, ("A", :left, 30)), grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, bottom_margin = -25Plots.px)
leg_plot_A = plot(Atag, leg_plot, layout=@layout([A{0.025h}; B]))

IntroFig = plot(leg_plot_A, pair_n_full_plot, layout = @layout([A{0.125h}; B]), top_margin = 25Plots.px, size = (1600,1400))
savefig(IntroFig, "figs/Intro_Figure.png")
