### Change in Fixed Point and Stability
# willsharpless@berkeley.edu or ucsd.edu
# Jul 25, 2021

using JLD, Plots, LightGraphs, LaTeXStrings, GraphPlot, Compose, SparseArrays
using LinearAlgebra, Statistics
using Cairo, Fontconfig, ImageMagick, Colors

my_modules_path = "/Users/willsharpless/Documents/Julia/modules/"
push!(LOAD_PATH, my_modules_path)
using Glv: pixs

function gLV_jac(x, p)
    n = size(x)[1]
    r = p[1:n];
    A = Array{eltype(x),2}(undef,n,n)
    for i=1:n; A[i,:] = p[n*i + 1: n*(i+1)]; end
    J = Diagonal(r + A*x) + Diagonal(x)*A
    return J
end

to_arkin = "/Users/willsharpless/Documents/Julia/arkin/"
p_full_all = load(to_arkin * "p_full_all_bestish.jld", "data")
full_STAT = load(to_arkin * "Full_Comm_statenv_all.jld", "data")

orgs = ["ra", "sk", "mp", "pk", "bm", "pa", "fg"]
conds = [L"0.31\:mM", L"1\:mM", L"3.1\:mM", L"10\:mM", L"25^{\circ} C", L"27.5^{\circ} C", L"30^{\circ} C", L"32.5^{\circ} C"]

n = 3; n_full = 7

dims_sub = [1,4,6]

_, pix_focus_iso, pix_focus_int = pixs(n_full, [[i] for i in dims_sub], dims_sub)
plix_full = BitVector(pix_focus_iso + pix_focus_int)
p_full_all_rapkpa = p_full_all[plix_full, :]

dims_sub = [1,6]
_, pix_focus_iso, pix_focus_int = pixs(n_full, [[i] for i in dims_sub], dims_sub)
plix_full = BitVector(pix_focus_iso + pix_focus_int)
p_full_all_rapa = p_full_all[plix_full, :]

dims_sub = [4,6]
_, pix_focus_iso, pix_focus_int = pixs(n_full, [[i] for i in dims_sub], dims_sub)
plix_full = BitVector(pix_focus_iso + pix_focus_int)
p_full_all_pkpa = p_full_all[plix_full, :]

dims_sub = [6]
_, pix_focus_iso, pix_focus_int = pixs(n_full, [[i] for i in dims_sub], dims_sub)
plix_full = BitVector(pix_focus_iso + pix_focus_int)
p_full_all_pa = p_full_all[plix_full, :]

# dims_load = collect.(unique([Set([i, j]) for i in dims_train for j in dims_train if i != j]))
# plix_subcomm, plix_full, ptix_full = pixs(n_full, dims_load, dims_sub)
# p_full_all_rapkpa = p_full_all[plix_full, :]

## Dominating Fixed Point Prediction vs. Reality

f_p_mean = zeros(n, 8)# mean final point of real timecourses
f_p_std = zeros(n, 8)# std

fp = zeros(n, 8) # fixed points of gLV systems
fp_eig = zeros(n, 8) # eig of gLV_jac at fp

## Compute mean observed point
for (io, o) ∈ enumerate([1,4,6])
    org_mask = [i*n_full + o for i=0:2]
    for c = 1:8
        f_p_org = full_STAT[org_mask,end,c]
        f_p_mean[io, c], f_p_std[io, c] = mean(f_p_org), std(f_p_org)
    end
end

## Plot final point in simulation after 24 hours and observed fixed point after 24 hours?
# is this really necessary?

## Compute ROC of undiluted dominating fp over ensemble

# import simulated fixed points from no more passaging




# for c = 1:8
#     if c ∈ [1, 2]
#         n = 2
#         dims_sub = [2,3]
#         p = p_full_all_pkpa[:,c]
#     elseif c == 8
#         n = 2
#         dims_sub = [1,3]
#         p = p_full_all_rapa[:,c]
#     elseif c == 4
#         n = 1
#         dims_sub = [3]
#         p = p_full_all_pa[:,c]
#     else # c == 5 might be something else
#         n = 3
#         dims_sub = [1,2,3]
#         p = p_full_all_rapkpa[:,c]
#     end

#     r = p[1:n];
#     A = Array{eltype(p),2}(undef,n,n)
#     for i=1:n; A[i,:] = p[n*i + 1: n*(i+1)]; end

#     fp[dims_sub, c] = -inv(A)*r
#     J = gLV_jac(fp[dims_sub, c], p)

#     # fp_eig[:,c] = eigen(J).values
#     println("c $c with ", eigvals(J))
#     # return J
#     # break
# end
