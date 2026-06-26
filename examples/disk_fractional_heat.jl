using SpacetimeSpectralPassExamples
using StaticArrays
using Plots, DelimitedFiles, SparseArrays, LaTeXStrings

# ─── figures ─────────────────────────────────────────────────────────────────
gr()
default(guidefontsize = 15, tickfontsize = 14, legendfontsize = 13, titlefontsize = 15)
FIGDIR = joinpath(@__DIR__, "Figures"); mkpath(joinpath(FIGDIR, "data"))
recompute = lowercase(get(ENV, "RECOMPUTE_FIGURE_DATA", "0")) in ("1", "true", "yes")
FLOOR = 1e-14

function cached(compute, name, header)
    path = joinpath(FIGDIR, "data", name)
    (!recompute && isfile(path)) && return readdlm(path, ',', Float64; skipstart = 1)
    d = compute()
    open(io -> (println(io, join(header, ",")); writedlm(io, d, ',')), path, "w")
    d
end

style(L) = (Dict(1 => :black, 2 => :royalblue, 4 => :purple)[L],
            Dict(1 => :circle, 2 => :square, 4 => :diamond)[L],
            Dict(1 => :solid, 2 => :dash, 4 => :dot)[L],
            L == 1 ? "L=1 (global)" : "L=$L (blocks)")

function yaxis(vals)
    top = 2ceil(Int, max(0.0, log10(maximum(max.(vals, FLOOR)))) / 2)
    e = top:-2:-14
    ((5e-15, 10.0^(top + 0.25)), (10.0 .^ e, ["10^{$x}" for x in e]))
end
intticks(n) = (t = collect(Int(round(minimum(n))):8:Int(round(maximum(n)))); (t, string.(t)))

# x is the data column (2 = N, 3 = time, 5 = memory); error is column 4.
function panel(data, x, xlabel; xscale = :identity, xticks = :auto, ann = false, legend = false, ylabel = L"\mathrm{max\ error\ at}\ t = %$(Int(T))")
    ylims, yticks = yaxis(data[:, 4])
    p = plot(; xscale, yscale = :log10, xlabel, ylabel, legend = legend ? :topright : false,
             frame = :box, grid = false, ylims, yticks, xticks, top_margin = 5Plots.mm)
    for L in (1, 2, 4)
        r = sort(findall(==(L), Int.(data[:, 1])), by = i -> data[i, 2])
        color, marker, line, label = style(L)
        plot!(p, data[r, x], max.(data[r, 4], FLOOR); marker, color, line, linewidth = 2, label)
        ann && L != 2 && for i in r
            data[i, 4] <= 1e-12 && continue
            annotate!(p, data[i, x], max(data[i, 4], FLOOR) / 1.8, text(string(Int(data[i, 2])), 12, color, :center, :top))
        end
    end
    p
end

function disk_summary(d1, dhalf, figname)
    row(d, lab, leg) = (panel(d, 2, L"\mathrm{time\ coefficients}\ N_t\ \mathrm{(per\ block)}"; xticks = intticks(d[:, 2]), legend = leg, ylabel = lab * "\n" * L"\mathrm{max\ error\ at}\ t = %$(Int(T))"),
                        panel(d, 3, L"\mathrm{median\ solve\ time\ (ms)}"; xscale = :log10, ann = true, ylabel = ""),
                        panel(d, 5, L"\mathrm{resident\ floats}"; xscale = :log10, ylabel = ""))
    plot(row(d1, L"\beta = 1", true)..., row(dhalf, L"\beta = 1/2", false)...,
         layout = @layout([a b c; d e f]), size = (1280, 760),
         left_margin = 8Plots.mm, bottom_margin = 6Plots.mm, top_margin = 6Plots.mm)
    savefig(joinpath(FIGDIR, figname))
end

function spy_scatter(A; title = "")
    r, c, _ = findnz(sparse(A))
    scatter(c, r; markersize = 2.4, markerstrokewidth = 0, markercolor = :black, legend = false,
            xlims = (0.5, maximum(c) + 0.5), ylims = (0.5, maximum(r) + 0.5), yflip = true,
            frame = :box, grid = false, title)
end
operator_spy(name, A) = (spy_scatter(A; ); savefig(joinpath(FIGDIR, name)))
# ─────────────────────────────────────────────────────────────────────────────

nblocks = 14                       # complete triangular Zernike truncation
K = zernike_ncoeffs(nblocks)       # N_Z = 105 spatial coefficients
T = 10.0
Ns = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48]   # time-coefficient sweep
Ls = (1, 2, 4)

# Manufactured u = g_β(t) Φ(x,y); g_β(t) = exp(-(4+2β) t), Φ sparse in Zernike.
phi = let c = zeros(K)
    active = (1, 2, 4, 7, 11, 16, 22, 29, 37, 46, 56, 67, 79, 88, 100)
    for (i, v) in zip(active, (1.0, -0.20, 0.12, -0.08, 0.05, 0.035, -0.030, 0.026,
                               -0.022, 0.018, -0.015, 0.012, -0.010, 0.008, -0.006))
        c[i] = v
    end
    c
end
g(beta, t)  = exp(-(4 + 2beta) * t)
gdot(beta, t) = -(4 + 2beta) * g(beta, t)
space(beta, xy) = evaldiskblock(reshape(phi, 1, :), 1.0, 0.0, xy; beta = beta)
exact(beta, t, xy) = g(beta, t) * space(beta, xy)

sample_points = [SVector(x, y) for x in range(-0.95, 0.95, 31), y in range(-0.95, 0.95, 31) if x^2 + y^2 <= 0.95^2]

# Forcing f = u_t + (-Δ)^β u, expanded in the C^{3/2} time basis, kept rows only.
function rhs_blocks(beta, prob, L)
    C, Lop = disk_operators_cached(nblocks; beta = beta)
    [begin
         t0 = (ell - 1) * prob.Δt
         Fdt = timecoeffs_basis(U32, t -> gdot(beta, t), t0, prob.Δt, prob.Nt)
         Ft  = timecoeffs_basis(U32, t -> g(beta, t),    t0, prob.Δt, prob.Nt)
         vec(Fdt * transpose(C * phi) + Ft * transpose(Lop * phi))[prob.keep]
     end for ell in 1:L]
end

final_error(beta, U, dt) =
    maximum(abs(evaldiskblock(U, dt, dt, xy; beta = beta) - exact(beta, T, xy)) for xy in sample_points)

function summary_data(beta)
    rows = Float64[]
    u0c = g(beta, 0.0) .* phi
    for L in Ls
        dt = T / L
        for N in Ns
            prob = disk_problem_sparse(beta, N, dt, nblocks)
            blocks = rhs_blocks(beta, prob, L)
            t_ms = benchmark_ms(() -> disk_solve_from_sparse(prob, u0c, blocks))
            U = disk_solve_from_sparse(prob, u0c, blocks)
            append!(rows, (L, N, t_ms, final_error(beta, U, dt), disk_memory(prob)))
        end
    end
    reshape(rows, 5, :)'
end

beta1 = cached("diskfractionalheat-zernike-summary-k100-T10-decay-beta1-droptol1e-15-sparselu-N48.csv",
               ["L", "N", "time_ms", "error", "resident_floats"]) do
    summary_data(1.0)
end
beta0p5 = cached("diskfractionalheat-zernike-summary-k100-T10-decay-beta0p5-droptol1e-15-sparselu-N48.csv",
                 ["L", "N", "time_ms", "error", "resident_floats"]) do
    summary_data(0.5)
end

operator_spy("operator-diskfractionalheat-zernike-spy-beta1.pdf", disk_operator_matrix(8, nblocks; beta = 1.0))
operator_spy("operator-diskfractionalheat-zernike-spy-beta0p5.pdf", disk_operator_matrix(8, nblocks; beta = 0.5))
disk_summary(beta1, beta0p5, "diskfractionalheat-zernike-summary.pdf")
plot(spy_scatter(disk_operator_matrix(8, nblocks; beta = 1.0); title = "β = 1"),
     spy_scatter(disk_operator_matrix(8, nblocks; beta = 0.5); title = "β = 1/2"),
     layout = (1, 2), size = (1080, 420))
savefig(joinpath(FIGDIR, "operator-diskfractionalheat-zernike-spy.pdf"))
println("Disk fractional heat figures written to ", FIGDIR)
