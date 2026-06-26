using SpacetimeSpectralPassExamples
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

# Error-vs-(column x) panel for the three series L = 1, 2, 4.
function panel(data, x, xlabel; xscale = :identity, xticks = :auto, ann = false, legend = false)
    ylims, yticks = yaxis(data[:, 4])
    p = plot(; xscale, yscale = :log10, xlabel, ylabel = L"\mathrm{max\ error\ at}\ t = %$(Int(T))",
             legend = legend ? :topright : false, frame = :box, grid = false, ylims, yticks,
             xticks, size = (560, 500), left_margin = 8Plots.mm, bottom_margin = 6Plots.mm, top_margin = 5Plots.mm)
    for L in (1, 2, 4)
        r = sort(findall(==(L), Int.(data[:, 1])), by = i -> data[i, 2])
        color, marker, line, label = style(L)
        plot!(p, data[r, x], max.(data[r, 4], FLOOR); marker, color, line, linewidth = 2, label)
        ann && L != 2 && for i in r
            annotate!(p, data[i, x], max(data[i, 4], FLOOR) / 1.8, text(string(Int(data[i, 2])), 12, color, :center, :top))
        end
    end
    p
end

function summary(data, stem)
    nt = Int.(data[:, 2]); xt = collect(minimum(nt):16:maximum(nt))
    savefig(panel(data, 2, L"\mathrm{time\ coefficients}\ N_t\ \mathrm{(per\ block)}"; xticks = (xt, string.(xt)), legend = true), joinpath(FIGDIR, "$stem-convergence.pdf"))
    savefig(panel(data, 3, L"\mathrm{median\ solve\ time\ (ms)}"; xscale = :log10, ann = true), joinpath(FIGDIR, "$stem-timing.pdf"))
    savefig(panel(data, 5, L"\mathrm{resident\ floats}"; xscale = :log10), joinpath(FIGDIR, "$stem-memory.pdf"))
end

function operator_spy(name, A)
    r, c, _ = findnz(sparse(A))
    scatter(c, r; markersize = 2.4, markerstrokewidth = 0, markercolor = :black, legend = false,
            xlims = (0.5, size(A, 2) + 0.5), ylims = (0.5, size(A, 1) + 0.5), yflip = true,
            frame = :box, grid = false, size = (760, 620))
    savefig(joinpath(FIGDIR, name))
end
# ─────────────────────────────────────────────────────────────────────────────

# Manufactured u = g(t) sin(π x); forcing f = u_t - u_xx.
g(t) = 1 + 0.25sin(t)
forcing_time(t) = π^2 * g(t) + 0.25cos(t)
exact(t, x) = g(t) * sinpi(x)

T = 10.0
Nx = 24
Nts_by_L = ((1, [8, 12, 16, 24, 32, 48, 64]),
            (2, [6, 8, 10, 12, 16, 24, 32, 48]),
            (4, [4, 6, 8, 10, 12, 16, 24, 32]))

data = cached("heat1p1-t10-forced-lumem-summary.csv", ["L", "Nt", "time_ms", "error", "resident_floats"]) do
    heat_experiment_data(Nts_by_L; Nx = Nx, T = T, time_profile = g, forcing_time = forcing_time, exact = exact)
end

operator_spy("operator-heat-spy.pdf", heat_problem_rect(1.0, 16, 16).A)
summary(data, "heat1p1")
println("Heat (1+1)D figures written to ", FIGDIR)
