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
linear_lims(v; pad = 0.04) = (lo = minimum(v); hi = maximum(v); s = hi - lo; s == 0 ? (lo - 1, hi + 1) : (lo - pad * s, hi + pad * s))
log_lims(v; pad = 0.08) = (p = max.(v, FLOOR); (lo, hi) = extrema(log10.(p)); s = hi - lo; s == 0 ? (10.0^(lo - 1), 10.0^(hi + 1)) : (10.0^(lo - pad * s), 10.0^(hi + pad * s)))
intticks(nt) = (t = collect(Int(round(minimum(nt))):16:Int(round(maximum(nt)))); (t, string.(t)))

function panel(data, x, xlabel; xscale = :identity, xticks = :auto, ann = false, legend = false,
               yax = yaxis(data[:, 4]), xlims = nothing, ylabel = L"\mathrm{max\ error\ at}\ t = %$(Int(T))")
    ylims, yticks = yax
    p = plot(; xscale, yscale = :log10, xlabel, ylabel, legend = legend ? :topright : false,
             frame = :box, grid = false, ylims, yticks, xticks, size = (560, 500),
             left_margin = 8Plots.mm, bottom_margin = 6Plots.mm, top_margin = 5Plots.mm)
    xlims !== nothing && plot!(p; xlims)
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
    savefig(panel(data, 2, L"\mathrm{time\ coefficients}\ N_t\ \mathrm{(per\ block)}"; xticks = intticks(data[:, 2]), legend = true), joinpath(FIGDIR, "$stem-convergence.pdf"))
    savefig(panel(data, 3, L"\mathrm{median\ solve\ time\ (ms)}"; xscale = :log10, ann = true), joinpath(FIGDIR, "$stem-timing.pdf"))
    savefig(panel(data, 5, L"\mathrm{resident\ floats}"; xscale = :log10), joinpath(FIGDIR, "$stem-memory.pdf"))
end

# Two-row comparison: derivative peeling (top) vs first-order system (bottom).
function comparison(derivative, system, stem)
    comb = vcat(derivative, system)
    yax = yaxis(comb[:, 4]); dx = linear_lims(comb[:, 2]); xt = intticks(comb[:, 2])
    tx = log_lims(comb[:, 3]); mx = log_lims(comb[:, 5])
    row(d, lab, leg) = (panel(d, 2, L"\mathrm{time\ coefficients}\ N_t\ \mathrm{(per\ block)}"; xticks = xt, xlims = dx, yax, legend = leg, ylabel = lab * "\n" * L"\mathrm{max\ error\ at}\ t = %$(Int(T))"),
                        panel(d, 3, L"\mathrm{median\ solve\ time\ (ms)}"; xscale = :log10, xlims = tx, yax, ann = true, ylabel = ""),
                        panel(d, 5, L"\mathrm{resident\ floats}"; xscale = :log10, xlims = mx, yax, ylabel = ""))
    plot(row(derivative, L"\mathrm{second\ order\ problem}", true)..., row(system, L"\mathrm{first\ order\ system}", false)...,
         layout = @layout([a b c; d e f]), size = (1280, 760),
         left_margin = 8Plots.mm, bottom_margin = 6Plots.mm, top_margin = 6Plots.mm)
    savefig(joinpath(FIGDIR, "$stem-comparison.pdf"))
end

function operator_spy(name, A)
    r, c, _ = findnz(sparse(A))
    scatter(c, r; markersize = 2.4, markerstrokewidth = 0, markercolor = :black, legend = false,
            xlims = (0.5, size(A, 2) + 0.5), ylims = (0.5, size(A, 1) + 0.5), yflip = true,
            frame = :box, grid = false, size = (760, 620))
    savefig(joinpath(FIGDIR, name))
end
# ─────────────────────────────────────────────────────────────────────────────

mode = 2
m2 = 20.0
T = 10.0
Nx = 32
exact(t, x) = cos(sqrt((mode * π)^2 + m2) * t) * sinpi(mode * x)
init_profile(x) = sinpi(mode * x)
Nts_by_L = ((1, [16, 24, 32, 48, 64, 80, 96, 112]),
            (2, [12, 16, 24, 32, 48, 64]),
            (4, [8, 12, 16, 24, 32, 48]))

derivative = cached("kleingordon-t10-mode2-m2pos20-nx32-derivative-lumem-summary.csv", ["L", "Nt", "time_ms", "error", "resident_floats"]) do
    wave_scalar_experiment_data(Nts_by_L; m2, exact, Nx, T, init_profile)
end
system = cached("kleingordon-t10-mode2-m2pos20-nx32-firstorder-lumem-summary.csv", ["L", "Nt", "time_ms", "error", "resident_floats"]) do
    wave_experiment_data(Nts_by_L; m2, exact, Nx, T, init_profile)
end

operator_spy("operator-klein-gordon-m2=20-spy.pdf", wave_problem_rect(1.0, 16, 16; m2 = m2).A)
summary(system, "kleingordon")
comparison(derivative, system, "kleingordon")
println("Klein-Gordon figures written to ", FIGDIR)
