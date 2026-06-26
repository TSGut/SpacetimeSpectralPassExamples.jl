using SpacetimeSpectralPassExamples
using Plots, DelimitedFiles, LaTeXStrings

# ─── figures ───
gr()
default(guidefontsize = 15, tickfontsize = 14, legendfontsize = 13, titlefontsize = 15)
FIGDIR = joinpath(@__DIR__, "Figures"); mkpath(joinpath(FIGDIR, "data"))
recompute = lowercase(get(ENV, "RECOMPUTE_FIGURE_DATA", "0")) in ("1", "true", "yes")
FLOOR = 1e-14
save(p, name) = savefig(p, joinpath(FIGDIR, name))
function cached(compute, name, header)
    path = joinpath(FIGDIR, "data", name)
    (!recompute && isfile(path)) && return readdlm(path, ',', Float64; skipstart = 1)
    d = compute()
    open(io -> (println(io, join(header, ",")); writedlm(io, d, ',')), path, "w")
    d
end
# ───────────────

DEEP_T   = 1000.0
WAVE_MODE = 2
KG_NX    = 20
KG_M2    = 20.0

# Block counts and the time-coefficient sweep per L.
BLOCK_NTS = Dict(50  => [64, 80, 96, 112, 128, 144],       # dt=20
                 100 => [32, 40, 48, 56, 64, 72, 80, 96])  # dt=10
BLOCK_LS = (50, 100)
# Converged N_t per block for the stability panel.
ERRCURVE_NT = Dict(50 => 144, 100 => 80)

style(L) = L == 50  ? (:royalblue, :square,  :dash, "L=50 (blocks)") :
                      (:purple,    :diamond, :dot,  "L=100 (blocks)")

ukg(t, x) = cos(sqrt((WAVE_MODE * π)^2 + KG_M2) * t) * sinpi(WAVE_MODE * x)
wave_initial_coeffs(N) = spacecoeffs(x -> sinpi(WAVE_MODE * x), N)

# Sweep over Nt: rows [L, Nt, time_ms, error, resident_floats].
function deep_kg_sweep(L, Nts)
    rows = Float64[]
    Δt = DEEP_T / L
    u0 = wave_initial_coeffs(KG_NX); v0 = zeros(KG_NX)
    for Nt in Nts
        t_ms = benchmark_ms(() -> stepsolve_wave_rect(u0, v0, Δt, L, Nt, KG_NX; m2 = KG_M2))
        uT, _ = stepsolve_wave_rect(u0, v0, Δt, L, Nt, KG_NX; m2 = KG_M2)
        err = maximum(abs(evalspace(uT, x) - ukg(DEEP_T, x)) for x in GRIDX)
        append!(rows, (L, Nt, t_ms, err, wave_memory(Nt, KG_NX, Δt; m2 = KG_M2)))
    end
    reshape(rows, 5, :)'
end

# Per-block error of the passed slice vs time (the stability test).
function deep_kg_error_curve(L, Nt)
    Δt = DEEP_T / L
    prob = wave_problem_rect(Δt, Nt, KG_NX; m2 = KG_M2)
    u = collect(float.(wave_initial_coeffs(KG_NX))); v = zeros(KG_NX)
    rows = Float64[]
    for ell in 1:L
        waveblock_rect!(prob.U, prob.V, prob, u, v)
        peel!(u, prob.U); peel!(v, prob.V)
        t = ell * Δt
        append!(rows, (t, maximum(abs(evalspace(u, x) - ukg(t, x)) for x in GRIDX)))
    end
    reshape(rows, 2, :)'
end

# Compute (cached).
tag = "kleingordon-deeptime-T1000-nx20"
sweep_header = ["L", "Nt", "time_ms", "error", "resident_floats"]
block50_data  = cached(() -> deep_kg_sweep(BLOCK_LS[1], BLOCK_NTS[50]), "$tag-L50.csv", sweep_header)
block100_data = cached(() -> deep_kg_sweep(BLOCK_LS[2], BLOCK_NTS[100]), "$tag-L100.csv", sweep_header)
errcurve50  = cached(() -> deep_kg_error_curve(BLOCK_LS[1], ERRCURVE_NT[50]), "$tag-errcurve-L50.csv", ["t", "error"])
errcurve100 = cached(() -> deep_kg_error_curve(BLOCK_LS[2], ERRCURVE_NT[100]), "$tag-errcurve-L100.csv", ["t", "error"])

# Plots.
clamp_err(v) = max.(v, FLOOR)

function error_yaxis(allerr; clamp_top = true)
    raw = log10(maximum(clamp_err(allerr)))
    top = 2 * ceil(Int, (clamp_top ? max(0.0, raw) : raw) / 2)
    exps = collect(top:-2:-14)
    ((FLOOR, 10.0^(top + 0.25)), (10.0 .^ exps, ["10^{$e}" for e in exps]))
end

function decade_ticks(vals)
    positive = vals[vals .> 0]
    exps = collect(floor(Int, log10(minimum(positive))):ceil(Int, log10(maximum(positive))))
    (10.0 .^ exps, ["10^{$e}" for e in exps])
end

ylims, yticks = error_yaxis(vcat(block50_data[:, 4], block100_data[:, 4]))
PANEL = (560, 500)
BSERIES = ((50, block50_data), (100, block100_data))

function add_series!(p, L, data, xc)
    color, marker, ls, label = style(L)
    o = sortperm(data[:, xc])
    plot!(p, data[o, xc], clamp_err(data[o, 4]), marker = marker, color = color,
          line = ls, linewidth = 2, label = label)
end

# (a) convergence
pa = plot(yscale = :log10, xlabel = L"\mathrm{time\ coefficients}\ N_t\ \mathrm{(per\ block)}",
          ylabel = L"\mathrm{max\ error\ at}\ t = %$(Int(DEEP_T))", legend = :topright, frame = :box,
          grid = false, ylims = ylims, yticks = yticks, size = PANEL,
          left_margin = 8Plots.mm, bottom_margin = 6Plots.mm, top_margin = 5Plots.mm)
for (L, data) in BSERIES; add_series!(pa, L, data, 2); end

# (b) timing
pb = plot(xscale = :log10, yscale = :log10, xlabel = L"\mathrm{median\ solve\ time\ (ms)}", ylabel = "",
          legend = false, frame = :box, grid = false, ylims = ylims, yticks = yticks,
          xticks = decade_ticks(vcat(block50_data[:, 3], block100_data[:, 3])),
          size = PANEL, left_margin = 8Plots.mm, bottom_margin = 6Plots.mm, top_margin = 5Plots.mm)
for (L, data) in BSERIES; add_series!(pb, L, data, 3); end
for (L, data) in BSERIES
    color, _, _, _ = style(L)
    for r in axes(data, 1)
        annotate!(pb, data[r, 3], max(data[r, 4], FLOOR) / 1.8, text(string(Int(data[r, 2])), 12, color, :center, :top))
    end
end

# (c) memory
pc = plot(xscale = :log10, yscale = :log10, xlabel = L"\mathrm{resident\ floats}", ylabel = "",
          legend = false, frame = :box, grid = false, ylims = ylims, yticks = yticks,
          xticks = decade_ticks(vcat(block50_data[:, 5], block100_data[:, 5])),
          size = PANEL, left_margin = 8Plots.mm, bottom_margin = 6Plots.mm, top_margin = 5Plots.mm)
for (L, data) in BSERIES; add_series!(pc, L, data, 5); end

# (d) stability
sty_ylims, sty_yticks = error_yaxis(vcat(errcurve50[:, 2], errcurve100[:, 2]); clamp_top = false)
pd = plot(yscale = :log10, xlabel = L"\mathrm{time}\ t", ylabel = L"\mathrm{max\ error\ at\ block\ final\ time}",
          legend = :bottomright, frame = :box, grid = false, ylims = sty_ylims, yticks = sty_yticks,
          size = PANEL, left_margin = 8Plots.mm, bottom_margin = 6Plots.mm, top_margin = 5Plots.mm,
          right_margin = 6Plots.mm)
for (L, data) in ((50, errcurve50), (100, errcurve100))
    color, _, ls, _ = style(L)
    plot!(pd, data[:, 1], clamp_err(data[:, 2]), color = color, line = ls, linewidth = 2,
          label = L"L = %$(L)\ (N_t = %$(ERRCURVE_NT[L]))")
end

save(pa, "kleingordon-deeptime-convergence.pdf")
save(pb, "kleingordon-deeptime-timing.pdf")
save(pc, "kleingordon-deeptime-memory.pdf")
save(pd, "kleingordon-deeptime-stability.pdf")

println("Deep-time KG panels written to ", FIGDIR)
println("L=100 final-time error range: ", extrema(block100_data[:, 4]))
println("L=50  final-time error range: ", extrema(block50_data[:, 4]))
println("error-vs-time finals (should be near floor): L50=", errcurve50[end, 2], " L100=", errcurve100[end, 2])
