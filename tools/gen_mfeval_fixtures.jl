# Generate Magic Formula reference values with MFeval.jl (a port of Marco Furlan's MATLAB mfeval).
#
#     make fixtures-mfeval
#
# which runs, for each tyre, `cargo run --example tir_canonical` (a .tir file with every
# coefficient as autonomousim reads it, so MFeval's defaults for missing keys do not matter) and
#
#     julia --project=$MFEVAL_JL tools/gen_mfeval_fixtures.jl <canonical.tir> <name> <out.json>
#
# MFeval runs in useMode 221: no input limits or low-speed reductions, α* = tan α, γ* = sin γ,
# no turn slip. Points are random (fixed seed) over load, slip, slip angle, camber and, for
# MF 6.x, pressure; plus pure-slip sweeps. Consumed by crates/vehicles/tests/tire_fixtures.rs.

using MFeval
using Random

const FIELDS = [:Fx, :Fy, :Mx, :My, :Mz, :Kxk, :Kya, :mux, :muy, :t, :Mzr, :sigmax, :sigmay, :Re, :omega, :two_a]

function points(p, rng)
    fz0 = p.fnomin * p.lfzo
    v61 = p.metadata.fittyp in (61, 62)
    pts = NTuple{6,Float64}[]
    for _ in 1:600
        fz = fz0 * (0.2 + 1.6 * rand(rng))
        kappa = rand(rng) < 0.3 ? 0.0 : 0.6 * (2rand(rng) - 1) * rand(rng)
        alpha = rand(rng) < 0.3 ? 0.0 : 0.35 * (2rand(rng) - 1) * rand(rng)
        gamma = rand(rng) < 0.4 ? 0.0 : 0.1 * (2rand(rng) - 1)
        press = v61 ? p.nompres * (0.8 + 0.4 * rand(rng)) : p.inflpres
        push!(pts, (fz, kappa, alpha, gamma, 5.0 + 30.0 * rand(rng), press))
    end
    for k in range(-0.5, 0.5; length = 41)
        push!(pts, (fz0, k, 0.0, 0.0, p.longvl, p.inflpres))
    end
    for a in range(-0.3, 0.3; length = 41)
        push!(pts, (fz0, 0.0, a, 0.0, p.longvl, p.inflpres))
    end
    pts
end

function main(tir, name, out)
    p = read_tir(tir)
    rng = MersenneTwister(20260924)
    modes = MFModes(221)
    open(out, "w") do io
        println(io, "{\"generator\": \"tools/gen_mfeval_fixtures.jl (MFeval.jl, useMode 221)\",")
        println(io, " \"tyre\": \"$name\",")
        println(io, " \"points\": [")
        pts = points(p, rng)
        for (i, (fz, kappa, alpha, gamma, vx, press)) in enumerate(pts)
            r = mfeval(p, MFInputs(fz, kappa, alpha, gamma, 0.0, vx, press), modes)
            vals = join(("\"$(f)\": $(repr(getfield(r, f)))" for f in FIELDS), ", ")
            sep = i == length(pts) ? "" : ","
            println(io, "  {\"fz\": $(repr(fz)), \"kappa\": $(repr(kappa)), \"alpha\": $(repr(alpha)), ",
                    "\"gamma\": $(repr(gamma)), \"vx\": $(repr(vx)), \"pressure\": $(repr(press)), $vals}$sep")
        end
        println(io, " ]}")
    end
end

main(ARGS...)
