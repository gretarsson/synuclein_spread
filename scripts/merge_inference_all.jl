#!/usr/bin/env julia
using PathoSpread
using MCMCChains
using Serialization
using Glob
using Printf

# ============================================================
# SETTINGS
# ============================================================
basedir = "simulations_mean"  # directory to search in
pattern = joinpath(basedir, "*_C*.jls")

# ============================================================
# FIND BASE FILE GROUPS
# ============================================================
paths = sort(glob(pattern))
if isempty(paths)
    error("No files matching $pattern found.")
end

# Group by base (everything before '_C#')
function base_name(path)
    m = match(r"^(.*)_C\d+\.jls$", path)
    return m === nothing ? nothing : m.captures[1]
end

bases = unique(filter(!isnothing, base_name.(paths)))

println("🔍 Found $(length(bases)) experiment bases:")
foreach(println, bases)

# ============================================================
# MERGE EACH EXPERIMENT’S CHAINS
# ============================================================
for base in bases
    chain_files = sort(glob(basename(base)*"_C*.jls", dirname(base)))
    outfile = base * ".jls"

    println("\n▶️  Merging $(length(chain_files)) chains for $(basename(base))...")
    inference_list = [load_inference(p) for p in chain_files]

    merged = deepcopy(inference_list[1])
    merged["chain"] = chainscat([inf["chain"] for inf in inference_list]...)

    # Merge optional fields
    for key in ["waic_i", "log_likelihoods"]
        if any(haskey(inf, key) for inf in inference_list)
            merged[key] = vcat([inf[key] for inf in inference_list if haskey(inf, key)]...)
        end
    end

    serialize(outfile, merged)
    @printf("✅ Saved merged inference → %s\n", outfile)
end

println("\n🎉 Done merging all experiments!")
