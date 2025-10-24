using PathoSpread
using MCMCChains
using Serialization
using Glob


# --- SETTINGS ---
base     = "simulations/hippo_DIFF_EUCL"
outfile  = base*".jls"

# --- FIND ALL MATCHING CHAINS ---
dir = dirname(base)
pattern = basename(base) * "_C*.jls"
paths = sort(glob(pattern, dir))  # e.g. [".../DIFFGA_ANTERO_C1.jls", "…_C2.jls", …]

if isempty(paths)
    error("No files matching pattern $(joinpath(dir, pattern)) found.")
end

println("Found $(length(paths)) chain files:")
foreach(println, paths)

# --- LOAD ALL CHAINS ---
inference_list = [load_inference(p) for p in paths]

# --- MERGE CHAINS ---
merged = deepcopy(inference_list[1])
merged["chain"] = chainscat([inf["chain"] for inf in inference_list]...)

# Optionally merge WAIC- or log-likelihood–related fields if they exist
for key in ["waic_i", "log_likelihoods"]
    if haskey(merged, key)
        merged[key] = vcat([inf[key] for inf in inference_list if haskey(inf, key)]...)
    end
end

# --- SAVE MERGED RESULT ---
serialize(outfile, merged)
println("✅ Saved merged inference → $outfile")
