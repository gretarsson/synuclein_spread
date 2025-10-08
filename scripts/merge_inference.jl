using PathoSpread
using MCMCChains
using Serialization

# --- SETTINGS ---
base     = "simulations/DIFFG_T2"
nchains  = 4
outfile  = "simulations/DIFFG_T-2.jls"

# --- LOAD ALL CHAINS ---
inference_list = Dict[]
for i in 1:nchains
    path = "$(base)_C$(i).jls"
    println("Loading $path ...")
    push!(inference_list, load_inference(path))
end

# --- MERGE CHAINS ---
merged = deepcopy(inference_list[1])
merged["chains"] = chainscat([inf["chain"] for inf in inference_list]...)

# Optionally merge WAIC- or log-likelihood–related fields if they exist
for key in ["waic_i", "log_likelihoods"]
    if haskey(merged, key)
        merged[key] = vcat([inf[key] for inf in inference_list if haskey(inf, key)]...)
    end
end

# --- SAVE MERGED RESULT ---
serialize(outfile, merged)
println("✅ Saved merged inference → $outfile")
