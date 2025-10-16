using CSV, DataFrames

# ───────────────────────────────────────────────
# File paths
# ───────────────────────────────────────────────
indir = "data"
copath_file = joinpath(indir, "hippocampal/CoPathQuant_pSyn.csv")
connectome_file = joinpath(indir, "W_labeled_filtered.csv")

# ───────────────────────────────────────────────
# Load datasets
# ───────────────────────────────────────────────
df_copath = CSV.read(copath_file, DataFrame)
W_labels = CSV.File(connectome_file) |> DataFrame

# ───────────────────────────────────────────────
# Extract all brain region names in CoPathQuant data
# (skip metadata columns)
# ───────────────────────────────────────────────
exclude = [:mouse, :mpi, :treatment]
region_cols = setdiff(names(df_copath), exclude)

# In CoPath, regions appear as 'iREGION' and 'cREGION'
# → Strip the hemisphere prefixes to get pure region names
function strip_prefix(region::Symbol)
    name = String(region)
    if startswith(name, "i") || startswith(name, "c")
        return name[2:end]  # remove hemisphere prefix
    else
        return name
    end
end

copath_regions = unique(strip_prefix.(region_cols))

# ───────────────────────────────────────────────
# Extract region labels from structural connectome
# (first row / column header — assumed identical)
# ───────────────────────────────────────────────
# The labels are typically in the first column and header row
connectome_regions = String.(names(W_labels)[2:end])  # skip the index column

# ───────────────────────────────────────────────
# Compare region sets
# ───────────────────────────────────────────────
missing_in_connectome = setdiff(copath_regions, connectome_regions)
extra_in_connectome   = setdiff(connectome_regions, copath_regions)
common_regions        = intersect(copath_regions, connectome_regions)

println("────────────── Region Mapping Summary ──────────────")
println("Regions in CoPathQuant: ", length(copath_regions))
println("Regions in connectome : ", length(connectome_regions))
println("Regions in both        : ", length(common_regions))
println("Missing in connectome  : ", length(missing_in_connectome))
println("Extra in connectome    : ", length(extra_in_connectome))
println()

if !isempty(missing_in_connectome)
    println("⚠️  Regions present in CoPathQuant but NOT in connectome:")
    println(join(sort(missing_in_connectome), ", "))
end

if !isempty(extra_in_connectome)
    println("\n⚠️  Regions present in connectome but NOT in CoPathQuant:")
    println(join(sort(extra_in_connectome), ", "))
end

# ───────────────────────────────────────────────
# Optional: inspect parent–daughter naming structure
# ───────────────────────────────────────────────
# If region labels contain hyphens, dots, or hierarchical identifiers,
# you can parse and analyze them:
parent_candidates = filter(r -> occursin('-', r), copath_regions)
if !isempty(parent_candidates)
    println("\nRegions with potential parent–daughter naming:")
    println(join(sort(parent_candidates), ", "))
end
