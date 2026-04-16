#!/usr/bin/env julia

using CSV
using DataFrames
using Statistics

# ============================================================
# SETTINGS
# ============================================================

# input_csv   = "data/hippocampal/hippocampal_syn_only.csv"
input_csv   = "data/total_path.csv"
adj_csv     = "data/W_labeled_filtered.csv"
output_dir  = "Results"
output_txt  = joinpath(output_dir, "regions_sorted_by_pathology.txt")

mkpath(output_dir)

# ============================================================
# HELPERS
# ============================================================

"""
Convert a value to Float64 if possible, otherwise return missing.

Handles:
- numbers
- strings like "4", "0.001", "1e-5"
- blanks / NA / NaN-like strings
"""
function to_float_or_missing(x)
    if x === missing
        return missing
    elseif x isa Number
        return Float64(x)
    elseif x isa AbstractString
        s = strip(x)
        if isempty(s) || lowercase(s) in ("na", "nan", "missing", "null")
            return missing
        end
        y = tryparse(Float64, s)
        return y === nothing ? missing : y
    else
        return missing
    end
end

"""
Mean ignoring missing values.
Returns missing if all entries are missing.
"""
function safe_mean(x)
    y = collect(skipmissing(x))
    isempty(y) && return missing
    return mean(y)
end

"""
Convert an entire column to Union{Missing,Float64}.
"""
function numericize_column(col)
    return Union{Missing,Float64}[to_float_or_missing(x) for x in col]
end

# ============================================================
# LOAD DATA
# ============================================================

# Be liberal about missing markers
df = CSV.read(input_csv, DataFrame; missingstring=["", "NA", "NaN", "nan", "missing", "null"])
Wdf = CSV.read(adj_csv, DataFrame; missingstring=["", "NA", "NaN", "nan", "missing", "null"])

# ============================================================
# IDENTIFY COLUMNS BY POSITION
# ============================================================

# General assumption:
# col 1 = mouse/sample ID
# col 2 = MPI
# col 3:end = regions
all_cols = names(df)

if length(all_cols) < 3
    error("Input CSV must have at least 3 columns: sample ID, MPI, and >=1 region column.")
end

id_col     = all_cols[1]
mpi_col    = all_cols[2]
region_cols = all_cols[3:end]

# ============================================================
# COERCE MPI + REGION COLUMNS TO NUMERIC
# ============================================================

df[!, mpi_col] = numericize_column(df[!, mpi_col])

for col in region_cols
    df[!, col] = numericize_column(df[!, col])
end

# Drop rows where MPI is missing, since grouping by missing MPI is not useful here
df = filter(row -> !ismissing(row[mpi_col]), df)

# ============================================================
# EXTRACT REGION INDEX MAP FROM ADJACENCY MATRIX
# ============================================================

# First column contains adjacency row labels
label_col = names(Wdf)[1]
adj_regions = String.(Wdf[!, label_col])

# 1-based Julia indexing
region_to_index = Dict(region => i for (i, region) in enumerate(adj_regions))

# ============================================================
# AVERAGE ACROSS MICE WITHIN EACH MPI
# ============================================================

grouped = combine(
    groupby(df, mpi_col),
    region_cols .=> safe_mean .=> region_cols
)

# Sort grouped rows by MPI ascending for readability
sort!(grouped, mpi_col)

# ============================================================
# FOR EACH REGION: FIND PEAK MEAN PATHOLOGY THROUGH TIME
# ============================================================

rows = NamedTuple[]

for region_sym in region_cols
    region = String(region_sym)

    vals = grouped[!, region_sym]
    mpis = grouped[!, mpi_col]

    valid_idx = findall(.!ismissing.(vals))
    isempty(valid_idx) && continue

    vals_valid = vals[valid_idx]
    mpis_valid = mpis[valid_idx]

    max_idx = argmax(vals_valid)
    peak_mean_pathology = vals_valid[max_idx]
    peak_mpi = mpis_valid[max_idx]

    adj_index = get(region_to_index, region, missing)

    push!(rows, (
        region = region,
        adjacency_index = adj_index,
        peak_mean_pathology = peak_mean_pathology,
        peak_mpi = peak_mpi
    ))
end

result_df = DataFrame(rows)

# ============================================================
# SORT BY PEAK MEAN PATHOLOGY (HIGHEST FIRST)
# ============================================================

sort!(result_df, :peak_mean_pathology, rev=true)

# ============================================================
# SAVE TXT
# ============================================================

open(output_txt, "w") do io
    println(io, "region\tadjacency_index\tpeak_mean_pathology\tpeak_mpi")
    for row in eachrow(result_df)
        println(io, "$(row.region)\t$(row.adjacency_index)\t$(row.peak_mean_pathology)\t$(row.peak_mpi)")
    end
end

println("Saved → $output_txt")