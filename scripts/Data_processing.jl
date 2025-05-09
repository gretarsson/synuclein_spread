module Data_processing

using CSV, DataFrames

export process_pathology

"""
    process_pathology(path_csv::String; W_csv::Union{Nothing,String}=nothing)

Read a “sample–time–region1…regionN” CSV and (optionally)
re‑order regions by a connectome CSV, then return a NamedTuple with

- `data::Array{Union{Float64,Missing}}` of size (regions, times, samples)
- `regions::Vector{Symbol}`
- `timepoints::Vector{<:Real}`
- `max_samples::Int`
"""
function process_pathology(path_csv::String; W_csv::Union{Nothing,String}=nothing)
    ### 1) load pathology CSV, parse "NA" → missing
    df = CSV.read(path_csv, DataFrame; missingstring=["NA"])

    ### 2) identify columns
    time_col     = names(df)[2]
    path_regions = names(df)[3:end]

    ### 3) determine region ordering
    region_order = path_regions
    if W_csv !== nothing
        cdf = CSV.read(W_csv, DataFrame)
        conn_regions = names(cdf)[2:end]
        missing_regions = setdiff(conn_regions, path_regions)
        if !isempty(missing_regions)
            error("These regions appear in the connectome but not in the pathology CSV:\n",
                  join(missing_regions, ", "))
        end
        region_order = conn_regions
    end

    ### 4) assemble timepoints & sample counts
    mpis      = sort(unique(df[!, time_col]))
    gdfs      = groupby(df, time_col)
    max_samps = maximum(nrow(g) for g in gdfs)

    n_regions = length(region_order)
    n_times   = length(mpis)

    ### 5) allocate & fill
    data = Array{Union{Float64,Missing}}(undef, n_regions, n_times, max_samps)
    fill!(data, missing)

    ### 6) populate
    group_map = Dict(g[1, time_col] => g for g in gdfs)
    for (j, mpi) in enumerate(mpis)
        sub = group_map[mpi]
        for s in 1:nrow(sub)
            for (k, reg) in enumerate(region_order)
                data[k, j, s] = sub[s, reg]
            end
        end
    end

    return data, mpis
end

end # module
