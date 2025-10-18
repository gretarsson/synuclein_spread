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
    df = CSV.read(path_csv, DataFrame; missingstring=["NA",""])

    ### 2) identify columns
    time_col     = names(df)[2]
    path_regions = names(df)[3:end]

    ### 3) determine region ordering
    region_order = path_regions
    # OLD
    #if W_csv !== nothing
    #    cdf = CSV.read(W_csv, DataFrame)
    #    conn_regions = names(cdf)[2:end]
    #    missing_regions = setdiff(conn_regions, path_regions)
    #    if !isempty(missing_regions)
    #        error("These regions appear in the connectome but not in the pathology CSV:\n",
    #              join(missing_regions, ", "))
    #    end
    #    region_order = conn_regions
    #end
    # NEW (handles regions missing in pathology data, fills them with missing rows)
    if W_csv !== nothing
        cdf = CSV.read(W_csv, DataFrame)
        conn_regions = names(cdf)[2:end]
        missing_regions = setdiff(conn_regions, path_regions)
    
        if !isempty(missing_regions)
            @warn "These regions appear in the connectome but not in the pathology CSV — filling with missing:" join(missing_regions, ", ")
            for r in missing_regions
                df[!, Symbol(r)] = Vector{Union{Missing, Float64}}(fill(missing, nrow(df)))
            end
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


function save_inference_MAP_csv(inference; path="inference_mode.csv")
    chain    = inference["chain"]
    priors   = inference["priors"]            # OrderedDict: param names in order
    labels   = get(inference, "labels", nothing)
    u0       = inference["u0"]
    seed_idx = inference["seed_idx"]

    # find joint MAP row
    df_chain = DataFrame(chain)
    imax = argmax(df_chain[!,:lp])
    row  = df_chain[imax, :]

    rows = NamedTuple[]

    # --- Parameters from p[i], names from priors (exclude σ) ---
    keys_vec = collect(keys(priors))
    stop_at = findfirst(==("σ"), keys_vec)
    stop_at === nothing && error("Could not find σ in priors")
    N_pars = stop_at - 1
    for i in 1:N_pars
        pname = keys_vec[i]
        pcol  = Symbol("p[$i]")
        @assert hasproperty(row, pcol) "Chain does not contain column $(String(pcol))"
        push!(rows, (index=i, name=pname, value=Float64(row[pcol]), category="parameter"))
    end

    # --- y0 vector (supports N or 2N) ---
    Nlabels = labels === nothing ? length(u0) : length(labels)
    nu0 = length(u0)
    for (i, v) in enumerate(u0)
        label =
            labels === nothing ? "Region_$i" :
            (nu0 == Nlabels    ? labels[i] :
             nu0 == 2*Nlabels  ? (i <= Nlabels ? labels[i] : string(labels[i-Nlabels], "_2")) :
             error("Mismatch: u0 has $nu0 entries but labels has $Nlabels"))
        push!(rows, (index=i, name=label, value=Float64(v), category="initial_condition"))
    end

    # --- seeded region row (index is node index) ---
    seed_label = labels === nothing ? "Region_$seed_idx" : labels[seed_idx]
    push!(rows, (index=seed_idx, name="SEED_$seed_label", value=Float64(u0[seed_idx]), category="seed"))

    CSV.write(path, DataFrame(rows))
    return path
end



