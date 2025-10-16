using CSV, DataFrames, Statistics

# ───────────────────────────────────────────────
# File paths
# ───────────────────────────────────────────────
indir = "data"
copath_file = joinpath(indir, "hippocampal/CoPathQuant_pSyn.csv")
connectome_file = joinpath(indir, "W_labeled_filtered.csv")
outfile = joinpath(indir, "hippocampal/CoPathQuant_aligned_to_connectome.csv")

println("Loading data…")

# ───────────────────────────────────────────────
# Load datasets
# ───────────────────────────────────────────────
df = CSV.read(copath_file, DataFrame)
W  = CSV.read(connectome_file, DataFrame)

# filter for the treatment type
left_df  = df[df.treatment .== "mPFF", :]

# Fix column naming
rename!(df, Symbol("% pathology pSYN") => :value)

# get right and left hemisphere of the pathology data
left_df  = df[df.side .== "left", :]
right_df = df[df.side .== "right", :]

# initialize output dataframe with unique mouse × mpi combinations
out = unique(df[:, [:mouse, :mpi]])

# get labels
labels = W[:,1]
for label in labels
    # get prefix and determine hemisphere
    prefix = first(label)
    if prefix == 'i'
        subdf = left_df    # "i" → ipsilateral → left hemisphere
    elseif prefix == 'c'
        subdf = right_df   # "c" → contralateral → right hemisphere
    else
        @warn "Unknown hemisphere prefix in label: $label"
        continue
    end
    
    # strip prefix to get base region name
    base_label = label[2:end]

    # check role of base_label in CoPath hierarchy
    is_parent  = any(subdf.parent .== base_label)
    is_daughter = any(subdf.daughter .== base_label)
    if is_parent && is_daughter
        role = "both"
    elseif is_parent
        role = "parent"
    elseif is_daughter
        role = "daughter"
    else
        role = "none"
    end

    # if daughter or both, just add the entries
    if role == "daughter" || role == "both"
        # extract the rows for this region
        region_df = subdf[(subdf.daughter .== base_label) .| (subdf.parent .== base_label), [:mouse, :mpi, :value]]
        rename!(region_df, :value => Symbol(label))
        display(region_df)
    
        # join into your main output DataFrame (assuming it’s called `out`)
                # region_df has :mouse, :mpi, and :value
        colname = Symbol(label)

        # create empty column with missing values
        out[!, colname] = Vector{Union{Missing, Float64}}(fill(missing, nrow(out)))

        # fill the correct entries
        for row in eachrow(region_df)
            mask = (out.mouse .== row.mouse) .& (out.mpi .== row.mpi)
            out[mask, colname] .= row[colname]
        end
        continue
    end
    # Case 2: label is only a parent
    if is_parent && !is_daughter
        # find all daughter regions of this parent (for this hemisphere)
        daughters = unique(subdf[subdf.parent .== base_label, :daughter])

        if isempty(daughters)
            @warn "Parent $base_label has no daughters listed in CoPath"
            continue
        end

        # build small dataframe of all daughter rows
        daughter_df = subdf[in.(subdf.daughter, Ref(daughters)), [:mouse, :mpi, :value]]

        # average pathology per mouse × mpi
        avg_df = combine(groupby(daughter_df, [:mouse, :mpi]), :value => mean => :value)

        # now fill these averaged values into the parent’s column in out
        colname = Symbol(label)
        out[!, colname] = Vector{Union{Missing, Float64}}(fill(missing, nrow(out)))

        for row in eachrow(avg_df)
            mask = (out.mouse .== row.mouse) .& (out.mpi .== row.mpi)
            out[mask, colname] .= row.value
        end

        continue  # skip to next label, since this case is complete
    end

    

end
out


# all expected labels from connectome (W)
expected_labels = String.(W[:, 1]);  # assumes first column of W contains the labels

# actual labels present in out (excluding :mouse and :mpi)
present_labels = setdiff(names(out), [:mouse, :mpi]);
present_labels = String.(present_labels);

# find which expected labels are missing
missing_labels = setdiff(expected_labels, present_labels);

println("Found $(length(missing_labels)) missing labels:")
display(missing_labels)


using DataFrames, StringDistances

# ───────────────────────────────────────────────
# Collect CoPath region names (parents + daughters)
# ───────────────────────────────────────────────
copath_labels = unique(vcat(df.parent, df.daughter))
copath_labels = filter(!ismissing, copath_labels)

# Strip hemisphere prefix from connectome missing labels
missing_bases = [l[2:end] for l in missing_labels]

println("Checking $(length(missing_bases)) missing connectome regions for close matches in CoPath…")

# ───────────────────────────────────────────────
# Substring / containment matches
# ───────────────────────────────────────────────
substring_hits = DataFrame(missing = String[], copath = String[], type = String[])
for m in missing_bases, c in copath_labels
    if occursin(m, c)
        push!(substring_hits, (m, c, "missing⊂copath"))
    elseif occursin(c, m)
        push!(substring_hits, (m, c, "copath⊂missing"))
    end
end

println("\nFound $(nrow(substring_hits)) substring overlaps:")
println(substring_hits)

# ───────────────────────────────────────────────
# Fuzzy matches (edit distance ≤ 2)
# ───────────────────────────────────────────────
fuzzy_hits = DataFrame(missing = String[], copath = String[], distance = Float64[])
for m in missing_bases, c in copath_labels
    dist = evaluate(Levenshtein(), m, c)
    if dist > 0 && dist ≤ 2
        push!(fuzzy_hits, (m, c, dist))
    end
end

println("\nFound $(nrow(fuzzy_hits)) fuzzy matches (edit distance ≤ 2):")
first(fuzzy_hits, 20)

# ───────────────────────────────────────────────
# Combined report
# ───────────────────────────────────────────────
println("\nSummary:")
println("• Total missing connectome labels: $(length(missing_labels))")
println("• With substring overlaps: $(nrow(substring_hits))")
println("• With fuzzy matches (possible typos): $(nrow(fuzzy_hits))")


# -------------------------------------------
# CHECK CONSISTENCY
using DataFrames, Statistics

# Helper: determine hemisphere and base region name
function parse_label(lbl::String)
    prefix = first(lbl)
    hemi = if prefix == 'i'
        "left"
    elseif prefix == 'c'
        "right"
    else
        missing
    end
    base = lbl[2:end]  # strip hemisphere prefix
    return hemi, base
end

# Extract region labels (skip mouse, mpi)
region_labels = filter!(x -> !(x in ["mouse", "mpi"]), names(out))

# DataFrame to store mismatches
mismatches = DataFrame(mouse = String[], mpi = Any[],
                       label = String[], out_value = Float64[],
                       df_value = Float64[])

for label in region_labels
    hemi, base = parse_label(String(label))
    subdf = df[df.side .== hemi, :]

    for row in eachrow(out)
        val_out = row[label]
        if ismissing(val_out)
            continue
        end

        # Find corresponding entries in df
        matches = subdf[(subdf.mouse .== row.mouse) .&
                        (subdf.mpi .== row.mpi) .&
                        ((subdf.parent .== base) .| (subdf.daughter .== base)), :]

        if nrow(matches) == 0
            @warn "No match found in df" mouse=row.mouse mpi=row.mpi label=label
            continue
        end

        # Take first match (should be unique)
        val_df = first(matches.value)

        # Compare values with a small tolerance
        if abs(val_out - val_df) > 1e-8
            push!(mismatches, (string(row.mouse), row.mpi, String(label), val_out, val_df))
        end
    end
end

println("Found $(nrow(mismatches)) mismatches between 'out' and 'df'.")
first(mismatches, 10)

