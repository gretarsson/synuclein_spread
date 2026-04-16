using CSV
using DataFrames

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

# Input data file to convert
const INPUT_FILENAME = "Syn Pathology_in MAPTApp KI.csv"

# Prefix used for output filenames
const OUTPUT_PREFIX = "syn_pathology"

# Directory containing the input/output data
const DATA_SUBDIR = joinpath("data", "syn_tau_abeta")

# Template file defining allowed columns and required order
const TEMPLATE_FILENAME = joinpath("data", "total_path.csv")

# -----------------------------------------------------------------------------
# Resolved paths
# -----------------------------------------------------------------------------

const INPUT_PATH    = joinpath(PROJECT_ROOT, DATA_SUBDIR, INPUT_FILENAME)
const TEMPLATE_PATH = joinpath(PROJECT_ROOT, TEMPLATE_FILENAME)
const OUT_DIR       = joinpath(PROJECT_ROOT, DATA_SUBDIR)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

function transform_prelimval(x)
    if ismissing(x)
        return missing
    end
    xv = Float64(x)
    return xv == -5.0 ? 0.0 : exp(xv)
end

function hemi_prefix(hemi)
    hemi_s = lowercase(strip(string(hemi)))
    if hemi_s == "left"
        return "i"
    elseif hemi_s == "right"
        return "c"
    else
        error("Unexpected hemisphere value: $(repr(hemi))")
    end
end

function normalize_mouse_id(x)
    s = strip(string(x))
    return replace(s, " " => "")
end

function build_output_for_genotype(df::DataFrame, template_cols::Vector{String}, genotype::String)
    sub = filter(:genotype => x -> lowercase(strip(string(x))) == genotype, df)

    sub.outcol = [hemi_prefix(h) * string(r) for (h, r) in zip(sub.hemi, sub.region)]
    sub.value = transform_prelimval.(sub.preLimVal)
    sub.mouse_clean = normalize_mouse_id.(sub.mouse)

    allowed_cols = Set(template_cols)
    sub = filter(:outcol => in(allowed_cols), sub)

    dup = combine(groupby(sub, [:mouse_clean, :mpi, :outcol]), nrow => :n)
    dup = filter(:n => >(1), dup)
    if nrow(dup) > 0
        println("Duplicate rows found for genotype=$genotype after mapping to output columns:")
        show(dup, allrows=true, allcols=true)
        println()
        error("Aborting due to duplicate rows.")
    end

    wide = unstack(
        sub[:, [:mouse_clean, :mpi, :outcol, :value]],
        [:mouse_clean, :mpi],
        :outcol,
        :value
    )

    rename!(wide, :mouse_clean => :mouse)

    for col in template_cols
        if !(col in names(wide))
            wide[!, col] = Vector{Union{Missing, Float64}}(missing, nrow(wide))
        end
    end

    wide = wide[:, vcat(["mouse", "mpi"], template_cols)]
    sort!(wide, [:mpi, :mouse])

    return wide
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    if !isfile(INPUT_PATH)
        error("Input file not found: $INPUT_PATH")
    end
    if !isfile(TEMPLATE_PATH)
        error("Template file not found: $TEMPLATE_PATH")
    end

    mkpath(OUT_DIR)

    df = CSV.read(INPUT_PATH, DataFrame)
    template = CSV.read(TEMPLATE_PATH, DataFrame)

    template_cols = string.(names(template)[3:end])

    genotypes_present = sort!(unique(lowercase.(strip.(string.(df.genotype)))))
    expected = Set(["mapt", "app"])
    unexpected = setdiff(genotypes_present, collect(expected))
    if !isempty(unexpected)
        error("Unexpected genotype labels found: $(unexpected)")
    end

    input_output_cols = Set([hemi_prefix(h) * string(r) for (h, r) in zip(df.hemi, df.region)])
    missing_from_template = sort!(collect(setdiff(input_output_cols, Set(template_cols))))
    if !isempty(missing_from_template)
        println("Dropping columns not present in total_path.csv:")
        println(join(missing_from_template, ", "))
    end

    mapt_wide = build_output_for_genotype(df, template_cols, "mapt")
    app_wide  = build_output_for_genotype(df, template_cols, "app")

    out_mapt = joinpath(OUT_DIR, "$(OUTPUT_PREFIX)_mapt.csv")
    out_app  = joinpath(OUT_DIR, "$(OUTPUT_PREFIX)_app.csv")

    CSV.write(out_mapt, mapt_wide; missingstring="NA")
    CSV.write(out_app, app_wide; missingstring="NA")

    println("Input file: $INPUT_PATH")
    println("Wrote: $out_mapt")
    println("Wrote: $out_app")
end

main()