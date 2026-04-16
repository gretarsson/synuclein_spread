using CSV
using DataFrames

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(PROJECT_ROOT, "data", "syn_tau_abeta")
const TEMPLATE_PATH = joinpath(PROJECT_ROOT, "data", "total_path.csv")

# Input files to process
const INPUT_FILES = [
    "Ab40_15M_PFF-PHF-Ctrl_MAPTApp KI.csv",
    "Ab42_15M_PFF-PHF-Ctrl_MAPTApp KI.csv",
]

# Prefix to use in output filenames for each input file
const FILE_PREFIX = Dict(
    "Ab40_15M_PFF-PHF-Ctrl_MAPTApp KI.csv" => "ab40_pathology",
    "Ab42_15M_PFF-PHF-Ctrl_MAPTApp KI.csv" => "ab42_pathology",
)

# Treatment labels in the input and corresponding output suffixes
const TREATMENT_SUFFIX = Dict(
    "mPFF"   => "mpff",
    "AD PHF" => "adphf",
    "none"   => "control",
)

const FIXED_MPI = 9

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

function build_output_for_treatment(df::DataFrame, template_cols::Vector{String}, treatment::String)
    sub = filter(:treatment => x -> strip(string(x)) == treatment, df)

    sub.outcol = [hemi_prefix(h) * string(r) for (h, r) in zip(sub.hemi, sub.region)]
    sub.value = transform_prelimval.(sub.preLimVal)
    sub.mouse_clean = normalize_mouse_id.(sub.mouse)
    sub.mpi = fill(FIXED_MPI, nrow(sub))

    allowed_cols = Set(template_cols)
    sub = filter(:outcol => in(allowed_cols), sub)

    # Guard against duplicate rows for the same mouse/timepoint/output column
    dup = combine(groupby(sub, [:mouse_clean, :mpi, :outcol]), nrow => :n)
    dup = filter(:n => >(1), dup)
    if nrow(dup) > 0
        println("Duplicate rows found for treatment=$treatment after mapping to output columns:")
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

    # Add any missing template columns as missing
    for col in template_cols
        if !(col in names(wide))
            wide[!, col] = Vector{Union{Missing, Float64}}(missing, nrow(wide))
        end
    end

    wide = wide[:, vcat(["mouse", "mpi"], template_cols)]
    sort!(wide, [:mpi, :mouse])

    return wide
end

function process_file(input_filename::String, template_cols::Vector{String})
    input_path = joinpath(DATA_DIR, input_filename)
    if !isfile(input_path)
        error("Input file not found: $input_path")
    end

    df = CSV.read(input_path, DataFrame)

    # Report columns that are dropped because they are not in the template
    input_output_cols = Set([hemi_prefix(h) * string(r) for (h, r) in zip(df.hemi, df.region)])
    missing_from_template = sort!(collect(setdiff(input_output_cols, Set(template_cols))))
    if !isempty(missing_from_template)
        println("For file $input_filename, dropping columns not present in total_path.csv:")
        println(join(missing_from_template, ", "))
    end

    treatments_present = sort!(unique(strip.(string.(df.treatment))))
    expected_treatments = collect(keys(TREATMENT_SUFFIX))
    unexpected = setdiff(treatments_present, expected_treatments)
    if !isempty(unexpected)
        error("Unexpected treatment labels found in $input_filename: $(unexpected)")
    end

    prefix = FILE_PREFIX[input_filename]

    println("Processing: $input_filename")
    for treatment in expected_treatments
        wide = build_output_for_treatment(df, template_cols, treatment)
        suffix = TREATMENT_SUFFIX[treatment]
        out_path = joinpath(DATA_DIR, "$(prefix)_$(suffix).csv")
        CSV.write(out_path, wide; missingstring="NA")
        println("Wrote: $out_path")
    end
end

function main()
    if !isfile(TEMPLATE_PATH)
        error("Template file not found: $TEMPLATE_PATH")
    end

    mkpath(DATA_DIR)

    template = CSV.read(TEMPLATE_PATH, DataFrame)
    template_cols = string.(names(template)[3:end])

    for input_filename in INPUT_FILES
        process_file(input_filename, template_cols)
    end
end

main()