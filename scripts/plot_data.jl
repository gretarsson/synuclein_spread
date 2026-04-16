#!/usr/bin/env julia

using CSV
using DataFrames
using Statistics
using Plots
using Colors

# ============================================================
# SETTINGS
# ============================================================

input_csv  = "data/total_path.csv"
adj_csv    = "data/W_labeled_filtered.csv"
output_dir = "figures/data_visualization"
mkpath(output_dir)

# -------------------------------
# Plot appearance
# -------------------------------
FIGSIZE            = (1100, 750)
DPI_VALUE          = 300

LINE_WIDTH         = 8.0
MARKER_SIZE        = 20
MARKER_STROKE      = 1.5

ERROR_LINE_WIDTH   = 8
ERROR_CAP_SIZE     = 10    # not directly supported by Plots.jl for all backends
SERIES_ALPHA       = 1.0

AXIS_LINE_WIDTH    = 5.0
GRID_ON            = false

LABEL_FONT_SIZE    = 28
TICK_FONT_SIZE     = 22
TITLE_FONT_SIZE    = LABEL_FONT_SIZE
LEGEND_FONT_SIZE   = 16

XTICK_ROTATION     = 0

XLABEL             = "MPI"
YLABEL             = "pathology"

# Error bar type: :sem or :std
ERRORBAR_TYPE      = :std

# Keep y-axis starting at zero
FORCE_YMIN_ZERO    = true

# Add some vertical headroom above top error bar
Y_HEADROOM_FACTOR  = 1.08

# Blue tone similar to the PDF
MAIN_COLOR = RGB(0.00, 0.45, 0.74)

# Choose backend if you want consistency
gr()

# ============================================================
# HELPERS
# ============================================================

"""
Convert a value to Float64 if possible, otherwise return missing.
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
Convert a full column to Vector{Union{Missing,Float64}}.
"""
function numericize_column(col)
    return Union{Missing,Float64}[to_float_or_missing(x) for x in col]
end

"""
Mean and uncertainty ignoring missing values.

Returns (mean, err). If all values are missing, returns (missing, missing).
If n = 1, err = 0.
"""
function mean_and_error(x; error_type=:sem)
    vals = collect(skipmissing(x))
    n = length(vals)

    if n == 0
        return (missing, missing)
    elseif n == 1
        return (vals[1], 0.0)
    else
        μ = mean(vals)
        s = std(vals)

        err = if error_type == :sem
            s / sqrt(n)
        elseif error_type == :std
            s
        else
            error("ERRORBAR_TYPE must be :sem or :std")
        end

        return (μ, err)
    end
end

"""
Create a filename-safe version of a string.
"""
function safe_filename(s::AbstractString)
    s2 = replace(s, r"[^\w\-\.\(\)]" => "_")
    s2 = replace(s2, r"_+" => "_")
    return s2
end

# ============================================================
# LOAD DATA
# ============================================================

df = CSV.read(
    input_csv,
    DataFrame;
    missingstring=["", "NA", "NaN", "nan", "missing", "null"]
)

Wdf = CSV.read(
    adj_csv,
    DataFrame;
    missingstring=["", "NA", "NaN", "nan", "missing", "null"]
)

all_cols = names(df)

if length(all_cols) < 3
    error("Input CSV must have at least 3 columns: sample ID, MPI, and >= 1 region column.")
end

id_col      = all_cols[1]
time_col    = all_cols[2]
region_cols = all_cols[3:end]

# Convert relevant columns to numeric
df[!, time_col] = numericize_column(df[!, time_col])
for col in region_cols
    df[!, col] = numericize_column(df[!, col])
end

# Drop rows with missing MPI
df = filter(row -> !ismissing(row[time_col]), df)

if nrow(df) == 0
    error("No valid rows remain after dropping missing time values.")
end

# ============================================================
# REGION -> INDEX MAP FROM ADJACENCY FILE
# ============================================================

label_col = names(Wdf)[1]
adj_regions = String.(Wdf[!, label_col])
region_to_index = Dict(region => i for (i, region) in enumerate(adj_regions))

# ============================================================
# TIMEPOINTS
# ============================================================

timepoints = unique(df[!, time_col])
timepoints = collect(skipmissing(timepoints))
sort!(timepoints)

# ============================================================
# GLOBAL PLOTTING DEFAULTS
# ============================================================

default(
    size = FIGSIZE,
    dpi = DPI_VALUE,
    linewidth = LINE_WIDTH,
    markersize = MARKER_SIZE,
    guidefont = font(LABEL_FONT_SIZE),
    tickfont = font(TICK_FONT_SIZE),
    titlefont = font(TITLE_FONT_SIZE),
    legendfont = font(LEGEND_FONT_SIZE),
    fg_color_axis = :black,
    fg_color_border = :black,
    fg_color_text = :black,
    fg_color_guide = :black
)
default(margin = 10Plots.mm)

# ============================================================
# PLOT REGION BY REGION
# ============================================================

for region_sym in region_cols
    region = String(region_sym)

    # Only plot regions that exist in the adjacency matrix
    idx = get(region_to_index, region, nothing)
    isnothing(idx) && continue

    means = Union{Missing,Float64}[]
    errs  = Union{Missing,Float64}[]

    for t in timepoints
        sub = df[df[!, time_col] .== t, region_sym]
        μ, e = mean_and_error(sub; error_type=ERRORBAR_TYPE)
        push!(means, μ)
        push!(errs, e)
    end

    # Remove timepoints where mean is missing
    valid = findall(.!ismissing.(means))
    isempty(valid) && continue

    x = Float64[timepoints[i] for i in valid]
    y = Float64[means[i] for i in valid]
    e = Float64[coalesce(errs[i], 0.0) for i in valid]

    # Lower error bar capped at zero
    lower = [min(err, yi) for (yi, err) in zip(y, e)]
    upper = e

    ymax = maximum(y .+ upper)
    ymin = FORCE_YMIN_ZERO ? 0.0 : minimum(y .- lower)

    p = plot(
        x, y;
        yerror = (lower, upper),
        color = MAIN_COLOR,
        linewidth = LINE_WIDTH,
        linealpha = SERIES_ALPHA,
        marker = :circle,
        markersize = MARKER_SIZE,
        markercolor = MAIN_COLOR,
        markerstrokecolor = :black,
        markerstrokewidth = MARKER_STROKE,
        xlabel = XLABEL,
        ylabel = YLABEL,
        title = "",
        label = nothing,
        legend = false,
        grid = GRID_ON,
        xrotation = XTICK_ROTATION,
        framestyle = :box,
        ylims = (ymin, ymax * Y_HEADROOM_FACTOR)
    )

    # Strengthen axis appearance
    plot!(
        p;
        foreground_color_axis = :black,
        foreground_color_border = :black,
        tick_direction = :out
    )

    # Re-draw the series with thicker error bars if backend supports it
    scatter!(
        p, x, y;
        yerror = (lower, upper),
        color = MAIN_COLOR,
        marker = :circle,
        markersize = MARKER_SIZE,
        markercolor = MAIN_COLOR,
        markerstrokecolor = :black,
        markerstrokewidth = MARKER_STROKE,
        label = nothing,
        seriesalpha = SERIES_ALPHA
    )

    # Always save by adjacency index
    outfile = joinpath(output_dir, "region_$(idx).png")

    savefig(p, outfile)
    println("Saved: $outfile")
end

println("Done. Saved all plots to: $output_dir")