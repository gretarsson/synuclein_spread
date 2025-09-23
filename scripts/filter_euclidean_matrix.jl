using CSV
using DataFrames

# Paths
dist_file   = "data/Euclidean_distance_matrix.csv"
filter_file = "data/W_labeled_filtered.csv"
out_file    = "data/Euclidean_distance_matrix_filtered.csv"

# Read CSVs
dist_df   = CSV.read(dist_file, DataFrame; header=true)
filter_df = CSV.read(filter_file, DataFrame; header=true)

# Extract region names from the filter file (row/column headers)
regions_to_keep = names(filter_df)[2:end]  # assume first col is row names
regions_to_keep = unique([regions_to_keep; filter_df[:,1]])  # col + row headers

# Filter rows and columns in the distance matrix
# Assume first column of dist_df is region names
dist_df = filter(row -> row[1] in regions_to_keep, dist_df)

# Keep only the first column (region names) and those matching in regions_to_keep
cols_to_keep = vcat(names(dist_df)[1], intersect(names(dist_df)[2:end], regions_to_keep))
dist_df = dist_df[:, cols_to_keep]

# Save filtered matrix
CSV.write(out_file, dist_df)

println("Filtered distance matrix written to: $out_file")
