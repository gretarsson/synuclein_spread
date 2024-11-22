# ANALYZE GENE ANALYSIS RESULTS
using Serialization
using QuadGK
include("helpers.jl");

# read the gene anlysis results
file_name = "null_d"
counts, labeled_counts, S, _ = deserialize("simulations/"*file_name*".jls")  
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];
S

# Create the histogram
all_labeled_counts = Dict(label => get(labeled_counts, label, 0) for label in gene_labels)  # add count of genes with no significance
portions = Dict(key => value / S for (key,value) in all_labeled_counts)

# ----------------------
num_bins = ceil(Int, sqrt(length(all_labeled_counts)))
hist = histogram(portions, bins=num_bins, 
    xlabel="Significance portion (S=$(S))", 
    ylabel="Frequency", 
    title="Null distribution, d", 
    yscale=:log10,
    legend=false);

# Calculate the 95th percentile
alpha = 0.05 / length(gene_labels)
upper_95_percentile = quantile(collect(values(portions)), 1-alpha)

# Display the sorted labels and their counts
sorted_labels = sort(collect(labeled_counts), by=x->x[2], rev=false);
println("Labels sorted by counts:")
for (label, count) in sorted_labels
    println("$label: $count")
end
display("Total number of significant genes: $(length(sorted_labels))")

# Show the plot
display(current())
Plots.savefig(hist,"figures/gene_analysis/histogram_"*file_name*".pdf")
