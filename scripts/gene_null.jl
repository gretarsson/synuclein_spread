# ANALYZE GENE ANALYSIS RESULTS
using Serialization
using QuadGK
include("helpers.jl");

# read the gene anlysis results
file_name = "null_d"
#counts, labeled_counts, rs, labeled_rs, S, _, significants = deserialize("simulations/"*file_name*".jls")  
counts, labeled_counts, S, _, significants = deserialize("simulations/"*file_name*".jls")  
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];
S

# Create the histogram
all_labeled_counts = Dict(label => get(labeled_counts, label, 0) for label in gene_labels)  # add count of genes with no significance
portions = Dict(key => value / S for (key,value) in all_labeled_counts)
#portions = [length(significant) / S for significant in significants]  # count #signifcant genes in each run

# ----------------------
num_bins = ceil(Int, sqrt(length(all_labeled_counts)))
hist = histogram(portions, bins=num_bins, 
    xlabel="Significance portion (S=$(S))", 
    ylabel="Frequency", 
    title="Null distribution, β", 
    yscale=:log10,
    label=false,
    legend=true);

# Calculate the 95th percentile
alpha = 0.001 / length(gene_labels)
upper_95_percentile = quantile(collect(values(portions)), 1-alpha)
println("The significance threshold is $(upper_95_percentile)")

# Display the sorted labels and their counts
sorted_labels = sort(collect(labeled_counts), by=x->x[2], rev=false);
println("Labels sorted by counts:")
for (label, count) in sorted_labels
    println("$label: $count")
end
display("Total number of significant genes: $(length(sorted_labels))")


# Add a vertical line for the threshold
vline!([upper_95_percentile], color=:red, lw=2, label="95% percentile $(upper_95_percentile)");


# Show the plot
display(current())
Plots.savefig(hist,"figures/gene_analysis/histogram_full_"*file_name*".pdf")
