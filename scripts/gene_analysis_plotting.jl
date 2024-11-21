# ANALYZE GENE ANALYSIS RESULTS
using Serialization

# read the gene anlysis results
file_name = "null_β"
counts, labeled_counts, S = deserialize("simulations/"*file_name*".jls")  
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];

# Ensure all integers are represented in counts (fill missing with 0)
all_labeled_counts = Dict(label => get(labeled_counts, label, 0) for label in gene_labels)
# Extract the counts
counts = collect(values(all_labeled_counts))

# Automatically determine number of bins (square root rule)
num_bins = ceil(Int, sqrt(length(counts)))

# Create the histogram
hist = histogram(counts, bins=num_bins, 
    xlabel="Significance counts (S=$(S))", 
    ylabel="Frequency", 
    title="Number of times gene is found significant", 
    yscale=:log10,
    legend=false)

# Display the sorted labels and their counts
sorted_labels = sort(collect(labeled_counts), by=x->x[2], rev=false);
println("Labels sorted by counts:")
for (label, count) in sorted_labels
    println("$label: $count")
end
display("Total number of significant genes: $(length(sorted_labels))")

# compute significant genes with mode
r2s, pvals, significant, gene_labels = gene_analysis(simulation,parameter;mode=true,show=false);
display("Significant genes for mode of posterior")
for signi in significant
    display(gene_labels[signi])
end

# Show the plot
display(current())
Plots.savefig(hist,"figures/gene_analysis/histogram_"*file_name*".pdf")
