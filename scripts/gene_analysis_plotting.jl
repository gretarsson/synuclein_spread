# ANALYZE GENE ANALYSIS RESULTS
using Serialization
include("helpers.jl");

# read the gene anlysis results
file_name = "gene_significance_d"
counts, labeled_counts, S, mode_significant = deserialize("simulations/"*file_name*".jls")  
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];
threshold = 0.0512  # 0.0512 for beta, 0.0503 for d
S

# Create the histogram
all_labeled_counts = Dict(label => get(labeled_counts, label, 0) for label in gene_labels)  # add count of genes with no significance
portions = Dict(key => value / S for (key,value) in all_labeled_counts)
num_bins = ceil(Int, sqrt(length(all_labeled_counts)))
hist = histogram(portions, bins=num_bins, 
    xlabel="Significance portion (S=$(S))", 
    ylabel="Frequency", 
    title="Number of times gene is found significant", 
    yscale=:log10,
    legend=false);
# add threshodl line
vline!([threshold], color=:red, lw=2, label=false);

# Display the sorted labels and their counts
sorted_labels = sort(collect(portions), by=x->x[2], rev=false);
significant_labels = []
println("Labels sorted by counts:")
for (label, count) in sorted_labels
    if count > threshold
        println("$label: $count")
        push!(significant_labels,label)
    end
end
display("Total number of significant genes: $(length(significant_labels))")
# save as jls as well as text file
serialize("simulations/"*file_name*"_list.jls",significant_labels)
open("simulations/"*file_name*"_list.txt", "w") do file
    write(file, join(significant_labels, ","))
end;
mode_significant_labels = gene_labels[mode_significant];
display("Total number of mode significant genes: $(length(mode_significant_labels))")
open("simulations/"*file_name*"_list_mode.txt", "w") do file
    write(file, join(mode_significant_labels, ","))
end;
# what is the intersection between distribution signifcance and mode significance
both_labels = intersect(significant_labels, mode_significant_labels)


# Show the plot
display(current())
Plots.savefig(hist,"figures/gene_analysis/histogram_"*file_name*".pdf")
