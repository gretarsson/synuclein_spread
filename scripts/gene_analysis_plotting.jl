# ANALYZE GENE ANALYSIS RESULTS
using Serialization, CSV
include("helpers.jl");

# read the gene anlysis results
file_name = "gene_significance_trunc_d"
counts, labeled_counts, rs, labeled_rs, S, mode_significant = deserialize("simulations/"*file_name*".jls");  
#counts, labeled_counts, S, mode_significant = deserialize("simulations/"*file_name*".jls");  
gene_data_full = readdlm("data/avg_Pangea_exp.csv",',');
gene_labels = gene_data_full[1,2:end];
threshold = 0.0002  # 0.001 for beta, 0.0002 for d
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


# create a CSV file with labels, r-values, and significance portion
labeled_rs_mean = Dict(key => mean(value) for (key,value) in labeled_rs) 
labeled_rs_vars = Dict(key => var(value) for (key,value) in labeled_rs) 
labeled_portions = Dict(key => value / S for (key,value) in labeled_counts)
significance_portions = []
rs_means = []
rs_vars = []
for (i,label) in enumerate(significant_labels)
    push!(significance_portions, labeled_portions[label])
    push!(rs_means,labeled_rs_mean[label])
    push!(rs_vars,labeled_rs_vars[label])
end
#gene_results = DataFrame(hcat(significant_labels,rs_means,rs_vars,significance_portions),["Label","r, mean", "r, variance","portion"])
#gene_results = DataFrame(hcat(significant_labels,rs_means,significance_portions),["gene","r","portions"])
gene_results = DataFrame(hcat(significant_labels,rs_means),["gene","r"])
CSV.write("simulations/gene_correlation_truncated_decay.csv",gene_results)

# Show the plot
display(current())
Plots.savefig(hist,"figures/gene_analysis/histogram_"*file_name*".pdf")
