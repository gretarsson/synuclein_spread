#!/usr/bin/env julia --project=.

using PathoSpread

# ---------------------------------------------------------------------
# USAGE
#   julia --project=. scripts/print_region_labels.jl data/W_labeled.csv [label1 label2 ...]
#
# EXAMPLES
#   julia scripts/print_region_labels.jl data/W_labeled.csv
#   julia scripts/print_region_labels.jl data/W_labeled.csv CA1 CA3 DG
#
# DESCRIPTION
#   - If only the W file is given, prints all region names and indices.
#   - If labels are given, prints only their indices and names (case-sensitive).
# ---------------------------------------------------------------------

function main()
    if length(ARGS) < 1
        println("Usage: julia print_region_labels.jl <W_file.csv> [label1 label2 ...]")
        return
    end

    w_file = ARGS[1]
    query_labels = length(ARGS) > 1 ? ARGS[2:end] : String[]

    # Read adjacency matrix and extract labels
    Lr, N, labels = read_W(w_file; direction=:retro)

    if isempty(query_labels)
        println("File: $w_file")
        println("Number of regions: $N\n")
        println("All region labels (1-based indices):")
        for (i, lbl) in enumerate(labels)
            println(rpad(i, 4), lbl)
        end
    else
        println("File: $w_file")
        println("Searching for labels: ", join(query_labels, ", "))
        println()
        found = false
        for lbl in query_labels
            idx = findfirst(==(lbl), labels)
            if isnothing(idx)
                println("Label not found: $lbl")
            else
                println("$(rpad(lbl, 15)) → index $(idx)")
                found = true
            end
        end
        if !found
            println("\nNo matching labels found.")
        end
    end
end

main()
