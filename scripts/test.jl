using PathoSpread
using CSV, DataFrames, CairoMakie, Statistics
using LinearAlgebra

# --- Load adjacency matrix ---
w_file = "data/W_labeled_filtered.csv"
df = CSV.read(w_file, DataFrame)
labels = String.(names(df)[2:end])
W = Matrix{Float64}(df[:, 2:end])
W[diagind(W)] .= 0.0
N = size(W, 1)

# --- Shuffle weights ---
Wshuf = PathoSpread.shuffle_weights(W)

# --- Strength measures ---
in_strength(W)  = vec(sum(W, dims=1))
out_strength(W) = vec(sum(W, dims=2))
tot_strength(W) = in_strength(W) .+ out_strength(W)

instr0 = in_strength(W);   outstr0 = out_strength(W);   totstr0 = tot_strength(W)
instr1 = in_strength(Wshuf); outstr1 = out_strength(Wshuf); totstr1 = tot_strength(Wshuf)

println("Mean in-strength (orig/shuf):  ", mean(instr0), " / ", mean(instr1))
println("Mean out-strength (orig/shuf): ", mean(outstr0), " / ", mean(outstr1))
println("Mean total strength (orig/shuf): ", mean(totstr0), " / ", mean(totstr1))
println("Mean (orig / shuf): ", mean(W), " / ", mean(Wshuf))
println("Std  (orig / shuf): ", std(W),  " / ", std(Wshuf))
sum_diag = sum(diag(W))

# -------------------------------------------------------------------------
# Figure 1: Original network
# -------------------------------------------------------------------------
f1 = Figure(resolution=(1350,400));
ax1a = Axis(f1[1,1], title="Original — In-strength", xlabel="Strength", ylabel="Density")
ax1b = Axis(f1[1,2], title="Original — Out-strength", xlabel="Strength", ylabel="Density")
ax1c = Axis(f1[1,3], title="Original — Total strength", xlabel="Strength", ylabel="Density")

hist!(ax1a, instr0; normalization=:pdf, bins=50, color=:dodgerblue)
hist!(ax1b, outstr0; normalization=:pdf, bins=50, color=:dodgerblue)
hist!(ax1c, totstr0; normalization=:pdf, bins=50, color=:dodgerblue)

display(f1)
# save("original_strengths.png", f1)

# -------------------------------------------------------------------------
# Figure 2: Shuffled network
# -------------------------------------------------------------------------
f2 = Figure(resolution=(1350,400))
ax2a = Axis(f2[1,1], title="Shuffled — In-strength", xlabel="Strength", ylabel="Density")
ax2b = Axis(f2[1,2], title="Shuffled — Out-strength", xlabel="Strength", ylabel="Density")
ax2c = Axis(f2[1,3], title="Shuffled — Total strength", xlabel="Strength", ylabel="Density")

hist!(ax2a, instr1; normalization=:pdf, bins=50, color=:crimson)
hist!(ax2b, outstr1; normalization=:pdf, bins=50, color=:crimson)
hist!(ax2c, totstr1; normalization=:pdf, bins=50, color=:crimson)

display(f2)
# save("shuffled_strengths.png", f2)



sum_orig = sum(W)
sum_shuf = sum(Wshuf)
println("Total weight (orig/shuf): ", sum_orig, " / ", sum_shuf)