# Stand-alone check: Normal(0, σ) vs MvNormal([0], σ^2 * I₁)

using Distributions, LinearAlgebra, Statistics, Random

# --- set σ here (or replace with your posterior estimate) ---
σ = 0.3
# e.g. if you have an inference dict:
# σ = median(vec(inference["chain"][:σ]))

# Distributions
d1 = Normal(0.0, σ)
d2 = MvNormal(zeros(1), Diagonal([σ^2]))  # 1×1 covariance = σ²

# Grid comparison of pdf/logpdf
xs = range(-5σ, 5σ; length=121)
pdf_diff   = maximum(abs.(pdf(d1, x) - pdf(d2, [x]) for x in xs))
lpdf_diff  = maximum(abs.(logpdf(d1, x) - logpdf(d2, [x]) for x in xs))

# Monte Carlo sanity check
Random.seed!(123)
nsamples = 50_000
s1 = rand(d1, nsamples)
s2 = vec(rand(d2, nsamples)[1, :])

mean_normal, sd_normal = mean(s1), std(s1)
mean_mvnorm, sd_mvnorm = mean(s2), std(s2)

println("σ = $σ")
println("max |pdf difference|    = $pdf_diff")
println("max |logpdf difference| = $lpdf_diff")
println("Normal:   mean=$(mean_normal), sd=$(sd_normal)")
println("MvNormal: mean=$(mean_mvnorm), sd=$(sd_mvnorm)")

# Tight equalities for densities; loose checks for MC estimates
@assert pdf_diff  < 1e-12 "pdfs disagree more than tolerance"
@assert lpdf_diff < 1e-12 "logpdfs disagree more than tolerance"
@assert abs(sd_normal - σ) < 5e-3 "Normal sd not close to σ"
@assert abs(sd_mvnorm - σ) < 5e-3 "MvNormal sd not close to σ"

println("OK: Normal(0, σ) == MvNormal([0], σ^2 * I₁) in 1D, σ is a standard deviation.")
