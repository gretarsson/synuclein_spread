#=
here we create a folder of analysis plots of interence results
=#
using PathoSpread

simulation = "DIFFG_RETRO"
display("Plotting simulations: $simulation")

# read file 
inference_obj = load_inference("simulations/"*simulation*".jls")

# look at chains
display(inference_obj["chain"])
#new_chain = inference_obj["chain"][:,:,[2,3,4]]
#inference_obj["chain"] = new_chain
#save_inference("simulations/" * simulation * "_CUT.jls", inference_obj)

# plot
setup_plot_theme!()  # set plotting settings
display("Plotting inference results...")
plot_inference(inference_obj,"figures/inferences/"*simulation, plot_priors_posteriors=true)  
display("Plots saved to figures/inferences/"*simulation)
display("---------------------------------------------------")

# plot with training data
#setup_plot_theme!()  # set plotting settings
#data_full, timepoints_full = PathoSpread.process_pathology("data/total_path.csv", W_csv="data/W_labeled_filtered.csv")
#plot_inference(inference_obj,"figures/"*simulation; full_data=data_full, full_timepoints=timepoints_full)  
#



using CSV, DataFrames, LinearAlgebra

df = CSV.read("data/W_labeled_filtered.csv", DataFrame)
W = Matrix{Float64}(df[:,2:end])

col_sums = sum(W, dims=1)           # 1×N row vector
row_sums = sum(W, dims=2)           # 1×N row vector
D_out    = Diagonal(vec(col_sums))  # convert to N×N diagonal matrix
D_in    = Diagonal(vec(row_sums))  # convert to N×N diagonal matrix
C_in = exp(D_in^(-1/2)*W*D_in^(-1/2))
C_out = exp(D_out^(-1/2)*W*D_out^(-1/2))
Cs = C_out[:,[1,2,3]]
Cvec = mean(Cs,dims=2)


2 .+ 0.0 .* Cvec