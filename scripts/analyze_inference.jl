#=
here we create a folder of analysis plots of interence results
=#
include("helpers.jl");


# simulation to analyze
simulation = "test_datadict";

# plot 
inference = deserialize("simulations/"*simulation*".jls")
#data = inference["data"]
#var_data = var3(data)
#mean_data = mean3(data)
#var(skipmissing(data[1,4,:]))
#var_data[1,4]
#mean_data[1,4]
#var_data_i = var_data[1,:]
#indices = findall(x -> isnan(x),var_data_i)
#var_data_i[indices] .= 0
#var_data_i


plot_inference(inference,"figures/"*simulation;plotscale=log10) 
