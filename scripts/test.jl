using Serialization
include("Data_processing.jl")
using .Data_processing: process_pathology

data1 = deserialize("data/total_path_3D.jls")
data2 = process_pathology("data/total_path.csv"; W_csv="data/W_labeled.csv")

isequal(data1,data2)

