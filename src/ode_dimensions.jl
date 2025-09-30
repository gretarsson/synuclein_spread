# ODE_dimensions.jl
# Maps ODE model names to a function returning the number of state variables given number of regions

const ode_dimensions = Dict(
    "DIFF"                  => N -> N,
    "DIFF_bidirectional"    => N -> N,
    "DIFF_bilateral"        => N -> N,

    "DIFFG"                 => N -> N,
    "DIFFG_bidirectional"   => N -> N,
    "DIFFG_bilateral"       => N -> N,

    "DIFFGA"                => N -> 2N,
    "DIFFGA_bidirectional"  => N -> 2N,
    "DIFFGA_bilateral"      => N -> 2N,

    "DIFFGAM"               => N -> 2N,
    "DIFFGAM_bidirectional" => N -> 2N,
    "DIFFGAM_bilateral"     => N -> 2N,
)

