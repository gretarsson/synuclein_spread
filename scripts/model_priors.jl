using OrderedCollections, Distributions

"""
    get_priors(ode::Function, K::Int)

Return a fresh `OrderedDict{String,Any}` of priors for the given `ode`,
with all of your β[1…K], d[1…K], etc., built in.
Throws an error if you ask for an ODE we haven’t defined.
"""
function get_priors(ode::String, K::Int)
    if ode === "DIFFGAM"
        return OrderedDict{String,Any}(
          # fixed‑size
          "rho"      => truncated(Normal(0,0.1), lower=0),
          "alpha"    => truncated(Normal(0,0.1), lower=0),

          # regional λ₀’s
          [ "lambda0[$i]"   => truncated(Normal(0,1), lower=0) for i in 1:K ]...,

          # regional λ∞’s
          [ "lambdaInf[$i]" => truncated(Normal(0,1), lower=0) for i in 1:K ]...,

          # more fixed‑size
          "theta"      => truncated(Normal(0,0.1), lower=0),
          "lambdaCrit" => truncated(Normal(0,1),   lower=0),
        )

    elseif ode === "DIFFGA"
        return OrderedDict{String,Any}(
          # fixed‑size
          "rho"      => truncated(Normal(0,0.1), lower=0),
          "alpha"    => truncated(Normal(0,0.1), lower=0),

          # regional β’s
          [ "beta[$i]"  => Normal(0,1) for i in 1:K ]...,

          # regional γ’s
          [ "gamma[$i]" => truncated(Normal(0,0.1), lower=0) for i in 1:K ]...,
        )

    elseif ode === "DIFFG"
        return OrderedDict{String,Any}(
          # fixed‑size
          "rho"      => truncated(Normal(0,0.1), lower=0),
          "alpha"    => truncated(Normal(0,0.1), lower=0),

          # regional β’s
          [ "beta[$i]"  => Normal(0,1) for i in 1:K ]...,
        )

    elseif ode === "DIFF"
        return OrderedDict{String,Any}(
          # fixed‑size
          "rho"      => truncated(Normal(0,0.1), lower=0),
        )

    # ———————————————————————————————
    # Bidirectional variants
    # ———————————————————————————————

    elseif ode === "DIFFGAM_bidirectional"
        return OrderedDict{String,Any}(
          # bidirectional ρ parameters
          "rhoRetro"  => truncated(Normal(0,0.1), lower=0),
          "rhoAntero" => truncated(Normal(0,0.1), lower=0),
          "alpha"     => truncated(Normal(0,0.1), lower=0),

          # regional λ₀’s
          [ "lambda0[$i]"   => truncated(Normal(0,1), lower=0) for i in 1:K ]...,

          # regional λ∞’s
          [ "lambdaInf[$i]" => truncated(Normal(0,1), lower=0) for i in 1:K ]...,

          # more fixed‑size
          "theta"      => truncated(Normal(0,0.1), lower=0),
          "lambdaCrit" => truncated(Normal(0,1),   lower=0),
        )

    elseif ode === "DIFFGA_bidirectional"
        return OrderedDict{String,Any}(
          # bidirectional ρ parameters
          "rhoRetro"  => truncated(Normal(0,0.1), lower=0),
          "rhoAntero" => truncated(Normal(0,0.1), lower=0),
          "alpha"     => truncated(Normal(0,0.1), lower=0),

          # regional β’s
          [ "beta[$i]"  => Normal(0,1) for i in 1:K ]...,

          # regional γ’s
          [ "gamma[$i]" => truncated(Normal(0,0.1), lower=0) for i in 1:K ]...,
        )

    elseif ode === "DIFFG_bidirectional"
        return OrderedDict{String,Any}(
          # bidirectional ρ parameters
          "rhoRetro"  => truncated(Normal(0,0.1), lower=0),
          "rhoAntero" => truncated(Normal(0,0.1), lower=0),
          "alpha"     => truncated(Normal(0,0.1), lower=0),

          # regional β’s
          [ "beta[$i]"  => Normal(0,1) for i in 1:K ]...,
        )

    elseif ode === "DIFF_bidirectional"
        return OrderedDict{String,Any}(
          # bidirectional ρ parameters
          "rhoRetro"  => truncated(Normal(0,0.1), lower=0),
          "rhoAntero" => truncated(Normal(0,0.1), lower=0),
        )


    # BILATERAL
    elseif ode === "DIFFGAM_bilateral"
        return OrderedDict{String,Any}(
          # fixed‑size
          "rho"      => truncated(Normal(0,0.1), lower=0),
          "alpha"    => truncated(Normal(0,0.1), lower=0),

          # regional λ₀’s
          [ "lambda0[$i]"   => truncated(Normal(0,1), lower=0) for i in 1:K ]...,

          # regional λ∞’s
          [ "lambdaInf[$i]" => truncated(Normal(0,1), lower=0) for i in 1:K ]...,

          # more fixed‑size
          "theta"      => truncated(Normal(0,0.1), lower=0),
          "lambdaCrit" => truncated(Normal(0,1),   lower=0),
        )


    else
        error("No priors defined for ODE: $ode")
    end
end
