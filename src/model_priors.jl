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
          [ "y0[$i]"   => Normal(0,1) for i in 1:K ]...,

          # regional λ∞’s
          [ "ydelta[$i]" => Normal(0,1) for i in 1:K ]...,

          # more fixed‑size
          "theta"      => truncated(Normal(0,0.1), lower=0),
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
          "alpha"    => truncated(Normal(0,0.1), lower=0),

          # regional λ₀’s
          [ "y0[$i]"   => Normal(0,1) for i in 1:K ]...,

          # regional λ∞’s
          [ "ydelta[$i]" => Normal(0,1) for i in 1:K ]...,

          # more fixed‑size
          "theta"      => truncated(Normal(0,0.1), lower=0),
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
    elseif ode === "DIFFG_bilateral"
      return OrderedDict{String,Any}(
        # global parameters
        "rho"      => truncated(Normal(0,0.1), lower=0),
        "alpha"    => truncated(Normal(0,0.1), lower=0),

        # bilateral group-level β’s (shared across hemispheres)
        [ "beta[$i]"  => Normal(0,1) for i in 1:K ]...,
      )
    elseif ode === "DIFFGA_bilateral"
      return OrderedDict{String,Any}(
        # global parameters
        "rho"      => truncated(Normal(0,0.1), lower=0),
        "alpha"    => truncated(Normal(0,0.1), lower=0),
  
        # bilateral group-level β’s
        [ "beta[$i]"  => Normal(0,1) for i in 1:K ]...,
  
        # bilateral group-level γ’s (growth / downstream factors)
        [ "gamma[$i]" => truncated(Normal(0,0.1), lower=0) for i in 1:K ]...,
      )
    elseif ode === "DIFFGAM_bilateral"
      return OrderedDict{String,Any}(
        # global parameters
        "rho"      => truncated(Normal(0,0.1), lower=0),
        "alpha"    => truncated(Normal(0,0.1), lower=0),
  
        # bilateral group-level y0’s
        [ "y0[$i]"     => Normal(0,1) for i in 1:K ]...,
  
        # bilateral group-level ydelta’s
        [ "ydelta[$i]" => Normal(0,1) for i in 1:K ]...,
  
        # global coupling
        "theta"      => truncated(Normal(0,0.1), lower=0),
      )
  

    # ———————————————————————————————
    # Communicability variants
    # ———————————————————————————————
    elseif ode === "DIFFG_comm_in" || ode === "DIFFG_comm_out"
      return OrderedDict{String,Any}(
        # global parameters
        "rho"      => truncated(Normal(0,0.1), lower=0),
        "alpha"    => truncated(Normal(0,0.1), lower=0),
        "beta_i"      => Normal(0,1),
        "beta_s"    => Normal(0,1),
      )
    elseif ode === "DIFFGA_comm_in" || ode === "DIFFGA_comm_out"
      return OrderedDict{String,Any}(
        # global parameters
        "rho"      => truncated(Normal(0,0.1), lower=0),
        "alpha"    => truncated(Normal(0,0.1), lower=0),
        "beta_i"      => Normal(0,1),
        "beta_s"    => Normal(0,1),
        "d"    => truncated(Normal(0,0.1), lower=0),
      )

    # ———————————————————————————————
    # Global variants
    # ———————————————————————————————
    elseif ode === "DIFFGA_global"
        return OrderedDict{String,Any}(
          # fixed‑size
          "rho"      => truncated(Normal(0,0.1), lower=0),
          "alpha"    => truncated(Normal(0,0.1), lower=0),
          "beta"     => Normal(0,1) ,
          "gamma"    => truncated(Normal(0,0.1), lower=0),
        )
    elseif ode === "DIFFG_global"
        return OrderedDict{String,Any}(
          # fixed‑size
          "rho"      => truncated(Normal(0,0.1), lower=0),
          "alpha"    => truncated(Normal(0,0.1), lower=0),
          "beta"     => Normal(0,1),
        )

    # ———————————————————————————————
    # Alpha variants
    # ———————————————————————————————
    elseif ode === "DIFFGA_alpha"
        return OrderedDict{String,Any}(
          # regional alpha
          "rho"      => truncated(Normal(0,0.1), lower=0),
          [ "alpha[$i]"  => truncated(Normal(0,0.1)) for i in 1:K ]...,

          # global β’s
          "beta"  => Normal(0,1),

          # regional γ’s
          [ "gamma[$i]" => truncated(Normal(0,0.1), lower=0) for i in 1:K ]...,
        )

    elseif ode === "DIFFG_alpha"
        return OrderedDict{String,Any}(
          # regional alpha
          "rho"      => truncated(Normal(0,0.1), lower=0),
          [ "alpha[$i]"  => truncated(Normal(0,0.1)) for i in 1:K ]...,

          # global β
          "beta"  => Normal(0,1),
        )

    else
        error("No priors defined for ODE: $ode")
    end
end
