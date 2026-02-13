# multinomial_model.jl — Multinomial (softmax) logistic regression NLPModel
#
# Minimize  f(x) = Σᵢ [ log Σₖ exp(aᵢᵀ xₖ) - aᵢᵀ x_{yᵢ} ]
#
# where x ∈ ℝ^{p·K} is the flattened weight matrix (K class predictors of
# dimension p), aᵢ are the feature columns of A (p × n), and yᵢ ∈ {1,…,K}
# are the class labels.
#
# This is a natural extension of binomial_model.jl to K > 2 classes.

using LinearAlgebra

export multinomial_model

"""
    nlp = multinomial_model(A, y, K)

Return an `NLPModel` for multinomial (softmax) logistic regression.

# Arguments
- `A :: Matrix{T}`:  feature matrix, size `p × n` (features × samples)
- `y :: Vector{Int}`: class labels in `{1, …, K}`, length `n`
- `K :: Int`:         number of classes

The decision variable is `x ∈ ℝ^{pK}`, reshaped as a `p × K` weight matrix
`W` where column `k` is the predictor for class `k`.

# Objective
```
f(x) = Σᵢ [ log Σₖ exp(aᵢᵀ wₖ) − aᵢᵀ w_{yᵢ} ]
```
Uses the log-sum-exp trick for numerical stability.
"""
function multinomial_model(A::AbstractMatrix{T}, y::AbstractVector{<:Integer}, K::Int) where {T <: Real}
  p, n = size(A)  # p features, n samples

  length(y) == n || throw(DimensionMismatch("length(y)=$(length(y)) ≠ n=$n"))
  all(1 .≤ y .≤ K) || throw(ArgumentError("labels must be in {1,…,$K}"))

  # Pre-allocate buffers
  S    = zeros(T, K, n)   # scores: S[k,i] = aᵢᵀ wₖ
  P    = zeros(T, K, n)   # probabilities (softmax)
  tmp  = zeros(T, K, n)   # workspace

  # Compute scores S[k,i] = wₖᵀ aᵢ for all k,i
  # W is p×K, A is p×n → S = W' * A  (K×n)
  function compute_scores!(S, x)
    W = reshape(x, p, K)
    mul!(S, W', A)    # S[k,i] = wₖᵀ aᵢ
    return S
  end

  # Compute softmax probabilities P[k,i] = exp(S[k,i]) / Σⱼ exp(S[j,i])
  # Uses log-sum-exp trick for stability
  function compute_softmax!(P, S)
    @inbounds for i in 1:n
      # Find max score for sample i
      mx = S[1, i]
      for k in 2:K
        mx = max(mx, S[k, i])
      end
      # Compute exp(S[k,i] - mx) and sum
      denom = zero(T)
      for k in 1:K
        P[k, i] = exp(S[k, i] - mx)
        denom += P[k, i]
      end
      # Normalize
      inv_denom = one(T) / denom
      for k in 1:K
        P[k, i] *= inv_denom
      end
    end
    return P
  end

  # Objective: f(x) = Σᵢ [ log Σₖ exp(sₖᵢ) - s_{yᵢ,i} ]
  function obj(x)
    compute_scores!(S, x)
    loss = zero(T)
    @inbounds for i in 1:n
      # log-sum-exp for sample i
      mx = S[1, i]
      for k in 2:K
        mx = max(mx, S[k, i])
      end
      lse = zero(T)
      for k in 1:K
        lse += exp(S[k, i] - mx)
      end
      loss += mx + log(lse) - S[y[i], i]
    end
    return loss
  end

  # Gradient: ∇f = vec(A * (P - E)'), where E[k,i] = 1{k == yᵢ}
  # ∇_{wₖ} f = A * (P[k,:] - e_k), where e_k[i] = 1{yᵢ == k}
  # Equivalently, G (p×K) = A * (P - E)'  ... actually:
  # G[:,k] = Σᵢ (P[k,i] - 1{yᵢ=k}) aᵢ = A * (P[k,:] - e_k)
  # Gradient: ∇_{wₖ} f = Σᵢ (P[k,i] - 1{yᵢ=k}) aᵢ
  # G (p×K) = A * (P - E)', where E[k,i] = 1{k == yᵢ}
  function grad!(g, x)
    compute_scores!(S, x)
    compute_softmax!(P, S)

    # tmp = P - E  (use tmp buffer so P stays clean for hprod!)
    copyto!(tmp, P)
    @inbounds for i in 1:n
      tmp[y[i], i] -= one(T)
    end

    G = reshape(g, p, K)
    mul!(G, A, tmp')   # G = A * (P-E)', size p×K
    return g
  end

  # Hessian-vector product: ∇²f · v
  #
  # The Hessian of multinomial logistic regression has block structure.
  # For the full weight vector x = vec(W), with W p×K:
  #
  # ∇²f · v = vec( A * D * (A' V) )
  #
  # where V = reshape(v, p, K), and D is a block-diagonal-like operator
  # acting on n samples. For each sample i, the K×K block is:
  #   Hᵢ = diag(pᵢ) - pᵢ pᵢᵀ
  # where pᵢ = P[:,i] is the softmax probability vector.
  #
  # So: (∇²f · v) reshaped as p×K has column k equal to:
  #   Σᵢ aᵢ · [pₖᵢ (vᵀ aᵢ)_k - pₖᵢ Σⱼ pⱼᵢ (vᵀ aᵢ)_j]  ... for each k
  #
  # More efficiently: let R = V' A (K×n), so R[k,i] = vₖᵀ aᵢ.
  # Then for each sample i, the Hessian-weighted output is:
  #   qₖᵢ = pₖᵢ (Rₖᵢ - Σⱼ pⱼᵢ Rⱼᵢ)
  # And the result is A * Q'  (p×K), where Q is K×n.
  function hprod!(hv, x, v; obj_weight = 1)
    compute_scores!(S, x)
    compute_softmax!(P, S)

    V = reshape(v, p, K)
    # R = V' * A  (K×n): R[k,i] = vₖᵀ aᵢ
    mul!(tmp, V', A)   # reuse tmp as R (K×n)
    R = tmp

    # Q[k,i] = P[k,i] * (R[k,i] - Σⱼ P[j,i] R[j,i])
    # Overwrite S as Q (reuse buffer)
    Q = S
    @inbounds for i in 1:n
      # dot product pᵢᵀ rᵢ
      pr = zero(T)
      for k in 1:K
        pr += P[k, i] * R[k, i]
      end
      for k in 1:K
        Q[k, i] = P[k, i] * (R[k, i] - pr)
      end
    end

    HV = reshape(hv, p, K)
    mul!(HV, A, Q')   # HV = A * Q', size p×K

    if obj_weight != 1
      rmul!(hv, obj_weight)
    end
    return hv
  end

  x0 = zeros(T, p * K)

  return ManualNLPModels.NLPModel(
    x0,
    obj;
    grad = grad!,
    hprod = hprod!,
    meta_args = Dict(:name => "Multinomial"),
  )
end
