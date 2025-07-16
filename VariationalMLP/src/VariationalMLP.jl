module VariationalMLP

using Flux
import Flux: logitcrossentropy, mse, trainable
using Random
using Functors

export VariationalDropoutMolchanov, VariationalDropoutKingma, VarChain,
       energy_loss, kl, make_model, model_sparsity, layer_sparsity,
       make_layer, make_variational

abstract type AbstractVariationalLayer end

# ========== 1. Layer Types =============

# mutable
struct VariationalDense{F} <: AbstractVariationalLayer
    μi::Matrix{Float32}
    logσ2i::Matrix{Float32}
    bias::Vector{Float32}
    activation::F 
end

Functors.@functor VariationalDense (μi,logσ2i, bias)

# Explicitly define trainable fields
trainable(layer::VariationalDense) = (; μi=layer.μi, logσ2i=layer.logσ2i, bias=layer.bias)


struct VariationalDropoutMolchanov{F} <: AbstractVariationalLayer
    θ::Matrix{Float32}
    logσ2::Matrix{Float32}
    bias::Vector{Float32}
    activation::F
end

Functors.@functor VariationalDropoutMolchanov (θ, logσ2, bias)
trainable(layer::VariationalDropoutMolchanov) = (; θ=layer.θ, logσ2=layer.logσ2, bias=layer.bias)




struct VariationalDropoutKingma{F} <: AbstractVariationalLayer
    μ::Matrix{Float32}
    logα::Matrix{Float32}
    bias::Vector{Float32}
    activation::F
end

Functors.@functor VariationalDropoutKingma (μ, logα, bias)
trainable(layer::VariationalDropoutKingma) = (; μ=layer.μ, logα=layer.logα, bias=layer.bias)

# ========== 2. Constructors ==============

# Helper function to convert pre-trained weights
# to mean and variance

function pretrain_init(pretrained_W, pretrained_b)

    θ_or_μ = copy(pretrained_W)

    # zero variance ensures weights
    # are initialised as per pre-trained values
    init_logσ2_or_logα = fill(0f0, size(pretrained_W))

    b = copy(pretrained_b)

    return θ_or_μ, init_logσ2_or_logα, b

end 



function make_variational(in::Int, out ::Int;
    activation = identity, 
    parameterisation=:molchanov,
    init = :random,
    pretrained_W = nothing,
    pretrained_b = nothing)

    # choose init strategy
    function init_params()

        if init ==:random 
            return randn(Float32, out, in), randn(Float32, out, in), zeros(Float32, out)


        elseif init ==:custom
            @assert pretrained_W !==nothing "Pretrained W must be provided for custom initialisation"
            @assert pretrained_b !==nothing "Pretrained b must be provided for custom initialisation"
            return pretrain_init(pretrained_W, pretrained_b)

        else
            error("Unsupported init method. Use :random, :custom")

        end 

    end

    # initialize the mean and variance accordingly
    mean_param, variance_param, bias = init_params()

    if parameterisation ==:molchanov 

        return VariationalDropoutMolchanov(mean_param, variance_param, bias, activation)

    elseif parameterisation ==:kingma

        return VariationalDropoutKingma(mean_param, variance_param,  bias, activation)

    elseif parameterisation ==:graves

        return VariationalDense(mean_param, variance_param, bias, activation)

    else
        error("Unsupported parameterisation: use :graves, :molchanov or :kingma")

    end

end 


# ========== 3. Forward Pass ==========

function (layer::VariationalDense)(x::AbstractMatrix)
  
    # sample noise N(0, 1)
    ϵ = randn(Float32, size(layer.μi))

    # sample weights; all operations element-wise
    W = @. layer.μi + exp(0.5f0 * layer.logσ2i) * ϵ

    return layer.activation.(W * x .+ layer.bias)

end 


function (layer::VariationalDropoutMolchanov)(x::AbstractMatrix)



    # sample noise N(0, 1)
    ϵ = randn(Float32, size(layer.θ))

    # sample weights; all operations element-wise
    W = @. layer.θ + exp(0.5f0 * layer.logσ2) * ϵ

    return layer.activation.(W * x .+ layer.bias)

end

function (layer::VariationalDropoutKingma)(x::AbstractMatrix)

    # sample noise N(0, 1)
    ϵ = randn(Float32, size(layer.μ))

    # sample weights as per Kingma et al. (2015)
    # μ + √exp(log(α)) μ ϵ
    W = @. layer.μ + sqrt(exp(layer.logα)) * layer.μ * ϵ

    return layer.activation.(W * x .+ layer.bias)
end

# ========== 4. VarChain ==========

struct VarChain{L}
    layers::L
end

VarChain(layers...) = VarChain(layers)
@functor VarChain

function (c::VarChain)(x)
    @inbounds for l in c.layers
        x = l(x)
    end
    x
end

# ========== 5. KL Divergence ==========

const k1, k2, k3 = 0.63576f0, 1.87320f0, 1.48695f0 # Molchanov et al.(2017)
const c1, c2, c3 = 1.16145124f0, -1.50204118f0, 0.58629921f0 # Kingma et al.(2015, p.6)
const μ_prior, σ2_prior = 0.0f0, 2.0f0 # Adaptive mean L2 Graves(2011, p.7)

function kl(layer::VariationalDense)

    # lnσ - 0.5 * lnσ²_i

    kl_ij = @. log(σ2_prior) - 0.5f0 *  layer.logσ2i  + 0.5f0 * (1.0f0 / σ2_prior^2 ) * ((layer.μi - μ_prior)^2 + exp(layer.logσ2i) - σ2_prior^2)

    return sum(kl_ij)

end 


function kl(layer::VariationalDropoutMolchanov)

    # lnσ² - ln θ²
    logα = @. layer.logσ2 - 2f0 * log(abs(layer.θ) + 1f-8)
        
    return sum(-1f0 * (-k1 .+ k1 * sigmoid.(k2 .+ k3 * logα) .- 0.5f0 * log1p.(1.0f0 ./ exp.(logα))))
end

function kl(layer::VariationalDropoutKingma)

    # exp(lnα)
    α = @. exp(layer.logα)

    return sum(-1f0 * (0.5f0 * layer.logα + c1 * α + c2 * α.^2 + c3 * α.^3))
end

# ========== 6. Loss Function ==========

function energy_loss(model::VarChain, x, y, N; kl_scale = 1.0f0, enable_warmup =:true, task_type::Symbol=:classification)

    # Eq. (3) in Molchanov et al. (2017); logitcrossentropy averages over batch M


    if task_type ==:classification

        nll = N * logitcrossentropy(model(x), y)

    elseif task_type ==:regression

        nll = N * mse(vec(model(x)), y)

    else 
        error("Not supported. Use :classification or :regression")

    end

   
    # We sum across layers; automatically select the correct kl based on layers
    kl_sum = sum(kl(layer) for layer in model.layers)

    # Modified version, for warmup
    if enable_warmup

        var_energy = nll + kl_scale * (kl_sum/N)

    else 
        var_energy = nll + kl_sum/ N

    end 

    return var_energy


end

# ========== 7. Model Constructor ==========

function make_model(sizes::Vector{Int};
                    activations::Vector = repeat([relu], length(sizes) - 2),
                    final_activation = identity,
                    variant::Symbol = :molchanov,
                    init::Symbol = :random,
                    pretrained_Ws = nothing,
                    pretrained_bs = nothing)

    @assert length(sizes) ≥ 2
    @assert length(activations) == length(sizes) - 2

    if init == :custom
        @assert pretrained_Ws !== nothing "You must provide pretrained_Ws for :custom init"
        @assert pretrained_bs !== nothing "You must provide pretrained_bs for :custom init"
        @assert length(pretrained_Ws) == length(sizes) - 1
        @assert length(pretrained_bs) == length(sizes) - 1
    end

    layers = Vector{AbstractVariationalLayer}(undef, length(sizes) - 1)

    for i in 1:length(sizes) - 2
        W = init == :custom ? pretrained_Ws[i] : nothing
        b = init == :custom ? pretrained_bs[i] : nothing
        layers[i] = make_layer(sizes[i], sizes[i+1];
                               activation = activations[i],
                               variant = variant,
                               init = init,
                               pretrained_W = W,
                               pretrained_b = b)
    end

    # Last layer
    W = init == :custom ? pretrained_Ws[end] : nothing
    b = init == :custom ? pretrained_bs[end] : nothing
    layers[end] = make_layer(sizes[end-1], sizes[end];
                             activation = final_activation,
                             variant = variant,
                             init = init,
                             pretrained_W = W,
                             pretrained_b = b)

    return VarChain(layers)
end

function make_layer(in::Int, out::Int;
                    activation = identity,
                    variant = :molchanov,
                    init = :random,
                    pretrained_W = nothing,
                    pretrained_b = nothing)

    return make_variational(in, out;
                            activation = activation,
                            parameterisation = variant,
                            init = init,
                            pretrained_W = pretrained_W,
                            pretrained_b = pretrained_b)
end

# =========== 8. Sparsity ===========

const SPARSITY_THRESHOLD_α = 3.0f0 
const SPARSITY_THREHOLD_λ = 0.83f0

@inline function layer_sparsity(layer::VariationalDropoutMolchanov; custom_threshold = SPARSITY_THRESHOLD_α)
    logα = @. layer.logσ2 - 2f0 * log(abs(layer.θ) + 1f-8)
    return count(x -> x ≥ custom_threshold, logα) / length(logα)
end

@inline function layer_sparsity(layer::VariationalDropoutKingma; custom_threshold = SPARSITY_THRESHOLD_α)
    return count(x -> x ≥ custom_threshold, layer.logα) / length(layer.logα)
end

@inline function layer_sparsity(layer::VariationalDense; custom_threshold = SPARSITY_THRESHOLD_λ)
    λ = @.  abs(layer.μi / exp(0.5f0 * layer.logσ2i))
    return count(x -> x ≤ custom_threshold, λ) / length(λ)
end


function model_sparsity(model::VarChain; custom_threshold = SPARSITY_THRESHOLD_α )
    [layer_sparsity(l; custom_threshold = custom_threshold) for l in model.layers]
end

# end of module
end 




