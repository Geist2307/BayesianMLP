module MCUtils

export mc_experiments, mc_preds, pred_stats, mc_sampling, prune_molchanov_model!, prune_kingma_model!, prune_dense!, random_prune!,
 layer_sparsity_calc_molchanov, layer_sparsity_calc_kingma, layer_sparsity_calc_dense

using Statistics: mean
using Statistics
using Plots
using Flux: onecold, softmax
using Random: randperm



# ============= 1. Classification MC =======================

function mc_preds(model, x, T, num_cls)

    """
    Applies model a number of T times and appends predictions.

    Args:

    model : trained model
    x: training data
    T: samples to be drawn/number of times to apply model
    num_cls : number of classes
    """

    preds = zeros(Float32, num_cls, size(x, 2), T)
    for t in 1:T
        preds[:, :, t] .= softmax(model(x))  # use probabilities
    end
    return preds
end

function pred_stats(mc_preds)

    """
    Returns mean and standard deviations of predictions
    """

    # mean of predictions
    μ = mean(mc_preds, dims = 3)[:, :, 1]

    # std of predictions (uncertainty in each of the final predictions)
    σ = std(mc_preds, dims = 3)[:, :, 1]

    return μ, σ

end

function mc_sampling(samples, model, X_test, y_test, num_cls)

    """
    Mean accuracy and standard deviation for a batch of predictions
    """

    # Get MC samples
    mc_predictions = mc_preds(model, X_test, samples, num_cls)

    # Posterior predictive mean and uncertainty in predictions
    μ, σ = pred_stats(mc_predictions)

    # Predicted classes from mean logits
    ŷ_class = onecold(μ, 0:num_cls-1)

    # Convert one-hot targets to class indices
    true_classes = onecold(y_test, 0:num_cls -1 )

    # Accuracy
    acc = mean(ŷ_class .== true_classes)

    return acc, μ, σ

end

function mc_experiments(n_repeats, step, min_samples, max_samples, model, X_test, y_test, num_cls; filename = "experiments")

    samples = min_samples:step:max_samples
    test_errors_mean = Float64[]
    test_errors_std = Float64[]


    for current_samples in samples

        # stores results of n repeats for fixed T
        mean_tes = Float32[]


        # repeat the sampling n_repeat times
        for _ in 1:n_repeats

           mc_predictions = mc_preds(model, X_test, current_samples, num_cls) 
         
            # Posterior predictive mean and uncertainty
            μ, σ = pred_stats(mc_predictions)

            # Predicted classes from mean logits
            ŷ_class = onecold(μ, 0:num_cls - 1)
            true_classes = onecold(y_test, 0:num_cls - 1)

            acc = mean(ŷ_class .== true_classes)
            mte = 1 - acc

            push!(mean_tes, mte) # collets n_repeats samples 
        end

    
        mean_te = mean(mean_tes)
        std_te = std(mean_tes)

        push!(test_errors_mean, mean_te * 100)
        push!(test_errors_std, std_te * 100)
    end

    test_errors_mean = Float64.(test_errors_mean)
    test_errors_std = Float64.(test_errors_std)

    println("Average test error per sample: $test_errors_mean")
    println("Standard deviations: $test_errors_std")

    fig = plot(samples, test_errors_mean,
        yerror = test_errors_std,
        marker = (:circle, 4),
        line = :solid,
        lw = 2,
        color = :blue,
        xlabel = "Number of MC samples for averaging",
        ylabel = "Test Classification Error (%)",
        title = "$filename",
        label = "Mean Test Error"
    )
    
    filename_png = filename * ".png"
    savefig(fig, filename_png)
    display(fig)
    println("Figure saved as $filename_png")


end

# ============== 2. Pruning ==============

function prune_molchanov_model!(model; logα_threshold = 3.0f0)
    for layer in model.layers
        if hasproperty(layer, :θ) && hasproperty(layer, :logσ2)
            
            _, mask = layer_sparsity_calc_molchanov(layer)
            # Zero out weights and variances at pruned positions
            layer.θ[mask] .= 0f0
            layer.logσ2[mask] .= -1f10
        end
    end
end

function prune_kingma_model!(model; logα_threshold = 3.0f0)
    for layer in model.layers
        if hasproperty(layer, :μ) && hasproperty(layer, :logα)
           
            _, mask = layer_sparsity_calc_kingma(layer)
            # Zero out weights and variances at pruned positions
            layer.μ[mask] .= 0f0
            layer.logα[mask] .= -1f10
        end
    end
end

function prune_dense!(model; λ_threshold = 0.83f0)
    for layer in model.layers
        if hasproperty(layer, :μi) && hasproperty(layer, :logσ2i)
           
            _, mask = layer_sparsity_calc_dense(layer)

            # Zero out weights and variances at pruned positions
            layer.μi[mask] .= 0f0
            layer.logσ2i[mask] .= -1f10
        end
    end
end


## Random pruning based on percentages 


function random_prune!(model)

    for layer in model.layers

        if hasproperty(layer, :θ) && hasproperty(layer, :logσ2)

            -, mask = layer_sparsity_calc_molchanov(layer)
        
            n_pruned = sum(mask)

            if n_pruned > 0

                # get random indices 
                inds = randperm(length(mask))[1 : n_pruned]

                layer.θ[inds] .= 0f0
                layer.logσ2[inds] .= -1f10
            
            else
                continue
            end 


        elseif hasproperty(layer, :μi) && hasproperty(layer, :logσ2i)

            _, mask = layer_sparsity_calc_dense(layer)

            n_pruned = sum(mask)

            if n_pruned > 0

                # get random indices 
                inds = randperm(length(mask))[1 : n_pruned]

                layer.μi[inds] .= 0f0
                layer.logσ2i[inds] .= -1f10

            else
                continue
            end

        elseif hasproperty(layer, :μ) && hasproperty(layer, :logα)

            _, mask = layer_sparsity_calc_kingma(layer)

            n_pruned = sum(mask)

            if n_pruned > 0

                # get random indices 
                inds = randperm(length(mask))[1 : n_pruned]

                layer.μ[inds] .= 0f0
                layer.logα[inds] .= -1f10

            else
                continue
            end 

        end 

    end 

end 


# ============== 3. Sparsity Calculation =======================


function layer_sparsity_calc_molchanov(layer; α_thresh = 3.0f0)

    # retrieve matrix of logα
    logα = layer.logσ2 .- log.(layer.θ .^ 2 .+ 1f-8)

    # create a mask such that large values of alpha
    # result in pruned parameters

    mask = logα .> α_thresh

    sparsity_percent = sum(mask) / length(mask)
    return sparsity_percent, mask
end


function layer_sparsity_calc_kingma(layer;  α_thresh = 3.0f0)

    # create a mask such that large values of alpha
    # result in pruned parameters
    logα = layer.logα
    mask = logα .> α_thresh


    sparsity_percent = sum(mask) / length(mask)
    return sparsity_percent, mask
end

function layer_sparsity_calc_dense(layer; λ_thresh = 0.83f0)


    λ = @. abs(layer.μi / exp(0.5f0 * layer.logσ2i))

    mask = λ .< λ_thresh

    sparsity_percent = sum(mask) / length(mask)
    return sparsity_percent, mask
end


# End of module
end