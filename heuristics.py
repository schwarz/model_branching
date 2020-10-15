import random
import torch

# Return the name of the model that is best for the next batch
def oracle(loss_fn, _latest_batches, future, models):
    # Oracle cheats by using loss_fn and models on future inputs and targets
    (future_inputs, future_targets) = future

    losses = []
    with torch.no_grad():
        for (model_key, model_actual) in models.items():
            model_actual.eval()
            pred = model_actual(future_inputs)
            loss = loss_fn(pred, future_targets).item()
            losses.append((model_key, loss))
    losses.sort(key=lambda tup: tup[1])
    return losses[0][0]


# Always return the synced model
def always_synced(_loss_fn, _latest_batches, _future, _models):
    return "synced"


# Same as best_historic but with an increasing * 1.1 multiplier for newer batches
def best_improving(loss_fn, latest_batches, _future, models):
    losses = {}
    for model_key, model_actual in models.items():
        multiplier = 1.0
        with torch.no_grad():
            model_actual.eval()
            for b in latest_batches:
                (inputs, targets) = b
                pred = model_actual(inputs)
                loss = loss_fn(pred, targets).item() * multiplier

                if model_key in losses:
                    losses[model_key] = (
                        losses[model_key] + (1.0 / len(latest_batches)) * loss
                    )
                else:
                    losses[model_key] = (1.0 / len(latest_batches)) * loss
                multiplier = multiplier * 1.1

    sorted_losses = sorted(losses.items(), key=lambda item: item[1])
    return sorted_losses[0][0]


# Return the name of the model with the lowest summed loss for all batches
def best_historic(loss_fn, latest_batches, _future, models):
    losses = {}
    for model_key, model_actual in models.items():
        with torch.no_grad():
            model_actual.eval()
            for b in latest_batches:
                (inputs, targets) = b
                pred = model_actual(inputs)
                loss = loss_fn(pred, targets).item()

                if model_key in losses:
                    losses[model_key] = (
                        losses[model_key] + (1.0 / len(latest_batches)) * loss
                    )
                else:
                    losses[model_key] = (1.0 / len(latest_batches)) * loss

    sorted_losses = sorted(losses.items(), key=lambda item: item[1])
    return sorted_losses[0][0]


# Return the name of the model with the lowest loss for the previous batch
def best_last(loss_fn, latest_batches, _future, models):
    return best_historic(loss_fn, latest_batches[-1:], _future, models)


# Return the name of the model with the lowest loss for the previous 2 batches
# This makes the model slower to react but might be more stable
def best_rolling(loss_fn, latest_batches, _future, models):
    return best_historic(loss_fn, latest_batches[-2:], _future, models)


# Return the name of a random model
def coinflip(_loss_fn, _latest_batches, _future, models):
    return random.choice(list(models.keys()))


# Return the purely local model
# Note: The local model is not part of the passed models dict
def local(_loss_fn, _latest_batches, _future, _models):
    return "local"


# Return the never changing static model
# Note: The static model is not part of the passed models dict
def static(_loss_fn, _latest_batches, _future, _models):
    return "static"
