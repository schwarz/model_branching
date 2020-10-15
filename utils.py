import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def closure(model, loss_fn, optimizer, input, target):
    def c():
        optimizer.zero_grad()
        out = model(input)
        loss = loss_fn(out, target)
        loss.backward()  # Compute the gradients and store in the model param tensors (.grad, .requires_grad)
        return loss

    return c


def collect_batch_of_batches(iterators):
    """Take one mini-batch from all iterators and return the concatenated batches.
    Also return a list of all the alive iterators, [] if none."""
    if len(iterators) == 0:
        return (None, None), []
    acc_inputs = None
    acc_targets = None
    alive_iterators = []
    for i in iterators:
        try:
            (batched_inputs, batched_targets) = next(i)
            alive_iterators.append(i)
            if acc_inputs is None:
                acc_inputs = batched_inputs
                acc_targets = batched_targets
            else:
                acc_inputs = torch.cat((acc_inputs, batched_inputs), 0)
                acc_targets = torch.cat((acc_targets, batched_targets), 0)
        except StopIteration:
            pass
    return (acc_inputs, acc_targets), alive_iterators
