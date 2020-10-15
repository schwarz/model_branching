"""Similar to node, but receives data from nodes to create the 
'perfect' global model. No synchronization here.
"""

import csv
import logging
import os
import queue
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy

import messages
import utils
from custom_dataset import SlidingWindowDataset
from nn.lstm import SequenceLSTM
from nn.utils import RMSLELoss

# Note: When msg.iteration > iteration, we already at least know the base model and data for iteration 0
def loop(
    initial_model,
    q_from_nodes,
    learning_rate,
    experiment,
    dataset,
    loss_function,
    device,
    scale_divisor,
    log_verbose=False,
):
    model = copy.deepcopy(initial_model)
    del initial_model

    opti = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = {"mse": nn.MSELoss(), "rmsle": RMSLELoss(), "mae": nn.L1Loss()}[
        loss_function
    ]
    if log_verbose:
        logging.basicConfig(
            format=" [%(levelname)s] PN: %(message)s", level=logging.DEBUG
        )
    else:
        logging.basicConfig(
            format=" [%(levelname)s] PN: %(message)s", level=logging.INFO
        )

    iteration = 0
    data = {}
    if not os.path.isdir("experiments/" + dataset):
        os.mkdir("experiments/" + dataset)
    if not os.path.isdir("experiments/{}/{}".format(dataset, experiment)):
        os.mkdir("experiments/{}/{}".format(dataset, experiment))
    log_file = open(
        "./experiments/{}/{}/acc_global.csv".format(dataset, experiment), mode="w"
    )
    log_writer = csv.writer(
        log_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    log_writer.writerow(["iteration", "id", "model_key", "loss"])
    while True:
        msg = None
        try:
            msg = q_from_nodes.get(True, 30)
        except queue.Empty:
            logging.debug("Nothing in data queue, final round of eval.")

        if msg is None or msg.iteration > iteration:
            # Timeout (= last round) or new iteration detected
            # If the iteration is higher, we assume all nodes have moved on due to PS sync
            logging.debug(
                "Evaluating perfect model against iteration={}".format(iteration)
            )

            # Evaluate the model against the data from each node so we can compare it
            with torch.no_grad():
                model.eval()
                for id, inputs_targets in data.items():
                    (inputs, targets) = inputs_targets
                    pred = model(inputs)
                    loss = loss_fn(pred * scale_divisor, targets * scale_divisor)
                    logging.info(
                        "Prediction measure iteration={} node={} loss={} ".format(
                            iteration, id, loss
                        )
                    )
                    log_writer.writerow([iteration, id, "global", loss.item()])

            # Concat the batch from each node and run a training step
            inps = []
            targs = []
            for inputs_targets in data.values():
                (i, t) = inputs_targets
                inps.append(i)
                targs.append(t)
            inputs = torch.cat(inps, 0)
            targets = torch.cat(targs, 0)

            model.train()
            utils.closure(model, loss_fn, opti, inputs, targets)()
            opti.step()

            data.clear()

        if not msg is None:
            data[msg.id] = (msg.inputs, msg.targets)
            if msg.iteration > iteration:
                iteration = msg.iteration
        else:
            # Final iteration completed, no more data
            break

    log_file.close()
    logging.info("Exiting")
