import copy
import csv
import logging
import messages
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import time

from nn.lstm import SequenceLSTM
from torch.utils.data import Dataset, DataLoader
from nn.utils import RMSLELoss

from custom_dataset import SlidingWindowDataset
import heuristics as heurs


def __model_key(generation, id):
    return "synced.{}/node.{}".format(generation, id)


def loop(
    id,
    q_to_ps,
    q_from_ps,
    paths,
    path_prefix,
    q_to_perfect_node,
    learning_rate,
    device,
    batch_size,
    window_size,
    loss_function,
    scale_divisor=1,
    iterations_limit=-1,
    experiment="changeme",
    dataset="turnstile",
    value_column="entries",
    log_verbose=False,
    feed_perfect_node=False,
    local_epochs=1,
):
    # Store models with the generation they were created in
    # e.g. synced.0/node1 # Received the model for generation 0 (includes that data), from then on we're branched
    models = {}
    heuristics = {
        "synced": heurs.always_synced,
        "greedy_oracle": heurs.oracle,
        "best_last": heurs.best_last,
        "best_last_2": heurs.best_rolling,
        "best_last_3": heurs.best_historic,
        "best_improving": heurs.best_improving,
        "random": heurs.coinflip,
        "local": heurs.local,
        "static": heurs.static,
    }
    # maps heuristic name to a generation model key if better than synced
    heuristic_model_mapping = {}
    latest_batches = []
    iteration = -1
    loss_fn = {"mse": nn.MSELoss(), "rmsle": RMSLELoss(), "mae": nn.L1Loss()}[
        loss_function
    ]

    if log_verbose:
        logging.basicConfig(
            format=" [%(levelname)s] N{} %(message)s".format(id), level=logging.DEBUG
        )
    else:
        logging.basicConfig(
            format=" [%(levelname)s] N{} %(message)s".format(id), level=logging.INFO
        )

    RESULTS_DIR = "experiments/{}/{}".format(dataset, experiment)
    quality_file = open("{}/acc_node_{}.csv".format(RESULTS_DIR, id), mode="w")
    quality_writer = csv.writer(
        quality_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    quality_writer.writerow(["iteration", "id", "model_key", "loss"])

    heuristics_file = open("{}/heu_node_{}.csv".format(RESULTS_DIR, id), mode="w")
    heuristics_writer = csv.writer(
        heuristics_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    heuristics_writer.writerow(["iteration", "id", "heuristic", "model"])

    iterators = []
    logging.info("Running")
    while True:
        # Get assembled batch of data from all our data streams
        (inputs, targets), iterators = utils.collect_batch_of_batches(iterators)
        if not inputs is None:
            logging.debug(
                "Data loaded iteration={} num_datapoints={}".format(
                    iteration, len(inputs)
                )
            )
        else:
            logging.debug(
                "No data loaded iteration={} num_datapoints=0".format(iteration)
            )
            if len(iterators) == 0:
                if len(paths) > 0:
                    # Load next set of iterators
                    next_paths = paths.pop(0)
                    logging.info("Loading files: " + ", ".join(next_paths))
                    # paths = [  [],  [],  [],  [] ]
                    # Take one p
                    dfs = [
                        pd.read_csv(path_prefix + f)
                        if f.endswith(".csv")
                        else pd.read_feather(path_prefix + f)
                        for f in next_paths
                    ]
                    dls = [
                        DataLoader(
                            SlidingWindowDataset(
                                torch.from_numpy(
                                    v[value_column].to_numpy() / scale_divisor
                                )
                                .double()
                                .to(device=device),
                                window_size,
                            ),
                            shuffle=False,
                            batch_size=batch_size,
                        )
                        for v in dfs
                    ]
                    iterators = [iter(dl) for dl in dls]
                    logging.debug("Trying again with new iterators")
                    # Try again with this new set of iterators
                    continue

                else:
                    # We're done, can't use any other iterators
                    logging.info("Retiring node - no more data to load.")
                    q_to_ps.put(
                        messages.GradientsUpdateMessage(
                            params=[], num_datapoints=0, node=id
                        )
                    )
                    time.sleep(1)  # for safety
                    break

        logging.debug("Waiting to receive state_dict from parameter server")
        (iteration, state_dict) = q_from_ps.get()
        assert iteration != -1
        logging.debug("state_dict received iteration={}".format(iteration))
        if iterations_limit != -1 and iteration >= iterations_limit:
            logging.info(
                "Exceeded debug iteration limit of {}.".format(iterations_limit)
            )
            break

        if feed_perfect_node:
            # Send data for this iteration to perfect node
            q_to_perfect_node.put(
                messages.DataForPerfectNodeMessage(
                    id=id, iteration=iteration, inputs=inputs, targets=targets
                )
            )

        synced_model = SequenceLSTM(1, 100).double()
        synced_model.load_state_dict(state_dict)
        synced_model.to(device=device)
        del state_dict

        if iteration == 0:
            # Add the first received model to our store for the local and static models
            models["local"] = copy.deepcopy(synced_model)
            models["static"] = copy.deepcopy(synced_model)

        # Heuristics on known models and new synced model
        # We don't know the current data here!
        # Check to make sure heuristics work and a local model also exists in the models dict
        if len(latest_batches) >= 3:
            for h_key, h_fn in heuristics.items():
                local_key = __model_key(iteration - 1, id)
                local_model = models.get(local_key, None)
                compare_models = {
                    "synced": synced_model,
                    local_key: local_model,
                }
                heuristic_model = models.get(
                    heuristic_model_mapping.get(h_key, None), None
                )
                if heuristic_model:
                    compare_models[
                        heuristic_model_mapping.get(h_key, None)
                    ] = heuristic_model

                best_key = h_fn(
                    loss_fn, latest_batches, (inputs, targets), compare_models
                )
                heuristic_model_mapping[h_key] = best_key

                heuristics_writer.writerow([iteration, id, h_key, best_key])

            # Update models to only include those that are in the heuristic model mapping
            models = {
                k: v for k, v in models.items() if k in heuristic_model_mapping.values()
            }

        # Exclusively for analysis
        # Evaluate all currently known models against the new data
        # Only those models we've staid with (branches) + current synced model
        with torch.no_grad():
            for model_key, model_actual in list(models.items()) + [
                ("synced", synced_model)
            ]:
                model_actual.eval()
                pred = model_actual(inputs)
                loss = loss_fn(pred * scale_divisor, targets * scale_divisor)
                logging.debug(
                    "Inference quality iteration={} model={} loss={}".format(
                        iteration, model_key, loss
                    )
                )
                quality_writer.writerow([iteration, id, model_key, loss.item()])

        for epoch in range(local_epochs):
            # Only those models we're continuing (branches)
            # Optimizers are lost between epochs, slightly worsening results
            for (model_key, model_actual) in models.items():
                if model_key != "static":
                    model_actual.train()
                    opti = optim.Adam(model_actual.parameters(), lr=learning_rate)
                    utils.closure(model_actual, loss_fn, opti, inputs, targets)()
                    opti.step()

            # Train synced model and send the state dict
            synced_model.train()
            synced_optimizer = optim.Adam(synced_model.parameters(), lr=learning_rate)
            utils.closure(synced_model, loss_fn, synced_optimizer, inputs, targets)()
            synced_optimizer.step()

        q_to_ps.put(
            messages.GradientsUpdateMessage(
                params=synced_model.state_dict(), num_datapoints=len(inputs), node=id
            )
        )
        logging.debug("Sent state_dict to PS iteration={}".format(iteration))

        # Store the synced model with training from this node
        models[__model_key(iteration, id)] = synced_model
        models.pop(__model_key(iteration, id - 1), None)  # TODO Inspect if necessary

        # Append at the end so that heuristics see the right last 3 batches
        latest_batches.append((inputs, targets))
        latest_batches = latest_batches[-3:]

    logging.info("Sleeping to exit after perfect global node")
    time.sleep(45)
    quality_file.close()
    heuristics_file.close()
