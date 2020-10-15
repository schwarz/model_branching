import argparse
import copy
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from custom_dataset import SlidingWindowDataset
from nn.lstm import SequenceLSTM

import node
import parameter_server
import perfect_node

import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model branching experiment.")
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=4,
        help="how many data points to load from each producer",
    )

    parser.add_argument(
        "--scale_divisor",
        type=int,
        default=1,
        help="divide all our data by this number",
    )

    parser.add_argument(
        "--window_size", type=int, default=50, help="how long each time series is",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=-1,
        help="limit the number of iterations to this",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="enable debug logging",
    )
    parser.add_argument(
        "--experiment_name", default="temp", help="unique name of the experiment",
    )
    parser.add_argument(
        "--dataset", default="turnstile", help="name of the dataset",
    )
    parser.add_argument(
        "--assignment",
        default="turnstile_10_stations.json",
        help="path to the file containing node assignments",
    )
    parser.add_argument(
        "--model_params", help="path to the file containing model parameters",
    )
    parser.add_argument(
        "--loss_function",
        default="rmsle",
        help="loss function to use in training (mse|rmsle|mae)",
    )
    parser.add_argument(
        "--column", default="entries", help="the y column of the timeseries",
    )
    parser.add_argument(
        "--perfect_node",
        dest="perfect_node",
        action="store_true",
        help="if given run the perfect node",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="learning rate to use at the nodes",
    )

    parser.add_argument(
        "--local_epochs", type=int, default=1, help="number of local epochs",
    )

    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    torch.manual_seed(4)  # random die throw
    mp.set_start_method("spawn", force=True)

    with open(args.assignment) as f:
        paths = json.load(f)
    num_processes = len(paths)

    path_prefix = "./data/{}/test/".format(args.dataset)

    device = None
    if not args.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA")
        device = torch.device("cuda")
    else:
        logging.info("Using CPU")
        device = torch.device("cpu")

    if not os.path.isdir("experiments/" + args.dataset):
        os.mkdir("experiments/" + args.dataset)
    RESULTS_DIR = "experiments/{}/{}".format(args.dataset, args.experiment_name)
    if not os.path.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    processes = []

    ps_model = SequenceLSTM(1, 100).double()
    ps_model.load_state_dict(torch.load(args.model_params))
    ps_model.to(device=device)

    q_from_nodes = mp.SimpleQueue()
    q_to_perfect_node = mp.Queue()
    qs = {rank: mp.SimpleQueue() for rank in range(num_processes)}

    ps = mp.Process(
        target=parameter_server.loop,
        args=(q_from_nodes, qs, ps_model, args.iterations, device,),
    )
    ps.start()
    processes.append(ps)

    if args.perfect_node:
        pn_model = copy.deepcopy(ps_model)
        pn_model.to(device=device)
        logging.info(args.perfect_node)
        pn = mp.Process(
            target=perfect_node.loop,
            args=(
                pn_model,
                q_to_perfect_node,
                args.learning_rate,
                args.experiment_name,
                args.dataset,
                args.loss_function,
                device,
                args.scale_divisor,
                args.verbose,
            ),
        )
        pn.start()
        processes.append(pn)
    else:
        logging.info("PN is skipped")

    # Start the nodes
    for rank in range(num_processes):
        p = mp.Process(
            target=node.loop,
            args=(
                rank,
                q_from_nodes,
                qs[rank],
                paths[rank],
                path_prefix,
                q_to_perfect_node,
                args.learning_rate,
                device,
                args.mini_batch_size,
                args.window_size,
                args.loss_function,
                args.scale_divisor,
                args.iterations,
                args.experiment_name,
                args.dataset,
                args.column,
                args.verbose,
                args.perfect_node,
                args.local_epochs,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
