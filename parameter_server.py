import collections
import logging
import time
import torch
import torch.optim as optim
import copy

import messages


# Note: In iteration 0, all nodes get a model from the PS. It's the only model they know
def loop(q_updates_from_nodes, qs_to_nodes, initial_model, iterations, device):
    """The parameter server starts with a model and receives models from worker nodes.
       Then it aggregates the model (weighted on datapoints) and sends back the state dict.
    """
    model = copy.deepcopy(initial_model)
    del initial_model

    logging.basicConfig(format=" [%(levelname)s] PS: %(message)s", level=logging.DEBUG)
    logging.debug("PS: Running")
    time.sleep(0.1)  # Give workers time
    iteration = 0
    while True:
        logging.debug("")  # Visual aid
        # Send out the latest model
        for q in qs_to_nodes.values():
            q.put((iteration, model.state_dict()))

        logging.debug(
            "Iteration started, awaiting updates iteration={}".format(iteration)
        )

        if iterations != -1 and iteration == iterations:
            logging.info("Configured number of iterations reached")
            break

        # We wait to get updates from all nodes
        updates = []
        updates_required = len(qs_to_nodes.items())
        while len(updates) < updates_required:
            msg = q_updates_from_nodes.get()
            if msg.num_datapoints > 0:
                updates.append((msg.params, msg.num_datapoints))
            else:
                qs_to_nodes.pop(msg.node)
                updates_required = updates_required - 1

        logging.debug(
            "Updates received - Synchronizing num_updates={}".format(len(updates))
        )

        if len(updates) == 0:
            logging.info("No data points in updates, exiting")
            del updates
            break

        # Aggregate models to make the new model
        total_n = sum([num_datapoints for (_params, num_datapoints) in updates])

        next_model = collections.OrderedDict()
        for (local_model, local_n) in updates:
            for key in model.state_dict().keys():
                if key in next_model:
                    next_model[key] += (local_n / total_n) * local_model[key]
                else:
                    next_model[key] = (local_n / total_n) * local_model[key]
        model.load_state_dict(next_model)
        model.to(device=device)
        iteration = iteration + 1

    # Give the nodes some time to shut down
    time.sleep(10)
    logging.info("Exiting")
