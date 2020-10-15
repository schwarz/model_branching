import numpy as np
import math
import os
import pandas as pd
import random
import json

random.seed(4)
np.random.seed(4)

pattern_length = 7

# First and last are always baseline
up_down = np.array([0.5, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5])
down_up = np.flip(up_down)

long_up = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
long_down = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

smushed_up_down = np.array([0.5, 0.75, 0.75, 0.5, 0.25, 0.25, 0.5])
smushed_down_up = np.flip(smushed_up_down)

up_z = np.array([0.5, 0.5, 1.0, 0.5, 0, 0.5, 0.5])
down_z = np.flip(up_z)

spikes_up = np.array([0.5, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5])
spikes_down = np.array([0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5])

fixed_curve = lambda n: np.repeat(np.array([n]), 7)

tall_ziggurat = np.array([0.5, 0.75, 1.0, 1.0, 1.0, 0.75, 0.5])
short_ziggurat = np.array([0.5, 0.625, 0.75, 0.75, 0.75, 0.625, 0.5])

# Sensor might have died
def zero_mutation(a):
    return a * 0.0


# Behavior change
def upper_halved_mutation(a):
    amin = min(a)
    amax = max(a)
    threshold = np.median(a)
    for i in range(len(a)):
        if a[i] >= threshold:
            a[i] = threshold + ((a[i] - threshold) * 0.5)
    return a


# Behavior change
def lower_halved_mutation(a):
    amin = min(a)
    amax = max(a)
    threshold = np.median(a)
    for i in range(len(a)):
        if a[i] < threshold:
            a[i] = threshold + ((a[i] - threshold) * 0.5)
    return a


unused_patterns = {
    "down_z": down_z,
    "spikes_down": spikes_down,
    "fixed_00": fixed_curve(0.0),
    "fixed_10": fixed_curve(1.0),
    "down_up": down_up,
    "smushed_up_down": smushed_up_down,
}

patterns = {
    "up_down": up_down,
    "long_up": long_up,
    "long_down": long_down,
    "smushed_down_up": smushed_down_up,
    "up_z": up_z,
    "spikes_up": spikes_up,
    "short_ziggurat": short_ziggurat,
    "tall_ziggurat": tall_ziggurat,
    "fixed_05": fixed_curve(0.5),
}

print(
    "There are",
    len(patterns),
    "patterns in use and",
    len(unused_patterns),
    "unused patterns",
)


train_n = 1400
valid_n = 200
test_n = 400

test = {}
train = {}
valid = {}

repeats = int(2002 / 7)  # End result: 2000 rows per file
curves_wanted = 1000

for c in range(curves_wanted):
    str_id = f"{c:03}"
    pattern_name = random.choice(list(patterns.keys()))
    pattern = patterns[pattern_name]
    patterned_baseline = np.tile(pattern, repeats)[:2000]

    noisy_baseline = None
    if "fixed" not in pattern_name:
        noisy_baseline = patterned_baseline * np.random.normal(
            1, 0.05, len(patterned_baseline)
        )
    else:
        noisy_baseline = patterned_baseline

    clamped_baseline = np.clip(noisy_baseline, 0, 1)
    scaled_baseline = clamped_baseline * 0.5
    offset_baseline = scaled_baseline + (np.random.uniform() * 0.50)

    trended_baseline = None
    if "fixed" not in pattern_name:
        trended_baseline = offset_baseline * np.linspace(
            1, 1.1, num=len(offset_baseline)
        )
    else:
        trended_baseline = offset_baseline

    # Safety clamp
    final_clamped_curve = np.clip(trended_baseline, 0.025, 0.975)

    train_df = pd.DataFrame({"y": final_clamped_curve[:train_n]})
    valid_df = pd.DataFrame({"y": final_clamped_curve[train_n : train_n + valid_n]})
    test_df = pd.DataFrame({"y": final_clamped_curve[train_n + valid_n :]})

    train["{}_{}".format(str_id, pattern_name)] = train_df
    valid["{}_{}".format(str_id, pattern_name)] = valid_df
    test["{}_{}".format(str_id, pattern_name)] = test_df

# Mutate 10 percent of the test dataset
num_mutated = int(len(test) * 0.1)
mutation_choices = [
    "upper_halved",
    "upper_halved",
    "lower_halved",
    "lower_halved",
    "zeroed",
]

for key in list(random.sample(list(test.keys()), num_mutated)):
    mutation = random.choice(mutation_choices)
    m_fn = None
    if mutation == "upper_halved":
        m_fn = upper_halved_mutation
    elif mutation == "lower_halved":
        m_fn = lower_halved_mutation
    elif mutation == "zeroed":
        m_fn = zero_mutation

    df = test[key]
    df["y"] = m_fn(df["y"])
    del test[key]
    test[key + "_" + mutation] = df

train_ks = list(train.keys())
train_ks.sort()

valid_ks = list(valid.keys())
valid_ks.sort()

test_ks = list(test.keys())
test_ks.sort()

if not os.path.isdir("train"):
    os.mkdir("train")
if not os.path.isdir("valid"):
    os.mkdir("valid")
if not os.path.isdir("test"):
    os.mkdir("test")

for (ds_name, ds) in [("train", train), ("valid", valid), ("test", test)]:
    for name, df in ds.items():
        df.to_feather("{}/{}.feather".format(ds_name, name))


for (k, vs) in [("train", train_ks), ("valid", valid_ks), ("test", test_ks)]:
    with open("{}.json".format(k), "w") as outfile:
        json.dump([v + ".feather" for v in vs], outfile)
