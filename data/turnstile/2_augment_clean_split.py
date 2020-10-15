# Augments net_entries and net_exits, splitting the dataset into:
# * train (< dec2019) and
# * valid (>= dec2019 < jan2020) and
# * test (>= jan2020)

# Relies on the files created by step 1
#
# There is some noise in the original dataset,
# additional readings might be inbetween the regular 4h intervals

# For training, it's ok when the data is a bit faulty

# Clean up the test data for the live experiments
# Objectives:
#   - 4 hour measurement interval
#   - No "debug" measurements
#   - No turnstiles that are decomissioned or have large gaps
# Solution: Simlpy remove all non-hourly measurements and check if there's enough data
# Caveats: Slight entry/exit delta inaccuracies whenever debug measurements are removed
#

import os
import pandas as pd
import json
from pathlib import Path

# 12 June 2020 - 1 January 2019. 528 days * 6 regular measurements
REGULAR_LEN = 528 * 6 - 20  # Arbitrary number, 20 measurement shifts

train_dfs = []
valid_dfs = []
test_dfs = []

for path in Path("extracted").rglob("*.csv"):
    station = str(path).split("/")[1]
    turnstile = str(path).split("/")[-1].split(".")[0]

    df = pd.read_csv(path, sep=",")
    if len(pd.concat([df[df["entries"].isnull()], df[df["exits"].isnull()]])) > 0:
        print("Dropped the following turnstile because of NAs:", str(path))
        continue

    df["observed_at"] = pd.to_datetime(df["observed_at"])
    df = df.sort_values("observed_at", axis=0, ascending=True)
    # The value can reset to zero, creating a negative difference.
    # Capped at 10_000, which shouldn't be reached in day to day measures
    df["net_entries"] = df["entries"].diff().abs().clip(0, 10_000)
    df["net_exits"] = df["exits"].diff().abs().clip(0, 10_000)
    df = df[1:]  # Skip the first row with a NaN
    df["entries"] = df["net_entries"].astype("int64")
    df["exits"] = df["net_exits"].astype("int64")

    df = df.drop(
        columns=["net_entries", "net_exits", "regular", "turnstile_id", "station"]
    )

    train_df = df.loc[(df["observed_at"] < "2018-12-01")].reset_index(drop=True)
    valid_df = df.loc[
        (df["observed_at"] >= "2018-12-01") & (df["observed_at"] < "2019-01-01")
    ].reset_index(drop=True)
    test_df = df.loc[(df["observed_at"] >= "2019-01-01")].reset_index(drop=True)

    if len(train_df) > 0:
        train_dfs.append((station, turnstile, train_df))
    if len(valid_df) > 0:
        valid_dfs.append((station, turnstile, valid_df))
    if len(test_df) > 0:
        test_dfs.append((station, turnstile, test_df))


final_test_dfs = []

for (station, turnstile, df) in test_dfs:
    df["observed_at"] = pd.to_datetime(df["observed_at"])
    df["m"] = df["observed_at"].dt.minute
    df["s"] = df["observed_at"].dt.second
    # Only regular measurements
    df = df[(df["m"] == 0) & (df["s"] == 0)]
    if len(df) < REGULAR_LEN:
        continue

    df = df.drop(["m", "s"], axis=1)
    df = df.reset_index(drop=True)
    final_test_dfs.append((station, turnstile, df))

for (t, dfs) in [("train", train_dfs), ("valid", valid_dfs), ("test", final_test_dfs)]:
    if not os.path.isdir(t):
        os.mkdir(t)
    for (station, turnstile, df) in dfs:
        if not os.path.isdir(t + "/" + station):
            os.mkdir(t + "/" + station)
        df.to_feather("{}/{}/{}.feather".format(t, station, turnstile))
    with open("{}.json".format(t), "w") as f:
        json.dump(
            [
                "{}/{}.feather".format(station, turnstile)
                for (station, turnstile, _df) in dfs
            ],
            f,
        )
