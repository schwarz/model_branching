import os
import pandas as pd
from datetime import datetime
import json

df = pd.read_csv("raw/residential_demands.csv", header=0, skiprows=1)

if not os.path.isdir("train"):
    os.mkdir("train")
if not os.path.isdir("valid"):
    os.mkdir("valid")
if not os.path.isdir("test"):
    os.mkdir("test")

filenames = []
max = 0
min = 20_000
for col in df.columns[1:]:
    household_df = df.loc[:, ["Time", col]].rename(
        columns={col: "Watts", "Time": "Timestamp"}
    )
    household_df["Timestamp"] = pd.to_datetime(
        household_df["Timestamp"], format="%d.%m.%Y %H:%M"
    )
    household_df = household_df[
        (household_df["Timestamp"] >= "2010-02-07")
        & (household_df["Timestamp"] < "2010-08-11")
    ]  # February to August

    household_df = household_df.resample("H", on="Timestamp").mean()  # average hourly

    colmax = household_df.loc[:, "Watts"].max()
    colmin = household_df.loc[:, "Watts"].min()
    if colmax > max:
        max = colmax
    if colmin < min:
        min = colmin

    filename = "household_{}.feather".format(str(col.split(" ")[1]).rjust(3, "0"))
    filenames.append(filename)

    # This split excludes the hottest months from the training dataset
    train_df = household_df[: int(len(household_df) * 0.7)]
    valid_df = household_df[int(len(household_df) * 0.7) : int(len(household_df) * 0.8)]
    test_df = household_df[int(len(household_df) * 0.8) :]
    train_df.reset_index().to_feather("train/{}".format(filename))
    valid_df.reset_index().to_feather("valid/{}".format(filename))
    test_df.reset_index().to_feather("test/{}".format(filename))

print("max", max)
print("min", min)

for t in ["train", "valid", "test"]:
    with open("{}.json".format(t), "w") as f:
        json.dump(["{}".format(filename) for filename in filenames], f)
