import csv
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

dir = "./raw/"

paths = os.listdir(dir)
list.sort(paths)  # start with oldest (ascending order)
paths = paths[1:]  # Skip the gitignore file
print("Paths to processes:", paths)

# go through each file, append every line we see to the file for this turnstile_id. Should be dated in right order
successes = []
for p in paths:
    full_path = dir + p
    print("File {}/{}: {}".format(len(successes) + 1, len(paths), full_path))
    try:
        df = pd.read_csv(
            full_path,
            sep=",",
            header=0,
            names=[
                "controlarea",
                "remoteunit",
                "scp",
                "station",
                "linename",
                "division",
                "date",
                "time",
                "desc",
                "entries",
                "exits",
            ],
            usecols=[
                "controlarea",
                "remoteunit",
                "scp",
                "station",
                "date",
                "time",
                "desc",
                "entries",
                "exits",
            ],
        )

        # unique id for individual turnstiles
        df["turnstile_id"] = (
            df["controlarea"] + "-" + df["remoteunit"] + "-" + df["scp"]
        )

        # normalize station names
        df["station"] = df["station"].str.replace(" ", "_")  # spaces
        df["station"] = df["station"].str.replace("B'WAY", "BROADWAY")  # bway
        df["station"] = df["station"].str.replace("/", "and")  # street/street

        # combine date and time
        date_time = df["date"] + " " + df["time"]
        df["observed_at"] = pd.to_datetime(date_time, format="%m/%d/%Y %H:%M:%S")

        # is_regular
        df["regular"] = df["desc"] == "REGULAR"

        # chunk by turnstile and append the data for each to a file
        for n, g in df.groupby("turnstile_id"):
            station_name = g["station"].iloc[0]
            station_dir = "./extracted/{}".format(station_name)
            if not os.path.isdir(station_dir):
                os.mkdir(station_dir)

            cols = [
                "turnstile_id",
                "station",
                "observed_at",
                "regular",
                "entries",
                "exits",
            ]

            with open(station_dir + "/" + n + ".csv", "a+") as f:
                g.to_csv(f, mode="a+", columns=cols, header=f.tell() == 0, index=False)

        successes.append(full_path)
    except:
        print("Warning: Error occured processing", full_path)
        raise

if len(successes) < len(paths):
    print("Warning: Success count doesn't match length of paths. Failures: ", failures)
