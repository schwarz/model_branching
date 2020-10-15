import csv
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


first = "./raw/turnstile_141018.txt"
last = "./raw/turnstile_200613.txt"
paths = [first, last]

turnstiles = {}
stations = {}

p = last
df = pd.read_csv(
    p,
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

df["station"] = df["station"].str.replace(" ", "_")  # spaces
df["station"] = df["station"].str.replace("B'WAY", "BROADWAY")  # bway
df["station"] = df["station"].str.replace("/", "and")  # street/street

df["turnstile_id"] = df["controlarea"] + "-" + df["remoteunit"] + "-" + df["scp"]

for n, g in df.groupby("turnstile_id"):
    station = g["station"].iloc[0]
    size = os.path.getsize("./extracted/{}/{}.csv".format(station, n))
    if size > 300000:
        print("Investigate ./extracted/{}/{}.csv".format(station, n))
        turnstiles[n] = True
        stations[station] = True


alltimers = []

for k, v in turnstiles.items():
    alltimers.append(k)

# print(alltimers[:10])

print(len(stations), "stations")
print(len(alltimers), "turnstiles from the start to the end of time")
