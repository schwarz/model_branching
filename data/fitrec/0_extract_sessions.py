# import numpy as np
import pandas as pd
import os
import pickle
import numpy as np


def normalize_sport(sport):
    sport = sport.replace(" ", "_")
    sport = sport.replace("(", "")
    sport = sport.replace(")", "")
    return sport


lines_processed = 0
remove_keys = [
    "since_begin",
    "time_elapsed",
    "longitude",
    "latitude",
    "since_last",
    "altitude",
    "distance",
]

# https://stackoverflow.com/a/57277669
def reverse_zscore(array, mean, std, zmult):
    return array / zmult * std + mean


# https://github.com/nijianmo/fit-rec/blob/master/FitRec/data_interpreter_Keras_aux.py
class metaDataEndomondo(object):
    def __init__(
        self,
        numDataPoints,
        encodingLengths,
        oneHotEncoders,
        oneHotMap,
        isSequence,
        isNominal,
        isDerived,
        variableMeans,
        variableStds,
    ):
        self.numDataPoints = numDataPoints
        self.encodingLengths = encodingLengths
        self.oneHotEncoders = oneHotEncoders
        self.oneHotMap = oneHotMap
        self.isSequence = isSequence
        self.isNominal = isNominal
        self.isDerived = isDerived
        self.variableMeans = variableMeans
        self.variableStds = variableStds


means = {}
stds = {}
# Load the metadata to undo z-score normalization
with open("endomondoHR_proper_metaData.pkl", "rb") as f:
    metaData = pickle.load(f)
    attrs = ["heart_rate", "derived_speed"]
    for attr in attrs:
        means[attr] = metaData.variableMeans[attr]
        stds[attr] = metaData.variableStds[attr]

# Want: heart_rate, derived_speed, timestamp?, id, user_id,
with open("raw/processed_endomondoHR_proper_interpolate.json") as f:
    if not os.path.isdir("extracted"):
        os.mkdir("extracted")

    for l in f:
        if lines_processed % 100 == 0:
            print("Lines processed:", lines_processed)
        bout = eval(l)
        for k in remove_keys:
            bout.pop(k, None)

        gender = bout.pop("gender", "none")
        sport = normalize_sport(bout.pop("sport", "none"))
        user_id = bout.pop("userId", "none")
        id = bout.pop("id", "none")

        if not os.path.isdir("extracted/{}".format(sport)):
            os.mkdir("extracted/{}".format(sport))
        if not os.path.isdir("extracted/{}/{}".format(sport, user_id)):
            os.mkdir("extracted/{}/{}".format(sport, user_id))

        df = pd.DataFrame(
            {
                "timestamp": bout["timestamp"],
                "heart_rate": reverse_zscore(
                    np.array(bout["heart_rate"]),
                    means["heart_rate"],
                    stds["heart_rate"],
                    5,
                ),
                "derived_speed": reverse_zscore(
                    np.array(bout["derived_speed"]),
                    means["derived_speed"],
                    stds["derived_speed"],
                    5,
                ),
            }
        )
        df.to_feather(
            "extracted/{}/{}/{}_{}_{}.feather".format(
                sport, user_id, user_id, gender, id
            )
        )
        lines_processed = lines_processed + 1


print("Processed", lines_processed, "lines.")
