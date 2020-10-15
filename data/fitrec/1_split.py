import os
import json


# Read workout filenames from (train|test|valid).json and copy the files there
for set in ["train", "test", "valid"]:
    if not os.path.isdir(set):
        os.mkdir(set)
    with open("{}.json".format(set)) as json_file:
        filenames = json.load(json_file)
        for p in filenames:
            os.system("mkdir -p {}/`dirname {}`".format(set, p))
            os.system("cp extracted/{} {}/{}".format(p, set, p))
