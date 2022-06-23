import os
import numpy as np

in_folder = "pan17/random"
out_file = "pan17/random.npy"
repeats = 2

experiments = []
for in_file in os.listdir(in_folder):
    with open(os.path.join(in_folder, in_file), "r") as f:
        contents = f.readlines()

    scores = list()
    for i in range(len(contents)):
        for _ in range(repeats):
            scores.append( float(contents[i].strip()) )
    experiments.append(scores)

experiments = np.array(experiments)
experiments = np.array([
    np.mean(experiments, axis=0),
    np.std(experiments, axis=0)
])

if os.path.exists(out_file):
    os.remove(out_file)
np.save(out_file, experiments)
pass