import os
import numpy as np

in_folder = "cifar10_custom/ens_c3_b2"
out_file = "cifar10_custom/ensemble_b2.npy"
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

if len(experiments) > 1:
    experiments = np.array(experiments)
    experiments = np.array([
        np.mean(experiments, axis=0),
        np.std(experiments, axis=0)
    ])
elif len(experiments) == 1:
    experiments = list(zip(experiments, [0.0]*len(experiments)))
    experiments = np.array(experiments).T
else:
    raise ValueError("List of Experiments is empty")


if os.path.exists(out_file):
    os.remove(out_file)
np.save(out_file, experiments)
pass