import argparse
import os
import glob
import pandas as pd
import sys
import pprint


parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str, default=None, help="Path of experiments")
args = parser.parse_args()

dir_path = args.path

metrics = ["eval/normalized_cost", "eval/normalized_reward"]
task_list = os.listdir(dir_path)
stat_dict = {k: {m: [] for m in metrics} for k in task_list}

for task in task_list:
    task_dir = os.path.join(dir_path, task)
    logfiles = glob.glob(f"{task_dir}/*/*/*/*/log.csv")
    for logfile in logfiles:
        temp = int(logfile.split('/')[-2].split('-')[-2])
        if not (temp == 0):
            continue
        # if not (temp == 0 or temp == 1 or temp == 42):
        #     continue
        # if not (temp == 0 or temp == 10 or temp == 20):
        #     continue
        df = pd.read_csv(logfile)
        for metric in metrics:
            if stat_dict[task][metric] == []:
                stat_dict[task][metric] = [df.iloc[-1][metric]]
            else:
                stat_dict[task][metric].append(df.iloc[-1][metric])

num_safe = 0
overall = {m: [] for m in metrics}
for k in stat_dict:
    for m in stat_dict[k]:
        overall[m].extend(stat_dict[k][m])
        stat_dict[k][m] = round(sum(stat_dict[k][m]) / len(stat_dict[k][m]), 2)
        if m == "eval/normalized_cost" and stat_dict[k][m] < 1.0:
            num_safe += 1

pprint.pprint(stat_dict)

for m in overall:
    overall[m] = round(sum(overall[m]) / len(overall[m]), 2)
pprint.pprint(overall)
print(f"Num of Safe Tasks: {num_safe} out of {len(task_list)}")