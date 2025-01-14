import json
import os
import re
import sys

import pandas as pd
import yaml

run_folder = "wandb/"
runs = [f for f in os.listdir(run_folder) if re.match(r"run.*", f)]

if len(runs) == 0:
    print("No runs or logs found")
    sys.exit(1)

runs.sort(key=lambda x: os.path.getmtime(run_folder + x), reverse=False)
metrics = ["polarity", "semantic", "combined"]
data = []

for run in runs:
    run_path = os.path.join(run_folder, run, "files")

    if not os.path.exists(os.path.join(run_path, "wandb-summary.json")):
        continue
    if not os.path.exists(os.path.join(run_path, "config.yaml")):
        continue

    with open(os.path.join(run_path, "wandb-summary.json"), "r") as f:
        summary = json.load(f)
        results = {k: v for k, v in summary.items() if k in metrics}
    if len(results) == 0:
        continue

    with open(os.path.join(run_path, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        model = config["model"]["value"]
        model_id = config["model_name"]["value"]
        dataset = config["dataset"]["value"]
        loss = config["loss"]["value"]
        samples = config["train_samples"]["value"]

    lambda_margin = re.search(r"lambda=([0-9.]+)", model_id)
    data.append(
        {
            "id": f"{dataset}-{samples}-{model_id}",
            "dataset": dataset,
            "model": model,
            "model_id": model_id,
            "lambda": float(lambda_margin.group(1)) if lambda_margin else None,
            "loss": loss,
            "samples": samples,
            **results,
        }
    )

df = pd.DataFrame(data)
df.to_csv("data/50-to-100k.csv", index=False)
