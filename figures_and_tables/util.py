import json
import os
import re

import pandas as pd
import wandb
import yaml

sns_config = {
    "font.family": "serif",
}

matplot_config = {
    "font.family": "serif",
    "lines.linewidth": 3,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 18,
}

tex_config = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
}

model_translations = {
    "e5-small-v2": "e5-small",
    "gte-base": "gte-base",
    "gte-small": "gte-small",
    "all-MiniLM-L6-v2": "minilm-6",
}
model_order = {"e5-small": 0, "gte-base": 1, "gte-small": 2, "minilm-6": 3}
metrics = ["polarity", "semantic", "combined"]


def translate_models(models):
    return [model_translations[m] for m in models]


# Appendix A
def aggregated_losses_for_model(
    df, model, dataset=None, drop=["dataset", "lambda", "samples"]
):
    group = ["model", "loss", "epoch"]
    if dataset:
        _df = df[(df.dataset == dataset) & (df.model == model)]
    else:
        _df = df[df.model == model]
        drop.pop(0)
        group.append("dataset")

    return _df.drop(columns=drop).groupby(group).mean().reset_index()


# Appendix B
def filter_on_metric(df, metric, to_drop=["polartiy", "semantic"]):
    for m in to_drop:
        if m != metric:
            df = df.drop(columns=m)
    return df


# Appendix A+B
class Baseline:
    def __init__(self, path):
        self.baseline_df = pd.read_csv(path)
        self.baseline_df = self.baseline_df[self.baseline_df.k == 16]

    def get(self, model, dataset=None, metric=None):
        if not dataset:
            return self.baseline_df[self.baseline_df.model == model]
        if not metric:
            return self.baseline_df[
                (self.baseline_df.model == model)
                & (self.baseline_df.dataset == dataset)
            ].iloc[0]
        return self.baseline_df[
            (self.baseline_df.model == model) & (self.baseline_df.dataset == dataset)
        ].iloc[0][metric]

    def get_extremes(
        self, model, dataset=None, agg=min, metrics=["polarity", "semantic"]
    ):
        df = self.get(model)
        all_values = df[metrics].values.flatten()
        return agg(all_values)


# unified row creation from wandb-style logs
def make_data_row(dataset, model, model_name, loss, samples, results):
    lambda_margin = re.search(r"lambda=([0-9.]+)", model_name)
    return {
        "dataset": dataset,
        "model": model,
        "model_id": model_name,
        "lambda": float(lambda_margin.group(1)) if lambda_margin else None,
        "loss": loss,
        "samples": samples,
        **results,
    }


# wandb from api
def get_data_from_api(run_id, history=False):
    api = wandb.Api()
    runs = api.runs(run_id)

    data = []
    for run in runs:
        results = run.summary._json_dict
        results = {k: v for k, v in results.items() if k in metrics}
        if len(results) == 0:
            continue

        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        row = make_data_row(
            dataset=config["dataset"],
            model=config["model"],
            model_name=config["model_name"],
            loss=config["loss"],
            samples=config["train_samples"],
            results=results,
        )
        if history:
            _history = [h for h in run.scan_history()]
            _history = [{k: v for k, v in h.items() if k in metrics} for h in _history]
            row["history"] = _history
        data.append(row)
    return data


# wandb from local
def get_data_from_local_runs(run_folder):
    runs = [f for f in os.listdir(run_folder) if re.match(r"run.*", f)]
    runs.sort(key=lambda x: os.path.getmtime(run_folder + x), reverse=False)

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
            row = make_data_row(
                dataset=config["dataset"]["value"],
                model=config["model"]["value"],
                model_name=config["model_name"]["value"],
                loss=config["loss"]["value"],
                samples=config["train_samples"]["value"],
                results=results,
            )
            data.append(row)
    return data
