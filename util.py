import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from model_config import ModelConfig, ParameterConfig


def get_loss_data(
    path: str = "data_generation/data",
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Load the loss data from the specified path.

    Args:
        path: The path to the loss data.

    Returns:
        A dictionary containing the loss data.
    """
    if not os.path.exists(path):
        inp = input(f"No data found in {path}. Do you want to generate it? [y/n] ")
        if inp == "y":
            os.chdir("data_generation")
            os.system("python generate.py")
            os.chdir("..")
        else:
            sys.tracebacklimit = 0
            raise FileNotFoundError(f"Please run data_generation/generate.py first.")

    datasets = os.listdir(path)
    loss_datas = defaultdict(dict)

    for dataset in datasets:
        loss_based_data = os.listdir(os.path.join(path, dataset))
        for loss_data in loss_based_data:
            for split in ["train", "test"]:
                loss_path = os.path.join(path, dataset, loss_data, split + ".csv")
                if loss_datas[dataset].get(loss_data) is None:
                    loss_datas[dataset][loss_data] = {}
                loss_datas[dataset][loss_data][split] = pd.read_csv(loss_path)

    return loss_datas


def get_original_datasets(
    path: str = "data_generation/datasets",
    datasets: list = ["sst2", "sarcastic-headlines"],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load the original datasets from the specified path.

    Args:
        path: The path to the original datasets.
        datasets: The datasets to load.

    Returns:
        A dictionary containing the original datasets.
    """
    data = {}
    for dataset in datasets:
        data[dataset] = {
            "train": pd.read_csv(f"{path}/{dataset}/train.csv"),
            "test": pd.read_csv(f"{path}/{dataset}/test.csv"),
        }
    return data


datasets = ["sst2", "sarcastic-headlines"]


def get_datasets() -> List[str]:
    return datasets


models = {
    "gte-base": "thenlper/gte-base",
    "gte-small": "thenlper/gte-small",
    "e5-small": "intfloat/e5-small-v2",
    "minilm-6": "sentence-transformers/all-MiniLM-L6-v2",
}


def get_models() -> Dict[str, str]:
    return models


dataset_map = {"sst2": "sst2", "sarcastic-headlines": "sarcasm"}


def get_setfit_model(model_name, dataset, sample_size="50k", user="USERNAME"):
    # path to published and trained models.
    return f"{user}/{model_name}-{dataset}-setfit-{sample_size}-v2"


def get_setfit_models() -> List[str]:
    setfit_models = []
    for model in models.keys():
        for dataset in get_datasets():
            setfit_model = get_setfit_model(
                model_name=model, dataset=dataset_map[dataset]
            )
            setfit_models.append(setfit_model)
    return setfit_models


def get_train_test(
    original_datasets: Dict[str, Dict[str, pd.DataFrame]],
    dataset_name: str,
    train_to_test_ratio: int = 5,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the original dataset into train and test sets.

    Args:
        original_datasets: A dictionary containing the original datasets.
        dataset_name: The name of the dataset to split.
        train_to_test_ratio: The train data must be this many times larger than the test data.
    Returns:
        A tuple containing the train and test dataframes.
    """
    original_data = original_datasets[dataset_name]
    train_df = original_data["train"]
    test_data = original_data["test"]
    size = len(test_data) * train_to_test_ratio
    sample_size = len(train_df) if len(train_df) < size else size
    train_data = train_df.sample(sample_size, random_state=random_seed)

    return train_data, test_data


def make_wandb_config(
    _name: str,
    _config: ModelConfig,
    params: ParameterConfig,
    dataset_name: str,
    train_samples: int,
):
    return {
        "model": _config.model_name.split("/")[-1],
        "model_name": _name,
        "dataset": dataset_name,
        "simple_model_name": _name.split("(")[0].strip(),
        "loss": _config.loss.__name__,
        "loss_params": _config.loss_params,
        "train_samples": train_samples,
        "batch_size": params.BATCH_SIZE,
        "k": params.K,
        "learning_rate": params.LEARNING_RATE,
        "eval_every": params.EVAL_EVERY,
        "epochs": params.N_ITERS,
    }
