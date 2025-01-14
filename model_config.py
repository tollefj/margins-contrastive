from dataclasses import dataclass
from typing import Any, Dict

import yaml
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import TripletDistanceMetric
from torch import nn

CosDist = TripletDistanceMetric.COSINE


@dataclass
class ModelConfig:
    model: SentenceTransformer
    model_name: str
    loss: nn.Module
    loss_params: Dict[str, Any]


def generate_model_configs(sentence_transf_models, loss_configs):
    """
    Generate a list of model configurations to train.

    Args:
        sentence_transf_models: a dictionary of sentence transformer models
        loss_configs: a dictionary of loss configurations (see loss_config.py)
    """
    configs = {}
    for model_name in sentence_transf_models:
        for loss_name, loss_config in loss_configs.items():
            margin_key = "margin" if "Contrastive" in loss_name else "triplet_margin"
            for margin in loss_config.get(margin_key, [None]):
                model_id = f"{loss_name}({model_name})"
                loss_params = {}
                if margin is not None:
                    model_id = f"{model_id}(lambda={margin})"
                    loss_params = {"distance_metric": CosDist, margin_key: margin}
                configs[model_id] = ModelConfig(
                    model=None,
                    model_name=sentence_transf_models[model_name],
                    loss=loss_config["lossfn"],
                    loss_params=loss_params,
                )
    return configs


@dataclass
class ParameterConfig:
    RANDOM_SEED: int
    SAMPLING_SEED: int
    VERBOSE: bool
    N_ITERS: int
    EVAL_EVERY: int
    BATCH_SIZE: int
    DEVICE: str
    K: int
    LEARNING_RATE: float


def get_parameters(path: str = "params.yml") -> ParameterConfig:
    """
    Load the configuration from the specified YAML file.

    Args:
        path: The path to the YAML file.

    Returns:
        A `ParameterConfig` object containing the configuration.
    """
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return ParameterConfig(**config_dict)
