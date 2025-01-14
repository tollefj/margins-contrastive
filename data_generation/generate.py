import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from query_model import QueryModel
from sentence_transformers import SentenceTransformer

from util import generate_training_examples, split_similar_sents_by_polarity_batch


# these losses are tested to approximate ~250000 examples per dataset
# note that the multiple loss requires 0 dropout for the sarcasm dataset
# resulting in ~250000 examples, thus defined as the reference size.
def get_loss_config(dataset_id: str) -> Dict[str, Dict[str, float]]:
    multiple_loss: float = 0.73 if "sst" in dataset_id else 0
    triplet_loss: float = 0.83 if "sst" in dataset_id else 0.73
    contr_loss: float = 0.77 if "sst" in dataset_id else 0.32
    loss_config: Dict[str, Dict[str, float]] = {
        "Triplet": {"dropout": triplet_loss},
        "MultipleNegatives": {"dropout": multiple_loss},
        "Contrastive": {"dropout": contr_loss},
    }
    return loss_config


def create_examples(
    dataset_id: str, baseline_model: SentenceTransformer, K: int = 16, batch: int = 512
) -> None:
    splits: Dict[str, pd.DataFrame] = {
        "train": pd.read_csv(f"datasets/{dataset_id}/train.csv"),
        "test": pd.read_csv(f"datasets/{dataset_id}/test.csv"),
    }
    loss_config = get_loss_config(dataset_id)
    for split, split_df in splits.items():
        print(f"Split: {split}")
        _model = QueryModel(baseline_model, split_df)
        similar_data = split_similar_sents_by_polarity_batch(
            split_df, _model, k=K, batch_size=batch
        )

        for loss_type, loss_conf in loss_config.items():
            print(loss_type, loss_conf)
            dropout = loss_conf["dropout"]
            examples = generate_training_examples(
                similar_data, split_df, loss_type, dropout
            )
            os.makedirs(f"data/{dataset_id}/{loss_type}", exist_ok=True)
            examples.to_csv(f"data/{dataset_id}/{loss_type}/{split}.csv")


if __name__ == "__main__":
    np.random.seed(42)
    model = SentenceTransformer("intfloat/e5-small")

    if torch.cuda.is_available():
        model = model.to("cuda")
    for dataset in ["sst2", "sarcastic-headlines"]:
        create_examples(dataset_id=dataset, baseline_model=model)
