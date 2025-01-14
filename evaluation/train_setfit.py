import sys
sys.path.append("..")
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset

from util import get_models

data_path = "data_generation/datasets/"
sst_path = os.path.join(data_path, "sst2")
sarcasm_path = os.path.join(data_path, "sarcastic-headlines")

df_sst2_train = pd.read_csv(os.path.join(sst_path, "train.csv"))
df_sst2_test = pd.read_csv(os.path.join(sst_path, "test.csv"))

df_sarc_train = pd.read_csv(os.path.join(sarcasm_path, "train.csv"))
df_sarc_test = pd.read_csv(os.path.join(sarcasm_path, "test.csv"))

custom_datasets = {
    "sarcasm": DatasetDict(
        {
            "train": Dataset.from_pandas(df_sarc_train),
            "test": Dataset.from_pandas(df_sarc_test),
        }
    ),
    "sst2": DatasetDict(
        {
            "train": Dataset.from_pandas(df_sst2_train),
            "test": Dataset.from_pandas(df_sst2_test),
        }
    ),
}

models = get_models()
num_classes = 2
SAMPLES = 625  # corresponds to 50.000 generated examples

for dataset_name, dataset in custom_datasets.items():
    for model_name, model in models.items():
        print(dataset_name, model_name)
        train_dataset = sample_dataset(
            dataset["train"], label_column="label", num_samples=SAMPLES
        )
        eval_dataset = dataset["test"]

        model = SetFitModel.from_pretrained(
            model,
            use_differentiable_head=True,
            head_params={"out_features": 2},
        )

        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=32,
            num_epochs=1,
        )
        trainer.freeze()
        trainer.train()
        trainer.unfreeze(keep_body_frozen=False)
        trainer.train(
            num_epochs=5,
            batch_size=32,
            body_learning_rate=3e-5,
            learning_rate=1e-2,
            l2_weight=0.01,
            show_progress_bar=False,
        )
        model_id_name = f"USERNAME/{model_name}-{dataset_name}-setfit-50k-v2"
        trainer.push_to_hub(model_id_name, private=True)
