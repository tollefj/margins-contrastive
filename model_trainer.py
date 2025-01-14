from copy import deepcopy
from time import time
from typing import Dict, List, Tuple

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from model_config import ModelConfig, ParameterConfig
from model_evaluator import EvaluatorModel, evaluate_model

loss_map = {
    "TripletLoss": "Triplet",
    "MultipleNegativesRankingLoss": "MultipleNegatives",
    "ContrastiveLoss": "Contrastive",
    "OnlineContrastiveLoss": "Contrastive",
}


def get_samples(
    loss_data: pd.DataFrame, loss_type: str, split: str, n: int, random_seed: int
) -> pd.DataFrame:
    """
    Returns a random sample of n rows from the specified split of the loss_data DataFrame for the given loss_type.

    Args:
        loss_data (pd.DataFrame): DataFrame containing the loss data.
        loss_type (str): The type of loss function to use.
        split (str): The split of the data to sample from (e.g. "train", "dev", "test").
        n (int): The number of samples to return.
        random_seed (int): The random seed to use for sampling.

    Returns:
        pd.DataFrame: A DataFrame containing a random sample of n rows from the specified split of the loss_data DataFrame.
    """
    loss_type = loss_map[loss_type]
    data = loss_data[loss_type][split]
    sample_frac = n / len(data)
    return data.sample(frac=sample_frac, random_state=random_seed)


class ModelTrainer:
    def __init__(
        self,
        config: ModelConfig,
        data_source: pd.DataFrame,
        name: str,
        device: str = "cuda",
    ):
        """
        Initializes a new instance of the ModelTrainer class.

        Args:
            config (ModelConfig): The configuration for the model.
            data_source (pd.DataFrame): The DataFrame containing the training data.
            name (str): The name of the model.
            device (str): The device to use for training (e.g. "cuda", "cpu").
        """
        self.data_source = data_source
        self.name = name
        self.config = config
        self.model = SentenceTransformer(config.model_name)
        self.model = self.model.to(device)

    def init_data(
        self, n_samples: int, random_seed: int, split: str = "train"
    ) -> List[InputExample]:
        """
        Initializes the training data for the model.

        Args:
            n_samples (int): The number of samples to use for training.
            random_seed (int): The random seed to use for sampling.
            split (str): The split of the data to use for training (e.g. "train", "dev", "test").

        Returns:
            List[InputExample]: A list of InputExample objects containing the training data.
        """
        loss_type = loss_map[self.config.loss.__name__]
        _data = self.data_source[loss_type][split]
        sample_frac = n_samples / len(_data)
        data = _data.sample(frac=sample_frac, random_state=random_seed)

        examples = []
        loss_name = self.config.loss.__name__.lower()
        if "triplet" in loss_name:
            training_data = zip(data["Anchor"], data["Positive"], data["Negative"])
            examples = [InputExample(texts=[a, p, n]) for a, p, n in training_data]

        elif "multiple" in loss_name:
            training_data = zip(data["Anchor"], data["Positive"])
            examples = [InputExample(texts=[a, p]) for a, p in training_data]

        elif "contrastive" in loss_name:
            training_data = zip(data["Anchor"], data["Positive"], data["label"])
            examples = [
                InputExample(texts=[a, p], label=label) for a, p, label in training_data
            ]

        return examples

    def train(
        self,
        n_samples: int,
        random_seed: int = 42,
        trained_model: SentenceTransformer = None,
        epochs: int = 1,
        warmup: int = 10000,
        lr: float = 2e-5,
        decay: float = 0.01,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> SentenceTransformer:
        """
        Trains the model.

        Args:
            n_samples (int): The number of samples to use for training.
            random_seed (int): The random seed to use for sampling.
            trained_model (SentenceTransformer): A pre-trained model to use for training.
            epochs (int): The number of epochs to train for.
            warmup (int): The number of warmup steps to use.
            lr (float): The learning rate to use.
            decay (float): The weight decay to use.
            batch_size (int): The batch size to use.
            verbose (bool): Whether to show the progress bar during training.

        Returns:
            SentenceTransformer: The trained model.
        """
        examples = self.init_data(n_samples, random_seed)
        dataloader = DataLoader(examples, batch_size=batch_size)
        loss = self.config.loss(model=self.model, **self.config.loss_params)
        M = self.model if trained_model is None else trained_model

        M.fit(
            train_objectives=[(dataloader, loss)],
            epochs=epochs,
            warmup_steps=warmup,
            optimizer_params={"lr": lr},
            weight_decay=decay,
            show_progress_bar=verbose,
        )
        self.model = M
        return M

    def evaluate(
        self,
        k: int,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        verbose: bool = True,
    ) -> None:
        """
        Evaluates the model.

        Args:
            k (int): The number of nearest neighbors to use for evaluation.
            train_data (pd.DataFrame): The training data to use for evaluation.
            test_data (pd.DataFrame): The test data to use for evaluation.
            verbose (bool): Whether to show the progress bar during evaluation.
        """
        evaluator = EvaluatorModel(
            self.model, self.name, train_data, test_data, k=k, verbose=verbose
        )
        evaluator.evaluate()


def train_model(
    _name,
    _config: ModelConfig,
    params: ParameterConfig,
    loss_data,
    train_data,
    test_data,
    TRAIN_SAMPLES,
) -> Tuple[SentenceTransformer, Dict[str, float]]:
    print(f"Training {_name} with {TRAIN_SAMPLES} samples")
    model_trainer = ModelTrainer(_config, data_source=loss_data, name=_name)
    reference_model: SentenceTransformer = deepcopy(model_trainer.model)
    reference_embeddings_train = reference_model.encode(train_data.text.tolist())
    reference_embeddings_test = reference_model.encode(test_data.text.tolist())

    trained = None
    all_scores = []
    for i in range(params.N_ITERS):
        print(f"Epoch {i+1}")
        trained = model_trainer.train(
            trained_model=trained,
            epochs=1,
            lr=params.LEARNING_RATE,
            batch_size=params.BATCH_SIZE,
            verbose=params.VERBOSE,
            n_samples=TRAIN_SAMPLES,
            random_seed=i,  # ensure we sample new data each epoch
        )
        eval_start = time()
        _scores = evaluate_model(
            model=trained,
            name=_name,
            train_data=train_data,
            test_data=test_data,
            reference_train_emb=reference_embeddings_train,
            reference_test_emb=reference_embeddings_test,
            k=params.K,
            verbose=params.VERBOSE,
        )
        eval_end = time()
        eval_elapsed = eval_end - eval_start
        print(f"Eval {i+1} completed in {eval_elapsed} seconds")
        all_scores.append(_scores)
    return trained, all_scores
