from typing import Any, Dict, Tuple

import faiss
import numpy as np
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from metrics import (
    polarity_score,
    semantic_similarity_score,
    weighted_polarity_semantic,
)


class EvaluatorModel:
    def __init__(
        self,
        model: SentenceTransformer,
        name: str,
        train_df: DataFrame,
        test_df: DataFrame,
        k: int = 16,
        verbose: bool = False,
    ):
        """
        Initializes an EvaluatorModel object.

        Args:
            model (SentenceTransformer): A SentenceTransformer model object.
            name (str): The name of the model.
            train_df (pandas.DataFrame): A pandas DataFrame containing the training data.
            test_df (pandas.DataFrame): A pandas DataFrame containing the test data.
            k (int, optional): The number of nearest neighbors to retrieve. Defaults to 16.
            verbose (bool, optional): Whether to show progress bars during encoding. Defaults to False.
        """
        self.model_name = name
        self.train_df = train_df
        self.test_df = test_df
        self.k = k

        # create index for searches
        self.embeddings = model.encode(
            self.train_df.text.tolist(),
            show_progress_bar=verbose,
            convert_to_tensor=False,
            batch_size=256,
        )
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        self.test_embeddings = model.encode(
            self.test_df.text.tolist(),
            show_progress_bar=verbose,
            convert_to_tensor=False,
            batch_size=256,
        )

        self.test_sents = self.test_df.text.tolist()
        self.test_labels = self.test_df.label.tolist()

    def evaluate(
        self,
        reference_train: np.ndarray,
        reference_test: np.ndarray,
        discount: bool = False,
        return_scores: bool = True,
        round_to: int = 3,
    ) -> Dict[str, float]:
        """
        Evaluate the performance of the model on the test set, using the reference training set for comparison.

        Args:
            reference_train (np.ndarray): The reference training set embeddings.
            reference_test (np.ndarray): The reference test set embeddings.
            discount (bool, optional): Whether to use discounting in the polarity score calculation. Defaults to False.
            return_scores (bool, optional): Whether to return the scores as a dictionary or print them. Defaults to True.

        Returns:
            Dict[str, float]: A dictionary with the mean polarity and semantic similarity rounded to 3 decimal places.
        """
        _, top_k_idxs = self.index.search(self.test_embeddings, self.k)

        polarity_scores = []
        semantic_similarity_scores = []

        for i in range(len(self.test_sents)):
            y = self.test_labels[i]
            ys = [self.train_df.iloc[idx].label for idx in top_k_idxs[i]]
            _PS = polarity_score(y, ys, use_discounting=discount)

            sent_embedding = reference_test[i]
            top_k_embeddings = reference_train[top_k_idxs[i]]
            _SS = semantic_similarity_score(
                sent_embedding, top_k_embeddings, use_discounting=discount
            )

            polarity_scores.append(_PS)
            semantic_similarity_scores.append(_SS)

        polarity = [p * 100 for p in polarity_scores]
        semantic = [s.item() * 100 for s in semantic_similarity_scores]

        if return_scores:
            return {
                "polarity": np.round(polarity, round_to),
                "semantic": np.round(semantic, round_to),
                "polarity_mean": np.mean(polarity).round(round_to),
                "polarity_std": np.std(polarity).round(round_to),
                "semantic_mean": np.mean(semantic).round(round_to),
                "semantic_std": np.std(semantic).round(round_to),
            }
        else:
            self.prettyprint_metrics(polarity, semantic)

    def prettyprint_metrics(self, polarity, semantic):
        print(
            f""" Model: {self.model_name}
        POLARITY
        mean: {np.round(np.mean(polarity), 3)} (max: {np.max(polarity)}, min: {np.min(polarity)})
        std: {np.round(np.std(polarity), 3)}
    
        SEMANTIC SIMILARITY
        mean: {np.round(np.mean(semantic), 3)} (max: {np.max(semantic)}, min: {np.min(semantic)})
        std: {np.round(np.std(semantic), 3)}
        """
        )


def evaluate_model(
    model: Any,
    name: str,
    train_data: DataFrame,
    test_data: DataFrame,
    reference_train_emb: Dict[str, Any],
    reference_test_emb: Dict[str, Any],
    k: int,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate the given model on the train and test data using the provided reference embeddings.

    Args:
        model (Any): The model to evaluate.
        name (str): The name of the model.
        train_data (DataFrame): The training data.
        test_data (DataFrame): The test data.
        reference_train_emb (Dict[str, Any]): The reference embeddings for the training data.
        reference_test_emb (Dict[str, Any]): The reference embeddings for the test data.
        k (int): The number of nearest neighbors to consider.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics.
    """
    evaluator = EvaluatorModel(
        model=model,
        name=name,
        train_df=train_data,
        test_df=test_data,
        k=k,
        verbose=verbose,
    )
    return evaluator.evaluate(reference_train_emb, reference_test_emb)
