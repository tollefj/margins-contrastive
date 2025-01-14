from typing import List

import numpy as np
from sentence_transformers.util import cos_sim


def load_weights(k: int, use_discounting: bool = False) -> np.ndarray:
    """
    Load weights for calculating polarity and semantic similarity scores.

    Args:
        k: Number of predicted embeddings.
        use_discounting: Whether to use discounting weights.

    Returns:
        An array of weights.
    """
    w = np.ones(k) / k
    if use_discounting:
        w = np.array([k - i for i in range(k)])
        w = w / np.sum(w)
    return w


def polarity_score(y: int, ys: List[int], use_discounting: bool = False) -> float:
    """
    Calculate polarity score.

    Args:
        y: Ground truth label.
        ys: List of predicted labels.
        use_discounting: Whether to use discounting weights.

    Returns:
        Polarity score.
    """
    k = len(ys)
    w = load_weights(k, use_discounting)
    return sum(w[i] * (1 - y ^ ys[i]) for i in range(k))


def semantic_similarity_score(
    sent_embedding: np.ndarray,
    predicted_embeddings: List[np.ndarray],
    use_discounting: bool = False,
) -> float:
    """
    Calculate semantic similarity score.

    Args:
        sent_embedding: Sentence embedding.
        predicted_embeddings: List of predicted embeddings.
        use_discounting: Whether to use discounting weights.

    Returns:
        Semantic similarity score.
    """
    k = len(predicted_embeddings)
    w = load_weights(k, use_discounting)
    similarities = [cos_sim(sent_embedding, e) for e in predicted_embeddings]
    return sum(w[i] * similarities[i] for i in range(k))


def weighted_polarity_semantic(
    similarity: float, polarity: float, beta: float = 0.5
) -> float:
    """
    Calculate weighted polarity-semantic score.

    Args:
        similarity: Semantic similarity score.
        polarity: Polarity score.
        beta: Weight for polarity score.

    Returns:
        Weighted polarity-semantic score.
    """
    return beta * polarity + (1 - beta) * similarity
