from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers.readers import InputExample as Ex
from tqdm import tqdm

MIN_THRESHOLD = 0.5


def split_similar_sents_by_polarity_batch(df, model, k=64, batch_size=512):
    split_data = []

    num_batches = (df.shape[0] + batch_size - 1) // batch_size

    for batch_start in tqdm(
        range(0, df.shape[0], batch_size),
        total=num_batches,
        desc="Split by class",
        leave=False,
    ):
        batch_end = min(batch_start + batch_size, df.shape[0])
        batch_texts = df.iloc[batch_start:batch_end, 0].values
        batch_labels = df.iloc[batch_start:batch_end, 1].values

        similar_idx, _ = model.query(batch_texts, k=k, min_threshold=MIN_THRESHOLD)

        for i in range(len(batch_texts)):
            text = batch_texts[i]
            label = batch_labels[i]
            similar_labels = df.iloc[similar_idx[i]]["label"].values

            similar_object = {
                "text": text,
                "label": label,
                "similar_sentences": defaultdict(list),
            }

            # To support multilabel classification, add each value found in the labels
            for value in np.unique(similar_labels):
                similar_object["similar_sentences"][value] = similar_idx[i][
                    similar_labels == value
                ].tolist()

            split_data.append(similar_object)

    return split_data


def generate_tuples(split_data, df, dropout=0.2, pos_label=1, neg_label=0):
    tuples = []

    for obj in split_data:
        sent = obj["text"]
        label = obj["label"]
        similar_idxs = obj["similar_sentences"]

        _class = similar_idxs[pos_label] if label else similar_idxs[neg_label]
        for s_idx in _class:
            if np.random.rand() < dropout:
                continue
            s_text = df.iloc[s_idx]["text"]
            tuples.append((sent, s_text))
    return tuples


def generate_triples(split_data, df, dropout=0.9, pos_label=1, neg_label=0):
    triples = []

    for obj in split_data:
        sent, label = obj["text"], obj["label"]
        similar_idxs = obj["similar_sentences"]

        # iterate the possible classes and create triples of:
        # ['Anchor 1', 'Positive 1', 'Negative 1']
        iter1 = similar_idxs[pos_label] if label == 1 else similar_idxs[neg_label]
        iter2 = similar_idxs[neg_label] if label == 1 else similar_idxs[pos_label]

        for s1_idx in iter1:
            for s2_idx in iter2:
                if np.random.rand() < dropout:
                    continue
                s1_text = df.iloc[s1_idx]["text"]
                s2_text = df.iloc[s2_idx]["text"]
                triples.append((sent, s1_text, s2_text))

    return triples


def generate_labelled_tuples(split_data, df, dropout=0.2):
    labelled_tuples = []

    for obj in split_data:
        sent = obj["text"]
        label = obj["label"]

        maximize = obj["similar_sentences"][0] if label else obj["similar_sentences"][1]
        minimize = obj["similar_sentences"][1] if label else obj["similar_sentences"][0]

        for target, data in enumerate([maximize, minimize]):
            for s_idx in data:
                if np.random.rand() < dropout:
                    continue
                s_text = df.iloc[s_idx]["text"]
                labelled_tuples.append(((sent, s_text), target))

    return labelled_tuples


A = "Anchor"
P = "Positive"
N = "Negative"


def generate_training_examples(split_data, df, model_name, dropout) -> Dataset:
    if "Contrastive" in model_name:
        examples = generate_labelled_tuples(split_data, df, dropout=dropout)
        input_ex = [Ex(texts=texts, label=label) for texts, label in examples]
        contrastive_texts = [ex.texts for ex in input_ex]
        contrastive_labels = [ex.label for ex in input_ex]
        contrastive_df = pd.DataFrame(contrastive_texts, columns=[A, P])
        contrastive_df["label"] = contrastive_labels

        return contrastive_df

    elif "Triplet" in model_name:
        examples = generate_triples(split_data, df, dropout=dropout)
        input_ex = [Ex(texts=ex) for ex in examples]
        triplets = [exaple.texts for exaple in input_ex]
        triplet_df = pd.DataFrame(triplets, columns=[A, P, N])

        return triplet_df

    elif "MultipleNegatives" in model_name:
        examples = generate_tuples(split_data, df, dropout=dropout)
        input_ex = [Ex(texts=ex) for ex in examples]
        multiple_neg = [ex.texts for ex in input_ex]
        multiple_df = pd.DataFrame(multiple_neg, columns=[A, P])

        return multiple_df
