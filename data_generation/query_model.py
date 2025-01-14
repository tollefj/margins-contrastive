import faiss


class QueryModel:
    def __init__(self, model, df, batch_size=256):
        self.model = model

        sentences = df.text.tolist()
        self.embeddings = model.encode(
            sentences,
            show_progress_bar=True,
            convert_to_tensor=False,
            batch_size=batch_size,
        )
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        self.sents = df.text.tolist()
        self.labels = df.label.tolist()

    def query(self, sents, k=16, min_threshold=0.5):
        vec = self.model.encode(sents)
        dists, indices = self.index.search(vec, k=k + 1)
        if len(sents) <= 1:
            indices = indices[0]
        indices = indices[:, 1:]
        dists = dists[:, 1:]

        final_dists = []
        final_indices = []
        for _dist, _index in zip(dists[0], indices[0]):
            if _dist > min_threshold:
                final_dists.append(_dist)
                final_indices.append(_index)
        return indices, dists
