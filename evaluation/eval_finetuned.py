from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, "SentEval")
import senteval
import os
import json
import pandas as pd

PATH_TO_SENTEVAL = "SentEval"
PATH_TO_DATA = "SentEval/data"

# params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, "kfold": 10}
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 5}
# much faster config, with similar scores
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, "kfold": 3}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 128, 'tenacity': 5, 'epoch_size': 3}
# SentEval required fns: prepare and batcher
def prepare(params, samples):
    pass

def eval_model(model_name):
    print(f"Evaluating {model_name}")
    model = SentenceTransformer(model_name)
    model = model.to("cuda")

    # https://github.com/UKPLab/sentence-transformers/issues/50#issuecomment-566452390
    def batcher(params, batch):
        sentences = []
        for sample in batch:
            untoken = ' '.join(sample).lower()
            if untoken == '':
                untoken = '-'
            sentences.append(untoken)
        return model.encode(sentences)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SICKEntailment', 'SST2', 'TREC', 'MRPC']
    results = se.eval(transfer_tasks)
    print(results)
    return results

finetuned_path = "../finetuned_model_store/"

eval_results = {}

for model_path in sorted(os.listdir(finetuned_path)):
    full_path = os.path.join(finetuned_path, model_path)
    print(full_path)
    eval_results[model_path] = eval_model(full_path)
    

with open('eval_results.json', 'w') as fp:
    json.dump(eval_results, fp)
    
    
acc_results = {}

for model_name, metrics in eval_results.items():
    model_acc = {}
    for metric_name, metric in metrics.items():
        model_acc[metric_name] = metric["acc"]
    acc_results[model_name] = model_acc

df = pd.DataFrame(acc_results).T
df["avg"] = df.mean(axis=1).round(2)
df = df.reset_index().rename(columns={"index": "model"})
df.to_csv("finetuned_senteval_results.csv", index=False)