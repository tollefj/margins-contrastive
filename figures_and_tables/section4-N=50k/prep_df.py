import sys

import pandas as pd

sys.path.append("../")
from util import get_data_from_api, model_translations


# run_id = "INSERT_USER/INSERT_RUN_ID"
# data = get_data_from_api(run_id, history=True)
# df = pd.DataFrame(data)
# df.to_csv("50k-history.csv")
def prep_df(path="50k-history.csv"):
    df = pd.read_csv(path, index_col=0)
    df.model = df.model.apply(lambda x: model_translations[x])
    df.loss = df.model_id.map(lambda x: x.split("(")[0])
    df = df.drop(
        columns=["model_id", "combined", "polarity", "semantic"]
    )  # these are the mean of metrics, skip 'em
    df["history"] = df["history"].apply(eval)
    df = df.explode("history").reset_index(drop=True)
    df["epoch"] = 1 + df.index % 10  # there's 10 epochs.
    df = df.join(pd.DataFrame(df["history"].tolist()))
    df = df.drop(columns=["history"])

    return df
