import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict


def read_lines_to_series(path):
    with open(path, "r", encoding="utf-8") as text_file:
        return pd.Series(text_file.read().splitlines())


def load_tweeteval(tasks="all", splits="all", data_dir="data/tweeteval"):
    if tasks == "all":
        tasks = ["hate", "offensive", "sentiment"]
    if splits == "all":
        splits = ["train", "val", "test"]
    datasets = {
        task: DatasetDict(
            {
                split: Dataset.from_pandas(
                    pd.DataFrame(
                        {
                            "text": read_lines_to_series(
                                f"{data_dir}/{task}/{split}_text.txt"
                            ),
                            "labels": read_lines_to_series(
                                f"{data_dir}/{task}/{split}_labels.txt"
                            ).astype(int),
                        }
                    )
                )
                for split in splits
            }
        )
        for task in tasks
    }
    return datasets


from sklearn.metrics import accuracy_score, f1_score


def trainer_compute_metrics(eval_preds):
    pred_logits, labels_logits = eval_preds
    preds = pred_logits.argmax(axis=1)
    labels = labels_logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def indice2logits(indice, num_classes):
    indice = np.array(indice)
    logits = np.zeros([len(indice), num_classes], dtype=float)
    logits[np.arange(len(indice)), indice] = 1.0
    return {"label_logits": logits}


def get_metrics(predict, inputs, labels, splits, metric_dicts):
    preds = {split: predict(inputs[split]) for split in splits}

    return [
        (split, metric_name, metric(labels[split], preds[split]))
        for split in splits
        for metric_name, metric in metric_dicts.items()
    ]


def get_labels(dataset_dict, label_name):
    return {split: dataset_dict[split][label_name] for split in dataset_dict.keys()}


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")
