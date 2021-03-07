import csv

import pandas as pd


def read_annotated_file(path, index="index", score=None):
    indices = []
    originals = []
    translations = []
    scores = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])
            scores.append(float(row[score]))

    return pd.DataFrame(
        {'index': indices,
         'original': originals,
         'translation': translations,
         score: scores
         })


def read_test_file(path, index="index"):
    indices = []
    originals = []
    translations = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])

    return pd.DataFrame(
        {'index': indices,
         'original': originals,
         'translation': translations,
         })
