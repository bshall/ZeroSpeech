from ABXpy.misc.any2h5features import convert

import ABXpy.task
import ABXpy.distances.distances as distances
import ABXpy.distances.metrics.cosine as cosine
import ABXpy.distances.metrics.dtw as dtw
import ABXpy.score as score
import ABXpy.misc.items as items
import ABXpy.analyze as analyze

import ast
import numpy as np
import pandas

import argparse
from pathlib import Path
from tqdm import tqdm
import json


def dtw_cosine_distance(x, y, normalized):
    return dtw.dtw(x, y, cosine.cosine_distance, normalized)


def average_abx(filename, task_type):
    df = pandas.read_csv(filename, sep='\t')
    if task_type == "across":
        # aggregate on context
        groups = df.groupby(["speaker_1", "speaker_2", "phone_1", "phone_2"],
                            as_index=False)
        df = groups["score"].mean()
    elif task_type == "within":
        arr = np.array(map(ast.literal_eval, df["by"]))
        df["speaker"] = [e for e, f, g in arr]
        df["context"] = [f for e, f, g in arr]

        # aggregate on context
        groups = df.groupby(["speaker", "phone_1", "phone_2"], as_index=False)
        df = groups["score"].mean()
    else:
        raise ValueError("Unknown task type: {0}".format(task_type))

    # aggregate on talker
    groups = df.groupby(["phone_1", "phone_2"], as_index=False)
    df = groups['score'].mean()
    average = df.mean()[0]
    average = (1.0 - average) * 100
    return average


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path to resume")
    parser.add_argument("--gen-dir", type=str, default="./generated")
    parser.add_argument("--speaker-list", type=str)
    parser.add_argument("--mel-path", type=str)
    parser.add_argument("--speaker", type=str)
    args = parser.parse_args()

    with Path("config.json").open() as file:
        params = json.load(file)

    task_path = Path("../ABX/info_test/by-context-across-speakers.abx")
    # task_path = Path("../../Datasets/zerospeech2017/data/test/english/1s/1s_across.abx")
    # task_path = Path("english_across.abx")
    feature_path = Path("./features.features")
    distance_path = Path("./data.distance")
    score_path = Path("./data.score")
    analyze_path = Path("./data.csv")

    convert(str("encoded/ZeroSpeech2019/english/test/"))

    distances.compute_distances(
        str(feature_path), "features", str(task_path),
        str(distance_path), dtw_cosine_distance,
        normalized=True, n_cpu=4)

    score.score(str(task_path), str(distance_path), str(score_path))

    analyze.analyze(str(task_path), str(score_path), str(analyze_path))

    print(average_abx(str(analyze_path), "across"))
