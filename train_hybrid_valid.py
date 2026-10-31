import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[name] = "4"

import time
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from hybrid_features import extract_features


def fit(features, labels, valid, alpha):
    count = valid.sum(axis=1, keepdims=True)

    centered_x = features - (
        (features * valid[..., None]).sum(
            axis=1,
            keepdims=True,
        )
        / count[..., None]
    )

    centered_y = labels - (
        (labels * valid).sum(
            axis=1,
            keepdims=True,
        )
        / count
    )

    keep = valid.reshape(-1)

    x = centered_x.reshape(
        -1,
        features.shape[-1],
    )[keep]

    y = centered_y.reshape(-1)[keep]

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0

    z = (x - mean) / scale

    gram = z.T @ z / len(z)
    gram.flat[:: gram.shape[0] + 1] += alpha

    weights = np.linalg.solve(
        gram,
        z.T @ y / len(y),
    )

    return mean, scale, weights, float(y.mean())


def main():
    start = time.perf_counter()

    bank, arrays = load_inputs("train")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")

    embeddings = arrays["embeddings"].astype(np.float64)
    rule_features = extract_features(bank)

    answers = [
        [
            parse_answer(candidate["text"])
            for candidate in question["candidates"]
        ]
        for question in bank
    ]

    valid = np.asarray([
        [answer is not None for answer in row]
        for row in answers
    ])

    labels = np.asarray([
        [
            answer is not None
            and answer == parse_answer(reference["answer"])
            for answer in row
        ]
        for row, reference in zip(answers, references)
    ], dtype=np.float64)

    embedding_mean, embedding_scale, embedding_weights, embedding_bias = fit(
        embeddings,
        labels,
        valid,
        alpha=1.0,
    )

    rule_mean, rule_scale, rule_weights, rule_bias = fit(
        rule_features,
        labels,
        valid,
        alpha=0.1,
    )

    output = Path("runs/hybrid_valid_model.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output,
        embedding_mean=embedding_mean.astype(np.float32),
        embedding_scale=embedding_scale.astype(np.float32),
        embedding_weights=embedding_weights.astype(np.float32),
        embedding_bias=np.array(
            embedding_bias,
            dtype=np.float32,
        ),
        rule_mean=rule_mean.astype(np.float32),
        rule_scale=rule_scale.astype(np.float32),
        rule_weights=rule_weights.astype(np.float32),
        rule_bias=np.array(
            rule_bias,
            dtype=np.float32,
        ),
        embedding_weight=np.array(
            0.75,
            dtype=np.float32,
        ),
        vote_weight=np.array(
            0.0,
            dtype=np.float32,
        ),
    )

    print("训练题数：", len(bank))
    print("规则特征数：", rule_features.shape[-1])
    print("模型已保存：", output)
    print("模型文件大小：", output.stat().st_size, "字节")
    print("训练用时：", round(time.perf_counter() - start, 2), "秒")


if __name__ == "__main__":
    main()