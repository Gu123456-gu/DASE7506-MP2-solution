import numpy as np

from common import parse_answer
from hybrid_features import extract_features


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("请提供 hybrid_model.npz")

    with np.load(checkpoint, allow_pickle=False) as model:
        embedding_mean = model["embedding_mean"]
        embedding_scale = model["embedding_scale"]
        embedding_weights = model["embedding_weights"]
        embedding_bias = float(model["embedding_bias"])

        rule_mean = model["rule_mean"]
        rule_scale = model["rule_scale"]
        rule_weights = model["rule_weights"]
        rule_bias = float(model["rule_bias"])

        embedding_weight = float(model["embedding_weight"])
        vote_weight = float(model["vote_weight"])

    # embedding 模型分数
    embedding_scores = (
        (arrays["embeddings"] - embedding_mean)
        / embedding_scale
    ) @ embedding_weights + embedding_bias

    # 算式一致性特征分数
    rule_features = extract_features(bank)

    rule_scores = (
        (rule_features - rule_mean)
        / rule_scale
    ) @ rule_weights + rule_bias

    # 融合两个模型
    scores = (
        embedding_weight * embedding_scores
        + (1.0 - embedding_weight) * rule_scores
    )

    # 处理答案和答案一致性
    for question_index, question in enumerate(bank):
        answers = [
            parse_answer(candidate["text"])
            for candidate in question["candidates"]
        ]

        for candidate_index, answer in enumerate(answers):
            if answer is None:
                # 无法解析最终答案的候选不参与选择
                scores[question_index, candidate_index] = -1e30
            else:
                support = answers.count(answer) / 4.0
                scores[question_index, candidate_index] += (
                    vote_weight * support
                )

    return scores.argmax(axis=1).astype(int).tolist()