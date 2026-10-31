import ast
import re
from fractions import Fraction

import numpy as np

from common import parse_answer


NUMBER = r"[-+]?\d+(?:\.\d+)?"


def safe_value(text):
    text = text.replace(",", "")
    text = text.replace("−", "-").replace("–", "-")

    for old, new in (
        ("\\times", "*"),
        ("\\cdot", "*"),
        ("×", "*"),
        ("\\div", "/"),
        ("÷", "/"),
    ):
        text = text.replace(old, new)

    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\\,", "")
    text = text.replace("^", "**")

    for _ in range(4):
        text = re.sub(
            r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
            r"(\1)/(\2)",
            text,
        )

    text = re.sub(r"\\text\s*\{[^{}]*\}", "", text)
    text = text.replace("$", "").replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()

    if not re.fullmatch(r"[0-9+*/(). \-]+", text):
        return None

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)

        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value

        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and abs(right) <= 8:
                return left ** right

        raise ValueError

    try:
        value = evaluate(ast.parse(text, mode="eval").body)
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def equation_features(text, answer):
    residuals = []
    last_rhs = None
    last_residual = None

    for line in re.split(r"[\n;]", text):
        if "=" not in line:
            continue

        pieces = line.split("=")

        for left, right in zip(pieces, pieces[1:]):
            left_value = safe_value(left)
            right_value = safe_value(right)

            if left_value is None or right_value is None:
                continue

            residual = abs(left_value - right_value) / (
                1 + abs(right_value)
            )

            residuals.append(residual)
            last_rhs = right_value
            last_residual = residual

    if not residuals:
        return [0.0] * 12

    final_value = (
        float(Fraction(answer))
        if answer is not None
        else 0.0
    )

    residuals = np.asarray(residuals, dtype=float)

    return [
        np.log1p(len(residuals)),
        float(np.sum(residuals < 1e-6)),
        float(np.sum(residuals >= 1e-4)),
        float(residuals[-1] >= 1e-4),
        float(residuals.mean()),
        float(residuals.max()),
        float(np.sum(residuals > 0.01)),
        float(
            last_rhs is not None
            and abs(last_rhs - final_value) < 1e-6
        ),
        float(last_residual < 1e-6),
        float(np.min(residuals)),
        float(np.median(residuals)),
        float(last_rhs is not None),
    ]


def extract_features(bank):
    rows = []

    for question in bank:
        answers = [
            parse_answer(candidate["text"])
            for candidate in question["candidates"]
        ]

        question_numbers = [
            float(value.replace(",", ""))
            for value in re.findall(NUMBER, question["question"])
        ]

        question_set = set(question_numbers)
        question_max = max(
            [abs(value) for value in question_numbers],
            default=1.0,
        )

        question_rows = []

        for candidate, answer in zip(
            question["candidates"],
            answers,
        ):
            text = candidate["text"]

            values = [
                float(value.replace(",", ""))
                for value in re.findall(NUMBER, text)
            ]

            final_value = (
                float(Fraction(answer))
                if answer is not None
                else 0.0
            )

            feature = equation_features(text, answer)

            feature += [
                np.log1p(len(values)),
                np.log1p(len(set(values))),
                float(final_value in question_set),
                float(
                    any(
                        abs(value - final_value) < 1e-8
                        for value in values
                    )
                ),
                np.log1p(
                    abs(final_value) / (1 + question_max)
                ),
                float(final_value == int(final_value)),
                float("\\boxed" in text),
                float("####" in text),
            ]

            question_rows.append(feature)

        rows.append(question_rows)

    return np.asarray(rows, dtype=np.float64)