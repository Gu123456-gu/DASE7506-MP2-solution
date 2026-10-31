# DASE7506-MP2-solution
# DASE7506 MP2 — Candidate Answer Verification

This project selects one of four supplied candidate solutions for each GSM8K question.

The final method combines a group-centered embedding model with a model using arithmetic and text features.

- Protocol: `dase7506-mp2-open-hard-verifier-v5`
- Final validation accuracy: **94.49% (754/798)**
- Final test accuracy: **82.86937901498929% (387/467)**

## Repository files

The submission uses the following files at the repository root:

| File | Purpose |
| --- | --- |
| `hybrid_features.py` | Arithmetic, numeric and formatting feature extraction |
| `hybrid_method.py` | Prediction interface |
| `train_hybrid_valid.py` | Final model training code |
| `hybrid_valid_model.npz` | Trained checkpoint, normalization statistics and fusion settings |
| `requirements.txt` | Python dependencies |
| `test-hybrid-valid.json` | Saved test predictions |
| `test-hybrid-valid-score.json` | Official test scoring output |
| `validation-hybrid-valid-score.json` | Official validation scoring output |

The report and remaining resource/version records must be added before the submission is complete.

## Method

Training uses the supplied training inputs and references.

Candidates whose final answers cannot be parsed by the unchanged official `common.parse_answer` function are excluded from fitting.

Candidate features and correctness labels are centered within each question over its parseable candidates. Two ridge models are trained:

1. An embedding model using the supplied 1,024-dimensional representations, with regularization coefficient `1.0`.
2. A model using 20 arithmetic, numeric and formatting features, with regularization coefficient `0.1`.

Feature normalization statistics are fitted on training data and saved in the checkpoint.

Arithmetic features include residuals of parseable equations and agreement between the last parsed equation and the candidate's final answer. These are partial consistency checks rather than a complete mathematical proof verifier.

The final score is:

```text
candidate_score = 0.75 * embedding_score + 0.25 * arithmetic_feature_score
```

The answer-support voting coefficient is `0.0`.

During inference, candidates with unparseable final answers are masked. The highest-scoring remaining candidate is selected. Ties use candidate order.

Inference uses only the supplied inputs, embeddings and checkpoint. It does not read reference answers or saved prediction files. No external datasets, additional pretrained weights or online API calls are required.

## Installation and official data

Use Python 3.12.

Download and extract the complete official repository:

https://github.com/xudongwu-0/DASE7506-MP2

Keep the official data, parser and evaluation scripts unchanged.

Copy these submission files into the official repository's `code/` directory:

```text
hybrid_features.py
hybrid_method.py
train_hybrid_valid.py
hybrid_valid_model.npz
requirements.txt
```

Open a PowerShell terminal in that `code/` directory.

Create a Python environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_data.py
```

Set the CPU thread limits before running prediction:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
```

On Linux, use `python3.12 -m venv .venv`, replace the Python executable with `.venv/bin/python`, and set the environment variables using `export`.

## Reproduce the submitted result without retraining

These commands load the submitted checkpoint copied directly into `code/`:

```powershell
.\.venv\Scripts\python.exe predict.py --method hybrid_method.py --checkpoint hybrid_valid_model.npz --split test --output runs/reproduced-test.json
.\.venv\Scripts\python.exe score_predictions.py --split test --predictions runs/reproduced-test.json --output runs/reproduced-test-score.json
```

Expected scoring fields:

```json
{
  "protocol": "dase7506-mp2-open-hard-verifier-v5",
  "split": "test",
  "correct": 387,
  "n": 467,
  "accuracy": 82.86937901498929
}
```

Timing fields depend on the machine. The reported leaderboard score is the official scorer's `accuracy` field.

## Reproduce training

Training is optional when evaluating the submitted checkpoint.

```powershell
.\.venv\Scripts\python.exe train_hybrid_valid.py
```

The training script writes:

```text
runs/hybrid_valid_model.npz
```

This is a different location from the submitted checkpoint copied directly into `code/`.

Evaluate the newly trained checkpoint on validation:

```powershell
.\.venv\Scripts\python.exe predict.py --method hybrid_method.py --checkpoint runs/hybrid_valid_model.npz --split validation --output runs/retrained-validation.json
.\.venv\Scripts\python.exe score_predictions.py --split validation --predictions runs/retrained-validation.json --output runs/retrained-validation-score.json
```

Expected validation result: **754/798 correct, approximately 94.49%**.

Final fitting uses a deterministic closed-form solve, with no random initialization or sampling.

## Experiments

| Method | Validation accuracy (%) | Public test accuracy (%) |
| --- | ---: | ---: |
| Always select candidate 0 | 78.20 | 49.68 |
| Majority vote over parseable answers | 86.22 | 69.16 |
| Initial embedding/vote fusion | 89.60 | 77.09 |
| Hard-question centered embedding model | 93.36 | 80.30 |
| Hard-question embedding/arithmetic hybrid | 94.11 | 81.37 |
| Final all-valid-candidate hybrid | 94.49 | 82.87 |

Validation contains 443 groups with only one distinct valid parsed answer and 355 groups with multiple distinct valid answers. All 467 test groups have multiple distinct valid answers. Overall validation and test accuracy therefore reflect different difficulty mixtures.

## Evaluation history and limitations

Several intermediate methods were scored on the public test bank during development. Test-reference scores were also examined for exploratory intermediate parameter comparisons during AI-assisted development.

The reported test result is therefore not an untouched-holdout estimate. This development history does not meet a strict protocol that reserves test references exclusively for final scoring and reproduction, and must also be disclosed in the report.

The final fitting script reads training references only. The inference implementation reads no references and contains no test-answer lookup.

The arithmetic features cannot verify every reasoning step or determine whether a candidate has correctly interpreted every condition in the question.

## Resource and version record

The following records remain to be completed:

| Item | Record |
| --- | --- |
| Python version used for the final run | TODO |
| NumPy version used for the final run | TODO |
| Local CPU and RAM | TODO |
| Training and search costs | TODO |
| Complete prediction plus independent scoring time, including process startup, using four threads | TODO |
| Submitted NPZ file size in bytes | TODO |
| Total uncompressed inference asset size | TODO |
| Checkpoint SHA256 | TODO |
| Submitted code commit | TODO |
| Report, maximum 10 pages | TODO |

The total uncompressed inference assets must be at most 16 MiB.

The earlier recorded runtime of `0.403` seconds belongs to a different model and is not a measurement of this final checkpoint. The official scorer's `scoring_seconds` field does not include the prediction process.

Runtime eligibility is determined on the course reference CPU.

## AI assistance and attribution

OpenAI Codex/ChatGPT provided substantial assistance with task explanation, algorithm and feature design, Python training and inference code, debugging, experiment interpretation and documentation.

The author ran the submitted scripts and official evaluator on a local Windows computer.

The dataset, fixed representations, prediction interface and evaluator are provided by the official DASE7506 MP2 project:

https://github.com/xudongwu-0/DASE7506-MP2

Questions and references derive from GSM8K:

https://github.com/openai/grade-school-math

Candidate solutions and fixed representations were prepared using Qwen3-0.6B as described in the official project.

Retain the supplied license and attribution files when redistributing official data.
