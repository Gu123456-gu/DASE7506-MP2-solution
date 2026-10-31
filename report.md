# Selecting Reliable LLM Reasoning
## Centered Linear Models with Arithmetic Features

**DASE7506 MP2 | GitHub: Gu123456-gu | 31 October 2026**

The final method selects a correct candidate for **754/798 validation questions (94.49%)** and **387/467 public test questions (82.86937901498929%)**. It combines two small linear models and uses no additional pretrained model or online API.

### 1. Task and data

Each question has four supplied candidate solutions. The program returns one index from 0 to 3. Selecting any candidate whose parsed final answer matches the reference earns one point. The official score measures final-answer accuracy, rather than whether every reasoning step is valid.

| Split | Questions | Candidates per question |
| --- | ---: | ---: |
| Training | 4,755 | 4 |
| Validation | 798 | 4 |
| Test | 467 | 4 |

The inputs are the question, candidate texts and fixed 1,024-dimensional candidate embeddings. Training references provide correctness labels. The supplied data, parser and scoring code are kept unchanged.

### 2. Method

**Research question:** Can relative comparison within a question and simple arithmetic checks improve selection over answer voting?

The final method has four steps:

1. **Prepare labels.** Use the official parser to extract candidate answers. A training candidate receives label 1 if its answer matches the training reference, otherwise 0. Unparseable candidates are excluded from fitting.
2. **Center within each question.** Subtract the mean feature vector and mean label of that question's parseable candidates. This encourages learning differences between competing candidates. All training questions are included.
3. **Fit two ridge models.** One uses the 1,024 embedding features; the other uses 20 arithmetic, numeric and formatting features. Column normalization is fitted on training data. Ridge regularization is 1.0 for embeddings and 0.1 for arithmetic features, using average squared loss.
4. **Combine and choose.** Compute the score below, mask unparseable answers and return the highest-scoring candidate. Ties select the earliest candidate.

```text
score = 0.75 * embedding_score + 0.25 * arithmetic_score
```

The 20 features summarize equation residuals, agreement between the last parsed equation and the final answer, numeric patterns and answer markers. These are partial checks: consistent arithmetic does not guarantee a correct interpretation of the question. The final answer-voting coefficient is zero.

Final fitting is a deterministic linear solve without random sampling. The checkpoint stores float32 weights, normalization statistics and fusion settings. Inference reads no reference answers or saved prediction files, and does not look up answers by question ID.

<!-- pagebreak -->

## 3. Experiments

### 3.1 Development results

All results use the official evaluator. The table shows the development trajectory; later rows change several settings together and are not isolated single-factor ablations.

| Method | Validation accuracy (%) | Public test accuracy (%) |
| --- | ---: | ---: |
| Always select candidate 0 | 78.20 | 49.68 |
| Majority vote over parseable answers | 86.22 | 69.16 |
| Initial embedding/vote fusion | 89.60 | 77.09 |
| Hard-question centered embedding | 93.36 | 80.30 |
| Hard-question embedding/arithmetic hybrid | 94.11 | 81.37 |
| Final all-valid-candidate hybrid | 94.49 | 82.87 |

The final model improves public test accuracy by **13.70 percentage points** over majority voting. This suggests that answer frequency alone leaves useful evidence unused, but the test-history limitation in Section 4 applies.

Moving from the earlier hard-question hybrid to the final hybrid changed the training subset, arithmetic regularization, fusion weight and voting weight. The improvement cannot be attributed solely to adding training questions.

### 3.2 Controlled answer-support ablation

On an earlier fixed embedding model, only the answer-support coefficient was changed. This experiment uses validation results:

| Support coefficient | Correct / 798 | Accuracy (%) |
| ---: | ---: | ---: |
| 0.00 | 707 | 88.60 |
| 0.25 | 715 | 89.60 |
| 0.50 | 714 | 89.47 |
| 0.75 | 708 | 88.72 |
| 1.00 | 706 | 88.47 |

A support coefficient of 0.25 adds eight correct selections, or **1.00 percentage point**, over zero support. Larger weights lose this benefit. Agreement therefore helps this initial model when given a moderate weight; it does not prove that voting helps every model. The final hybrid uses zero voting weight.

An earlier comparison found 47 validation questions solved only by voting and 66 solved only by the initial embedding model. Their errors were complementary, motivating fusion rather than relying on either signal alone.

### 3.3 What the evidence supports

The recorded results support using evidence beyond majority agreement. They also motivate relative candidate comparison and arithmetic features. However, the available record does not isolate the contribution of every component of the final model. A stronger follow-up would remove each feature group while keeping the training data, regularization and fusion settings fixed.

<!-- pagebreak -->

## 4. Interpretation and limitations

### 4.1 Why validation is higher than test

Validation and test contain different mixtures of answer diversity. The final validation predictions break down as follows:

| Validation group | Correct / questions | Accuracy (%) |
| --- | ---: | ---: |
| One distinct parseable answer | 443/443 | 100.00 |
| Multiple distinct parseable answers | 311/355 | 87.61 |
| Complete validation bank | 754/798 | 94.49 |

Groups with only one distinct parseable answer make up 55.51% of validation. Because every official group has at least one correct candidate, any parseable candidate in such a group gives the correct final answer. Success on these groups provides little evidence of resolving competing answers.

All 467 test groups have multiple distinct valid parsed answers. Comparing the competing-answer validation score of 87.61% with the test score of 82.87% gives a **4.74-point gap**, smaller than the overall validation-test gap of 11.62 points. The different mixture helps interpret the gap, but does not prove that it explains the entire difference.

### 4.2 Failure analysis

The final results contain **44 validation errors** and **80 public test errors**. No measured breakdown by error cause was recorded. The implementation has several clear limitations:

- A calculation can be correct while solving the wrong interpretation of a question.
- The simple equation parser can skip variables, units or complex notation. Numeric extraction is not a complete quantity parser.
- The official final-answer parser can reject a correct-looking answer. Formatting also affects the numeric and answer-marker features.
- Formatting correlations may change on a new bank, and the model has no calibrated confidence or abstention option.

### 4.3 Evaluation history

Several intermediate methods were scored on the public test bank during development. Test-reference scores were also examined for exploratory intermediate parameter comparisons during AI-assisted development. This does not meet the course's strict rule reserving test references for final scoring and reproduction.

The reported test score is therefore **not an untouched-holdout estimate**. The final fitting script itself reads training references only, and inference reads no references or answer lookups. Evaluation of the frozen method on a separate unseen bank would give stronger evidence of generalization.

### 4.4 AI assistance and reuse

OpenAI Codex/ChatGPT substantially assisted with task explanation, method and feature design, Python code, debugging, experiment interpretation and documentation. This included the exploratory test-score comparisons disclosed above. The author ran the submitted scripts and official evaluator on a Windows computer.

Data, fixed representations, the interface and evaluator come from the official MP2 project. Questions and references derive from GSM8K. Candidate solutions and representations were prepared using Qwen3-0.6B as described in the official manifest; this method loads no Qwen model weights.

<!-- pagebreak -->

## 5. Resources and reproduction

### 5.1 Verified checkpoint record

These measurements were taken from the checkpoint actually uploaded to the public repository.

| Item | Record |
| --- | --- |
| Checkpoint | hybrid_valid_model.npz |
| NPZ file size | 15,140 bytes |
| Uncompressed NPY members, including headers | 13,824 bytes |
| Loaded numeric-array payload | 12,544 bytes |
| Contents | 10 float32 arrays: weights, normalization and settings |
| Python | 3.12, as reported by the author |
| Declared dependency | numpy==2.5.3; installed version not recorded |
| CPU / RAM | Not recorded |
| Final full evaluation time | Not measured |
| Training / aggregate search cost | Not recorded completely |
| Random seed | Not needed for final deterministic fitting |

All inference parameters are in this checkpoint. Even counting the full archive, its size is below the **16 MiB** limit. Runtime eligibility remains unverified: the course requires a fresh prediction process plus independent scoring, including startup and loading, within **30 seconds** on the reference CPU using four total threads. Scoring alone took 0.024935 seconds; this is not the full runtime. The earlier 0.403-second total belongs to a different model.

**Checkpoint SHA256:**

```text
f808a0cbdc07c3032a4b9986636013c4cbc2088a60cf4803acf05cc6e6a63b09
```

**Inspected code/checkpoint source commit:**

```text
c3bd1ab90a9782377d34734efbc28925d1d577b2
```

The final repository submission should identify the later commit containing this report and completed resource records.

### 5.2 Reproduce the score

Follow the repository README for installation and training. Copy the submitted scripts and checkpoint into the official project's code directory. Use four threads and run these as two separate commands, each entered on one line:

```powershell
py -3.12 predict.py --method hybrid_method.py --checkpoint hybrid_valid_model.npz --split test --output runs/reproduced-test.json
py -3.12 score_predictions.py --split test --predictions runs/reproduced-test.json --output runs/reproduced-test-score.json
```

Expected result: correct 387, n 467, accuracy 82.86937901498929, under protocol dase7506-mp2-open-hard-verifier-v5. No retraining is needed. Optional training uses `py -3.12 train_hybrid_valid.py` and writes a new checkpoint to runs/hybrid_valid_model.npz.

### Sources

- [Official MP2 project and guide](https://github.com/xudongwu-0/DASE7506-MP2).
- [Submission repository and official scoring JSONs](https://github.com/Gu123456-gu/DASE7506-MP2-solution).
- [GSM8K data attribution](https://github.com/openai/grade-school-math). Retain supplied license notices when redistributing official data.
