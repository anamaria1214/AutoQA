# AutoQA Model Comparison — Virtual Teaching Platform

## Overview

This project evaluates and compares the performance of two large language models — **GPT-4o** and **GPT-5.1** — in the automatic quality assessment (AutoQA) of recorded lessons on a virtual teaching platform. Beyond measuring inter-model reliability, the analysis explores whether the scores produced by each model are predictive of real student outcomes: specifically, whether a student attends the next lesson and whether they remain enrolled one month later.

The goal is to determine if automated lesson quality scores can serve as an early signal of student churn or retention.

---

## Research Questions

1. **Reliability** — How consistently do GPT-4o and GPT-5.1 score the same lessons? Do they agree on which items pass or fail?
2. **Predictive validity** — Are the items scored by the models statistically associated with whether a student attends the next class or stays enrolled after one month?
3. **Practical utility** — Which items are the most informative for identifying at-risk students early?

---

## Dataset

| File | Description |
|---|---|
| `docs/autoqa_output_gpt4o_240.csv` | Scores from GPT-4o across 16 quality items for 240 lesson records |
| `docs/autoqa_output_gpt51_240.csv` | Scores from GPT-5.1 across the same 16 items and records |
| `docs/outcomes_240.csv` | Student outcome data: next lesson attendance and month-1 retention |

Each record is identified by a `record_id` (UUID). Quality items (`item_1` through `item_16`) are binary (0/1).

**Outcome variables:**

- `next_lesson_attended` — Whether the student attended the immediately following lesson (0/1)
- `m1_retained` — Whether the student was still enrolled one month after the evaluated lesson (0/1)

---

## Methodology

### 1. Inter-Model Reliability

For each of the 16 quality items, the following are computed between GPT-4o and GPT-5.1:

- **Agreement rate** — Percentage of records where both models gave the same score
- **Cohen's Kappa (κ)** — Agreement corrected for chance. Interpreted as:
  - κ < 0.40 → Poor/Slight
  - 0.40 ≤ κ < 0.60 → Moderate
  - 0.60 ≤ κ < 0.80 → Substantial
  - κ ≥ 0.80 → Near-perfect
- **Prevalence Drift** — Absolute difference in the proportion of 1s between models, indicating systematic scoring bias

### 2. Predictive Validity

Using GPT-4o scores merged with student outcome data:

- **Point-biserial correlation** — Measures the strength of association between each binary item score and each continuous outcome variable
- **Retention gap** — Difference in mean item score between retained and churned students, indicating practical discriminative power
- **P-values** — Statistical significance of each correlation

---

## Project Structure

```
.
├── docs/
│   ├── autoqa_output_gpt4o_240.csv
│   ├── autoqa_output_gpt51_240.csv
│   └── outcomes_240.csv
├── script.py
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

---

## Usage

```bash
python script.py
```

The script will:

1. Load and merge all three CSV files
2. Compute reliability metrics (agreement, kappa, drift) for all 16 items
3. Compute predictive validity metrics (correlations, retention gap) for all 16 items
4. Print both result tables to the console
5. Display two plots:
   - **Cohen's Kappa by item** — Reliability across models
   - **Point-biserial correlation by item** — Predictive power for month-1 retention

---

## Output Tables

### Reliability Table

| Column | Description |
|---|---|
| `Item` | Quality item identifier (item_1 to item_16) |
| `Agreement (%)` | Percentage of records where both models agreed |
| `Kappa` | Cohen's Kappa (chance-corrected agreement) |
| `Drift` | Absolute difference in prevalence between models |

### Predictive Validity Table

| Column | Description |
|---|---|
| `Item` | Quality item identifier |
| `Corr_Next_Lesson` | Point-biserial correlation with next lesson attendance |
| `Corr_M1_Retained` | Point-biserial correlation with month-1 retention |
| `P_Value_M1` | P-value for the month-1 retention correlation |
| `Retention_Gap` | Difference in mean item score between retained vs. churned students |

---

## Interpreting Results

An item is considered **reliable** when its Kappa is ≥ 0.60, meaning both models agree substantially on how to score it regardless of the lesson.

An item is considered **predictively valid** when it shows a statistically significant correlation with `m1_retained` (p < 0.05) and a meaningful retention gap. Items that are both reliable and predictively valid are the strongest candidates for inclusion in an automated early-warning system for student churn.

Items with high drift but low kappa may reflect ambiguous rubric definitions that models interpret differently — these are candidates for rubric revision.

---

## Potential Applications

- **Early churn detection** — Flag students at risk of dropping out based on lesson quality scores from the previous session
- **Instructor feedback** — Identify which lesson quality dimensions most impact student continuation
- **Model selection** — Choose the most consistent and valid model for production AutoQA pipelines
- **Rubric refinement** — Use drift and kappa data to identify items that need clearer scoring criteria
