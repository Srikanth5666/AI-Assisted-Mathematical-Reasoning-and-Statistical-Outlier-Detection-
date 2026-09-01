# AI-Assisted Mathematical Reasoning and Statistical Outlier Detection

**Independent Mathematics & AI Project — 2026**  
**Author:** Venkata Srikanth Kompala  
**Academic background:** Bachelor of Science in Mathematical Sciences — Mathematics, Statistics & Computer Science

## Overview

This independent project explores how classical statistical reasoning and machine-learning-based anomaly detection can be combined with human verification to identify unusual observations in a synthetic quantitative dataset.

The project is designed around a current AI-training workflow: generate a quantitative problem, establish a verifiable mathematical/statistical solution, use computational tools to test the result, and critically evaluate the output rather than treating model output as automatically correct.

## Objectives

1. Apply descriptive statistics to a structured dataset.
2. Detect unusual observations using standardized scores.
3. Compare statistical screening with Isolation Forest.
4. Examine why an observation may be mathematically/statistically unusual.
5. Demonstrate a human-in-the-loop validation workflow relevant to AI evaluation.
6. Produce reproducible analysis using Python.

## Dataset

The dataset is **synthetic and educational**. It does not represent real students, customers, employees, or any other identifiable people.

Variables:

- `study_hours`: simulated weekly study hours
- `practice_score`: simulated quantitative practice score
- `accuracy`: simulated solution accuracy
- `record_id`: synthetic record identifier

Six observations were deliberately injected as unusual cases to test the detection workflow.

## Mathematical Method

For a variable x, the standardized score is:

z = (x - μ) / σ

where μ is the sample mean and σ is the population-style standard deviation used for the screening calculation.

A record is flagged by the statistical screen when its maximum absolute standardized score exceeds **2.5**.

## Machine-Learning Method

The project also uses `IsolationForest`. The algorithm isolates observations through recursive random partitioning; observations that require shorter paths to become isolated are more likely to be anomalies.

The implementation uses:

- 200 trees
- contamination = 0.05
- random_state = 42

## Human Verification / AI Evaluation Angle

A useful AI-evaluation workflow is:

**Prompt → AI solution → independent mathematical derivation → computational check → error analysis → final ranking**

For a mathematical answer, the reviewer should check:

- arithmetic correctness
- formula selection
- assumptions
- logical sequence
- completeness
- final answer
- edge cases
- consistency with an independent calculation

The project deliberately treats AI as an assistant rather than as the final authority.

## Results

The synthetic experiment contains **6 deliberately injected anomalies**.

The combined screening method flagged **6 records**.

Against the synthetic ground truth:

- True positives: **6**
- False positives: **0**
- False negatives: **0**
- True negatives: **114**
- Precision: **100.00%**
- Recall: **100.00%**
- F1 score: **100.00%**

These metrics are included only as an educational benchmark because the ground truth was intentionally constructed for this project.

## Files

- `analysis.py` — complete reproducible Python analysis
- `outlier_analysis_results.csv` — generated analysis output
- `requirements.txt` — Python dependencies
- `project_report.pdf` — professional project report
- `figures/` — visualisations

## How to Run

```bash
python -m pip install -r requirements.txt
python analysis.py
```

## Limitations

- The dataset is synthetic.
- The injected anomalies were intentionally constructed.
- The project does not claim clinical, financial, educational, or operational validity.
- Outlier detection is context-dependent; a statistical anomaly is not automatically an error.
- Machine-learning scores should be interpreted alongside domain knowledge.

## Future Work

Possible extensions include:

- testing Local Outlier Factor alongside Isolation Forest
- using robust statistics such as median/MAD
- evaluating sensitivity to different contamination assumptions
- creating an AI benchmark of mathematical prompts
- comparing multiple LLM-generated solutions against independently verified answers
- building a rubric for mathematical reasoning quality
- adding formal LaTeX proofs and symbolic verification

## Academic Integrity Note

This is an **independent project created in 2026**, not a project completed as part of the author's bachelor's degree and not a peer-reviewed publication. The dataset and results are created for educational demonstration.

## References

- scikit-learn documentation: Isolation Forest and outlier detection.
- General statistical concepts: standardization, descriptive statistics, and anomaly detection.
