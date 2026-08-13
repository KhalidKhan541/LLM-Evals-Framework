# LLM Evaluation + Evals Framework

End-to-end evaluation system: task management, LLM-as-judge scoring, RAGAS metrics, regression detection, A/B testing, and leaderboard dashboard.

## Features

| Feature | Description |
|---------|-------------|
| **Task Management** | Dataset loading, filtering, splitting, stratified sampling |
| **LLM-as-Judge** | Rubric-based scoring with custom criteria and pairwise comparison |
| **RAGAS Metrics** | Faithfulness, relevance, context precision/recall for RAG |
| **Regression Detection** | Statistical tests (Welch's t, Cohen's d) when model versions change |
| **A/B Testing** | Prompt variant testing with Bayesian and frequentist analysis |
| **Leaderboard** | Dashboard comparing models + prompt variants |

## Quick Start

```bash
pip install -r requirements.txt

# Run evaluation
python run.py eval my_dataset --model my_model

# Run RAGAS evaluation
python run.py ragas my_dataset --rag-function my_rag_fn

# Compare model versions
python run.py compare baseline_scores.json current_scores.json

# Generate dashboard
python run.py dashboard --entries leaderboard_entries.json
```

## Usage Examples

```python
from src.pipeline import EvalPipeline

pipeline = EvalPipeline(output_dir='output')

# Full evaluation
results = pipeline.run_full_evaluation(
    dataset_name='qa_dataset',
    models={'gpt4': gpt4_fn, 'claude': claude_fn},
    prompt_variants={'v1': '{input}', 'v2': 'Answer: {input}'},
)

# RAGAS evaluation
ragas_results = pipeline.run_ragas_evaluation('rag_dataset', rag_fn)

# Regression check
pipeline.run_regression_check(baseline_scores, current_scores)

# A/B test
pipeline.run_ab_test(
    'prompt_comparison',
    [{'id': 'a', 'name': 'Short', 'prompt': '{input}'},
     {'id': 'b', 'name': 'Detailed', 'prompt': 'Please answer: {input}'}],
    eval_fn,
)
```

## Architecture

```
LLM-Evals-Framework/
├── src/
│   ├── task_manager.py        # Dataset management
│   ├── llm_judge.py           # LLM-as-judge with rubrics
│   ├── ragas_metrics.py       # RAGAS faithfulness/relevance
│   ├── regression_detector.py # Statistical regression detection
│   ├── ab_testing.py          # Prompt A/B testing
│   ├── dashboard.py           # Leaderboard dashboard
│   ├── eval_harness.py        # Core evaluation harness
│   └── pipeline.py            # Main orchestrator
├── run.py                     # CLI
├── requirements.txt
└── README.md
```

## Dependencies

- numpy, scipy, pandas - Statistics and data
- scikit-learn - ML utilities
