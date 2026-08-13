import time
import json
import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field

from .task_manager import TaskManager, EvalTask, EvalResult
from .llm_judge import LLMJudge, JudgeResult

@dataclass
class EvalConfig:
    """Configuration for evaluation run."""
    dataset_name: str
    model_fn: Callable[[str], str]
    judge: LLMJudge = None
    n_samples: int = None
    categories: List[str] = None
    difficulties: List[str] = None
    seed: int = 42
    timeout: float = 30.0
    retry_count: int = 2

@dataclass
class EvalRun:
    """Complete evaluation run results."""
    run_id: str
    config: Dict[str, Any]
    results: List[EvalResult]
    aggregated_scores: Dict[str, float]
    duration_seconds: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()

class EvalHarness:
    """End-to-end evaluation harness for LLMs."""
    
    def __init__(self, task_manager: TaskManager = None):
        self.task_manager = task_manager or TaskManager()
        self.logger = logging.getLogger(__name__)
        self.runs: List[EvalRun] = []
    
    def run_evaluation(self, config: EvalConfig,
                      model_name: str = 'unknown',
                      prompt_variant: str = 'default') -> EvalRun:
        """Run complete evaluation."""
        import uuid
        from datetime import datetime
        
        run_id = f"eval_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        self.logger.info(f"Starting evaluation: {run_id}")
        
        # Get tasks
        tasks = self.task_manager.filter_tasks(
            config.dataset_name,
            category=config.categories[0] if config.categories else None,
            difficulty=config.difficulties[0] if config.difficulties else None,
            limit=config.n_samples,
        )
        
        if not tasks:
            raise ValueError(f"No tasks found for dataset '{config.dataset_name}'")
        
        self.logger.info(f"Running {len(tasks)} evaluations")
        
        # Run evaluations
        results = []
        for i, task in enumerate(tasks):
            result = self._evaluate_single_task(
                task, config, model_name, prompt_variant
            )
            results.append(result)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {i+1}/{len(tasks)}")
        
        # Aggregate scores
        aggregated = self._aggregate_results(results)
        
        duration = time.time() - start_time
        
        eval_run = EvalRun(
            run_id=run_id,
            config={
                'dataset': config.dataset_name,
                'model': model_name,
                'prompt_variant': prompt_variant,
                'n_samples': len(tasks),
            },
            results=results,
            aggregated_scores=aggregated,
            duration_seconds=duration,
        )
        
        self.runs.append(eval_run)
        
        self.logger.info(f"Evaluation complete: {run_id}")
        self.logger.info(f"Overall score: {aggregated.get('overall', 0):.3f}")
        
        return eval_run
    
    def _evaluate_single_task(self, task: EvalTask, config: EvalConfig,
                             model_name: str, prompt_variant: str) -> EvalResult:
        """Evaluate a single task."""
        # Get model output
        model_output = None
        error = None
        latency = 0
        
        for attempt in range(config.retry_count + 1):
            try:
                start = time.time()
                model_output = config.model_fn(task.input_text)
                latency = (time.time() - start) * 1000
                break
            except Exception as e:
                error = str(e)
                self.logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt < config.retry_count:
                    time.sleep(1)
        
        # Judge the output
        scores = {}
        if model_output and config.judge:
            try:
                judge_prompt = config.judge.build_judge_prompt(
                    task.input_text,
                    model_output,
                    task.expected_output,
                    task.context,
                )
                
                # In production, call LLM here
                # For now, use placeholder scoring
                scores = self._placeholder_scoring(
                    task.input_text, model_output, task.expected_output
                )
            except Exception as e:
                self.logger.warning(f"Judge failed: {e}")
        
        return EvalResult(
            task_id=task.task_id,
            model_name=model_name,
            prompt_variant=prompt_variant,
            model_output=model_output,
            scores=scores,
            latency_ms=latency,
            error=error,
        )
    
    def _placeholder_scoring(self, input_text: str, output: str,
                           expected: str = None) -> Dict[str, float]:
        """Placeholder scoring for demo purposes."""
        import random
        
        # Simple token overlap scoring
        output_tokens = set(output.lower().split())
        
        scores = {
            'relevance': random.uniform(0.6, 0.95),
            'accuracy': random.uniform(0.5, 0.9),
            'completeness': random.uniform(0.4, 0.85),
            'clarity': random.uniform(0.6, 0.95),
        }
        
        if expected:
            expected_tokens = set(expected.lower().split())
            overlap = len(output_tokens & expected_tokens) / max(len(expected_tokens), 1)
            scores['exact_match'] = overlap
        
        return scores
    
    def _aggregate_results(self, results: List[EvalResult]) -> Dict[str, float]:
        """Aggregate results across all tasks."""
        if not results:
            return {}
        
        all_scores = {}
        for result in results:
            for metric, score in result.scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        aggregated = {}
        for metric, scores in all_scores.items():
            aggregated[metric] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
            }
        
        # Overall score
        all_means = [v['mean'] for v in aggregated.values()]
        aggregated['overall'] = float(np.mean(all_means)) if all_means else 0
        
        return aggregated
    
    def compare_runs(self, run1: EvalRun, run2: EvalRun) -> Dict[str, Any]:
        """Compare two evaluation runs."""
        metrics = set(run1.aggregated_scores.keys()) | set(run2.aggregated_scores.keys())
        
        comparison = {}
        for metric in metrics:
            score1 = run1.aggregated_scores.get(metric, {})
            score2 = run2.aggregated_scores.get(metric, {})
            
            if isinstance(score1, dict) and isinstance(score2, dict):
                comparison[metric] = {
                    'run1': score1.get('mean', 0),
                    'run2': score2.get('mean', 0),
                    'diff': score2.get('mean', 0) - score1.get('mean', 0),
                }
        
        return {
            'run1_id': run1.run_id,
            'run2_id': run2.run_id,
            'comparison': comparison,
            'winner': 'run1' if run1.aggregated_scores.get('overall', 0) >
                      run2.aggregated_scores.get('overall', 0) else 'run2',
        }
    
    def export_results(self, run: EvalRun, path: str):
        """Export evaluation results to JSON."""
        data = {
            'run_id': run.run_id,
            'config': run.config,
            'aggregated_scores': run.aggregated_scores,
            'duration_seconds': run.duration_seconds,
            'timestamp': run.timestamp,
            'results': [
                {
                    'task_id': r.task_id,
                    'model_name': r.model_name,
                    'prompt_variant': r.prompt_variant,
                    'scores': r.scores,
                    'latency_ms': r.latency_ms,
                    'error': r.error,
                }
                for r in run.results
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def generate_report(self, run: EvalRun) -> str:
        """Generate evaluation report."""
        scores = run.aggregated_scores
        
        lines = [
            f"# Evaluation Report: {run.run_id}",
            f"",
            f"## Configuration",
            f"- Dataset: {run.config.get('dataset', 'N/A')}",
            f"- Model: {run.config.get('model', 'N/A')}",
            f"- Prompt Variant: {run.config.get('prompt_variant', 'N/A')}",
            f"- Samples: {run.config.get('n_samples', len(run.results))}",
            f"- Duration: {run.duration_seconds:.1f}s",
            f"",
            f"## Scores",
        ]
        
        for metric, value in sorted(scores.items()):
            if isinstance(value, dict):
                lines.append(f"- **{metric}**: {value['mean']:.3f} ± {value['std']:.3f}")
            else:
                lines.append(f"- **{metric}**: {value:.3f}")
        
        # Error summary
        errors = [r for r in run.results if r.error]
        if errors:
            lines.append(f"")
            lines.append(f"## Errors ({len(errors)})")
            for e in errors[:5]:
                lines.append(f"- Task {e.task_id}: {e.error}")
        
        return "\n".join(lines)
