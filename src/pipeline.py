import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

from .task_manager import TaskManager, EvalTask
from .llm_judge import LLMJudge, RubricCriterion
from .ragas_metrics import RAGASMetrics
from .regression_detector import RegressionDetector
from .ab_testing import ABTester, ExperimentVariant
from .eval_harness import EvalHarness, EvalConfig
from .dashboard import LeaderboardDashboard

class EvalPipeline:
    """Main pipeline orchestrating all evaluation components."""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.task_manager = TaskManager(os.path.join(output_dir, 'data'))
        self.judge = LLMJudge()
        self.ragas = RAGASMetrics()
        self.regression_detector = RegressionDetector()
        self.ab_tester = ABTester()
        self.dashboard = LeaderboardDashboard(output_dir)
        self.evalHarness = EvalHarness(self.task_manager)
        
        self.results = {}
    
    def run_full_evaluation(self, dataset_name: str,
                           models: Dict[str, Callable],
                           prompt_variants: Dict[str, str] = None,
                           rubric: List[RubricCriterion] = None) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("LLM EVALS FRAMEWORK - FULL EVALUATION")
        self.logger.info("=" * 60)
        
        if rubric:
            self.judge = LLMJudge(custom_rubric=rubric)
        
        all_results = {}
        
        # Evaluate each model
        for model_name, model_fn in models.items():
            self.logger.info(f"\nEvaluating model: {model_name}")
            
            variants = prompt_variants or {'default': ''}
            
            for variant_name, prompt_template in variants.items():
                self.logger.info(f"  Variant: {variant_name}")
                
                # Create wrapped model function
                def wrapped_model(text, pt=prompt_template, fn=model_fn):
                    prompt = pt.format(input=text) if pt else text
                    return fn(prompt)
                
                config = EvalConfig(
                    dataset_name=dataset_name,
                    model_fn=wrapped_model,
                    judge=self.judge,
                )
                
                run = self.evalHarness.run_evaluation(
                    config,
                    model_name=model_name,
                    prompt_variant=variant_name,
                )
                
                key = f"{model_name}_{variant_name}"
                all_results[key] = run
                
                # Add to dashboard
                self.dashboard.add_entry(
                    model_name=model_name,
                    prompt_variant=variant_name,
                    scores=run.aggregated_scores,
                    n_evaluations=len(run.results),
                )
        
        # Generate reports
        self.dashboard.save_snapshot()
        leaderboard_path = self.dashboard.generate_html_report()
        
        # Compare models
        model_comparisons = {}
        model_names = list(models.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                comparison = self.dashboard.compare_models(model_names[i], model_names[j])
                model_comparisons[f"{model_names[i]}_vs_{model_names[j]}"] = comparison
        
        self.results = {
            'runs': {k: v.run_id for k, v in all_results.items()},
            'leaderboard': leaderboard_path,
            'comparisons': model_comparisons,
            'summary': self.dashboard.generate_summary(),
        }
        
        return self.results
    
    def run_ragas_evaluation(self, dataset_name: str,
                            rag_fn: Callable[[str], Dict],
                            questions: List[str] = None) -> Dict[str, Any]:
        """Run RAGAS-specific evaluation."""
        self.logger.info("=" * 60)
        self.logger.info("RAGAS EVALUATION")
        self.logger.info("=" * 60)
        
        tasks = self.task_manager.load_dataset(dataset_name)
        
        if questions:
            tasks = [t for t in tasks if t.input_text in questions]
        
        ragas_results = []
        
        for task in tasks:
            try:
                rag_output = rag_fn(task.input_text)
                
                result = self.ragas.compute_all(
                    answer=rag_output.get('answer', ''),
                    question=task.input_text,
                    context=rag_output.get('context', []),
                    reference=task.expected_output,
                )
                
                ragas_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"RAGAS eval failed for task {task.task_id}: {e}")
        
        aggregated = self.ragas.aggregate_results(ragas_results)
        
        self.results['ragas'] = aggregated
        return aggregated
    
    def run_regression_check(self, baseline_scores: Dict[str, List[float]],
                            current_scores: Dict[str, List[float]],
                            baseline_name: str = 'baseline',
                            current_name: str = 'current') -> Dict[str, Any]:
        """Run regression detection between model versions."""
        self.logger.info("=" * 60)
        self.logger.info("REGRESSION DETECTION")
        self.logger.info("=" * 60)
        
        comparison = self.regression_detector.compare_models(
            baseline_name, baseline_scores,
            current_name, current_scores,
        )
        
        report = self.regression_detector.generate_report(comparison)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'regression_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.results['regression'] = {
            'comparison': {
                k: v for k, v in comparison.items()
                if k != 'all_results'
            },
            'report_path': report_path,
        }
        
        return self.results['regression']
    
    def run_ab_test(self, experiment_name: str,
                   variants: List[Dict],
                   evaluation_fn: Callable) -> Dict[str, Any]:
        """Run prompt A/B test."""
        self.logger.info("=" * 60)
        self.logger.info(f"A/B TEST: {experiment_name}")
        self.logger.info("=" * 60)
        
        exp_variants = [
            ExperimentVariant(
                variant_id=v['id'],
                name=v['name'],
                prompt=v['prompt'],
                model=v.get('model'),
            )
            for v in variants
        ]
        
        exp_id = self.ab_tester.create_experiment(experiment_name, exp_variants)
        
        # Run evaluation for each variant
        for variant in exp_variants:
            scores = evaluation_fn(variant.prompt)
            self.ab_tester.record_batch(exp_id, {variant.variant_id: scores})
        
        # Analyze
        result = self.ab_tester.analyze_experiment(exp_id)
        
        # Bayesian analysis
        bayesian = self.ab_tester.bayesian_analysis(exp_id)
        
        # Generate report
        report = self.ab_tester.generate_report(exp_id)
        
        self.results['ab_test'] = {
            'experiment_id': exp_id,
            'winner': result.winner,
            'is_significant': result.is_significant,
            'p_value': result.p_value,
            'lift': result.lift,
            'bayesian': bayesian,
            'report': report,
        }
        
        return self.results['ab_test']
    
    def save_all_results(self):
        """Save all results to files."""
        path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"All results saved to {path}")
