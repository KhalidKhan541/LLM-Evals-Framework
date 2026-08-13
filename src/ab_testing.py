import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

@dataclass
class ExperimentVariant:
    """A variant in an A/B test."""
    variant_id: str
    name: str
    prompt: str
    model: str = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    """Result of A/B test for a variant."""
    variant_id: str
    n_samples: int
    scores: List[float]
    mean_score: float
    std_score: float
    ci_lower: float
    ci_upper: float
    
@dataclass
class ABTestResult:
    """Complete A/B test result."""
    experiment_name: str
    variants: List[ExperimentResult]
    winner: str = None
    p_value: float = None
    is_significant: bool = False
    lift: float = None
    details: Dict[str, Any] = field(default_factory=dict)

class ABTester:
    """Prompt A/B testing with statistical significance."""
    
    def __init__(self, confidence_level: float = 0.95,
                 min_samples: int = 30):
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        self.experiments: Dict[str, Dict] = {}
    
    def create_experiment(self, name: str,
                         variants: List[ExperimentVariant]) -> str:
        """Create a new A/B experiment."""
        exp_id = hashlib.md5(name.encode()).hexdigest()[:8]
        
        self.experiments[exp_id] = {
            'name': name,
            'variants': {v.variant_id: v for v in variants},
            'results': {},
            'created_at': datetime.now().isoformat(),
        }
        
        return exp_id
    
    def record_result(self, experiment_id: str,
                     variant_id: str, score: float):
        """Record a single evaluation result."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        if variant_id not in exp['results']:
            exp['results'][variant_id] = []
        
        exp['results'][variant_id].append(score)
    
    def record_batch(self, experiment_id: str,
                    variant_scores: Dict[str, List[float]]):
        """Record batch of results for all variants."""
        for variant_id, scores in variant_scores.items():
            for score in scores:
                self.record_result(experiment_id, variant_id, score)
    
    def analyze_experiment(self, experiment_id: str) -> ABTestResult:
        """Analyze A/B test results with statistical tests."""
        exp = self.experiments[experiment_id]
        
        variant_results = []
        all_scores = {}
        
        for variant_id, scores in exp['results'].items():
            if len(scores) < 2:
                continue
            
            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            n = len(scores)
            
            se = std / np.sqrt(n)
            z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            
            result = ExperimentResult(
                variant_id=variant_id,
                n_samples=n,
                scores=scores,
                mean_score=float(mean),
                std_score=float(std),
                ci_lower=float(mean - z * se),
                ci_upper=float(mean + z * se),
            )
            variant_results.append(result)
            all_scores[variant_id] = scores
        
        # Statistical testing
        winner = None
        p_value = None
        is_significant = False
        lift = None
        
        if len(variant_results) >= 2:
            # Compare variants pairwise
            best = max(variant_results, key=lambda x: x.mean_score)
            runner_up = sorted(variant_results, key=lambda x: x.mean_score, reverse=True)[1]
            
            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(
                best.scores, runner_up.scores, equal_var=False
            )
            
            p_value = float(p_val)
            is_significant = p_val < (1 - self.confidence_level)
            
            if is_significant:
                winner = best.variant_id
                lift = (best.mean_score - runner_up.mean_score) / runner_up.mean_score * 100
        
        return ABTestResult(
            experiment_name=exp['name'],
            variants=variant_results,
            winner=winner,
            p_value=p_value,
            is_significant=is_significant,
            lift=float(lift) if lift else None,
            details={
                'confidence_level': self.confidence_level,
                'min_samples': self.min_samples,
                'total_samples': sum(v.n_samples for v in variant_results),
            }
        )
    
    def bayesian_analysis(self, experiment_id: str,
                         prior_alpha: float = 1,
                         prior_beta: float = 1) -> Dict[str, Any]:
        """Bayesian A/B analysis using Beta-Binomial model."""
        exp = self.experiments[experiment_id]
        
        results = {}
        for variant_id, scores in exp['results'].items():
            # Convert scores to binary (success/failure relative to threshold)
            threshold = np.median(scores)
            successes = sum(1 for s in scores if s > threshold)
            n = len(scores)
            
            # Posterior
            post_alpha = prior_alpha + successes
            post_beta = prior_beta + (n - successes)
            
            # Sample posterior
            samples = np.random.beta(post_alpha, post_beta, 10000)
            
            results[variant_id] = {
                'posterior_mean': float(np.mean(samples)),
                'posterior_std': float(np.std(samples)),
                'credible_interval': float(np.percentile(samples, [2.5, 97.5])),
                'prob_best': 0,  # Calculated below
            }
        
        # Calculate probability of being best
        variant_ids = list(results.keys())
        if len(variant_ids) >= 2:
            samples_dict = {}
            for vid in variant_ids:
                scores = exp['results'][vid]
                threshold = np.median(scores)
                successes = sum(1 for s in scores if s > threshold)
                n = len(scores)
                samples_dict[vid] = np.random.beta(
                    prior_alpha + successes,
                    prior_beta + (n - successes),
                    10000
                )
            
            # Count wins
            all_samples = np.column_stack([samples_dict[vid] for vid in variant_ids])
            winners = np.argmax(all_samples, axis=1)
            
            for i, vid in enumerate(variant_ids):
                results[vid]['prob_best'] = float(np.mean(winners == i))
        
        return results
    
    def required_sample_size(self, mde: float = 0.05,
                            baseline_std: float = 0.5) -> int:
        """Calculate required sample size for desired power."""
        alpha = 1 - self.confidence_level
        power = 0.8
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) ** 2 * baseline_std ** 2) / (mde ** 2)
        
        return int(np.ceil(n))
    
    def sequential_testing(self, experiment_id: str,
                          variant_id: str,
                          look_fraction: float = 0.2) -> Dict[str, Any]:
        """Sequential testing with optional stopping."""
        exp = self.experiments[experiment_id]
        scores = exp['results'].get(variant_id, [])
        
        if len(scores) < self.min_samples:
            return {'can_stop': False, 'reason': 'insufficient_samples'}
        
        n = len(scores)
        look_at = int(n * look_fraction)
        
        # O'Brien-Fleming boundary
        n_looks = int(1 / look_fraction)
        
        for i in range(1, n_looks + 1):
            fraction = i * look_fraction
            boundary = stats.norm.ppf(1 - (1 - self.confidence_level) / (2 * i))
            
            current_scores = scores[:int(n * fraction)]
            mean = np.mean(current_scores)
            se = np.std(current_scores, ddof=1) / np.sqrt(len(current_scores))
            
            z = mean / se if se > 0 else 0
            
            if abs(z) > boundary:
                return {
                    'can_stop': True,
                    'reason': 'significance_reached',
                    'look': i,
                    'z_stat': float(z),
                    'boundary': float(boundary),
                }
        
        return {'can_stop': False, 'reason': 'continue'}
    
    def generate_report(self, experiment_id: str) -> str:
        """Generate A/B test report."""
        result = self.analyze_experiment(experiment_id)
        
        lines = [
            f"# A/B Test Report: {result.experiment_name}",
            f"",
            f"## Configuration",
            f"- Confidence Level: {result.details['confidence_level']:.0%}",
            f"- Total Samples: {result.details['total_samples']}",
            f"",
            f"## Results",
        ]
        
        for v in sorted(result.variants, key=lambda x: x.mean_score, reverse=True):
            lines.append(f"")
            lines.append(f"### {v.variant_id}")
            lines.append(f"- Mean Score: {v.mean_score:.4f} ± {v.std_score:.4f}")
            lines.append(f"- 95% CI: [{v.ci_lower:.4f}, {v.ci_upper:.4f}]")
            lines.append(f"- Samples: {v.n_samples}")
        
        lines.append(f"")
        lines.append(f"## Statistical Analysis")
        lines.append(f"- p-value: {result.p_value:.4f}" if result.p_value else "- p-value: N/A")
        lines.append(f"- Significant: {'Yes' if result.is_significant else 'No'}")
        
        if result.winner:
            lines.append(f"- Winner: **{result.winner}**")
            lines.append(f"- Lift: {result.lift:+.1f}%")
        
        return "\n".join(lines)
