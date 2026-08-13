import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class RegressionResult:
    """Result of regression detection analysis."""
    metric_name: str
    baseline_mean: float
    current_mean: float
    change: float
    change_pct: float
    is_regression: bool
    severity: str
    p_value: float
    statistical_power: float
    details: Dict[str, Any] = field(default_factory=dict)

class RegressionDetector:
    """Detect performance regressions when model versions change."""

    def __init__(self, threshold_pct: float = 5.0,
                 significance_level: float = 0.05,
                 min_effect_size: float = 0.2):
        self.threshold_pct = threshold_pct
        self.significance_level = significance_level
        self.min_effect_size = min_effect_size
        self.baseline_scores: Dict[str, List[float]] = {}
        self.current_scores: Dict[str, List[float]] = {}

    def set_baseline(self, scores: Dict[str, List[float]]):
        """Set baseline model scores."""
        self.baseline_scores = scores

    def set_current(self, scores: Dict[str, List[float]]):
        """Set current model scores."""
        self.current_scores = scores

    def detect_regressions(self) -> List[RegressionResult]:
        """Detect regressions across all metrics."""
        results = []

        all_metrics = set(self.baseline_scores.keys()) | set(self.current_scores.keys())

        for metric in all_metrics:
            baseline = self.baseline_scores.get(metric, [])
            current = self.current_scores.get(metric, [])

            if baseline and current:
                result = self.detect_metric_regression(metric, baseline, current)
                results.append(result)

        return results

    def detect_metric_regression(self, metric_name: str,
                                baseline: List[float],
                                current: List[float]) -> RegressionResult:
        """Detect regression for a single metric."""
        baseline_mean = np.mean(baseline)
        current_mean = np.mean(current)

        change = current_mean - baseline_mean
        change_pct = (change / baseline_mean * 100) if baseline_mean != 0 else 0

        # Statistical tests
        # Welch's t-test (doesn't assume equal variance)
        t_stat, p_value = stats.ttest_ind(current, baseline, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(baseline, ddof=1) + np.var(current, ddof=1)) / 2
        )
        cohens_d = abs(change) / pooled_std if pooled_std > 0 else 0

        # Statistical power (approximate)
        power = self._compute_power(len(baseline), len(current), cohens_d)

        # Determine if regression (for most metrics, decrease is bad)
        is_regression = (
            change_pct < -self.threshold_pct and
            p_value < self.significance_level and
            cohens_d >= self.min_effect_size
        )

        # Severity
        severity = self._classify_severity(change_pct, p_value, cohens_d)

        return RegressionResult(
            metric_name=metric_name,
            baseline_mean=float(baseline_mean),
            current_mean=float(current_mean),
            change=float(change),
            change_pct=float(change_pct),
            is_regression=is_regression,
            severity=severity,
            p_value=float(p_value),
            statistical_power=float(power),
            details={
                't_statistic': float(t_stat),
                'cohens_d': float(cohens_d),
                'baseline_n': len(baseline),
                'current_n': len(current),
                'baseline_std': float(np.std(baseline, ddof=1)),
                'current_std': float(np.std(current, ddof=1)),
            }
        )

    def _compute_power(self, n1: int, n2: int, effect_size: float) -> float:
        """Approximate statistical power."""
        from scipy.stats import norm

        alpha = self.significance_level
        z_alpha = norm.ppf(1 - alpha / 2)

        se = np.sqrt(1/n1 + 1/n2)
        ncp = effect_size / se

        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        return float(power)

    def _classify_severity(self, change_pct: float, p_value: float,
                          cohens_d: float) -> str:
        """Classify regression severity."""
        if change_pct < -20 and p_value < 0.01 and cohens_d > 0.8:
            return 'critical'
        elif change_pct < -10 and p_value < 0.05:
            return 'high'
        elif change_pct < -5 and p_value < 0.1:
            return 'medium'
        elif change_pct < -self.threshold_pct:
            return 'low'
        else:
            return 'none'

    def compare_models(self, baseline_name: str, baseline_scores: Dict[str, List[float]],
                      current_name: str, current_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compare two model versions comprehensively."""
        self.baseline_scores = baseline_scores
        self.current_scores = current_scores

        regressions = self.detect_regressions()

        # Summary
        n_regressions = sum(1 for r in regressions if r.is_regression)
        n_improvements = sum(1 for r in regressions if r.change_pct > self.threshold_pct)

        severity_counts = {}
        for r in regressions:
            severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1

        return {
            'baseline': baseline_name,
            'current': current_name,
            'n_metrics': len(regressions),
            'n_regressions': n_regressions,
            'n_improvements': n_improvements,
            'severity_counts': severity_counts,
            'regressions': [r for r in regressions if r.is_regression],
            'all_results': regressions,
            'overall_status': 'regression' if n_regressions > 0 else 'stable',
        }

    def sliding_window_detection(self, scores: List[float],
                                window_size: int = 30,
                                step: int = 1) -> List[Dict]:
        """Detect regressions using sliding window."""
        results = []

        for i in range(0, len(scores) - window_size, step):
            window = scores[i:i+window_size]
            baseline = scores[:i] if i > 0 else scores[:window_size]

            if len(baseline) < 10:
                continue

            t_stat, p_value = stats.ttest_ind(window, baseline, equal_var=False)
            mean_change = np.mean(window) - np.mean(baseline)
            pct_change = mean_change / np.mean(baseline) * 100 if np.mean(baseline) != 0 else 0

            results.append({
                'window_start': i,
                'window_end': i + window_size,
                'window_mean': float(np.mean(window)),
                'baseline_mean': float(np.mean(baseline)),
                'change_pct': float(pct_change),
                'p_value': float(p_value),
                'is_regression': pct_change < -self.threshold_pct and p_value < self.significance_level,
            })

        return results

    def generate_report(self, comparison: Dict) -> str:
        """Generate regression detection report."""
        lines = [
            f"# Regression Detection Report",
            f"",
            f"Baseline: {comparison['baseline']}",
            f"Current: {comparison['current']}",
            f"",
            f"## Summary",
            f"- Metrics analyzed: {comparison['n_metrics']}",
            f"- Regressions detected: {comparison['n_regressions']}",
            f"- Improvements: {comparison['n_improvements']}",
            f"- Overall status: **{comparison['overall_status'].upper()}**",
            f"",
            f"## Regressions",
        ]

        for r in comparison.get('regressions', []):
            lines.append(f"")
            lines.append(f"### {r.metric_name}")
            lines.append(f"- Change: {r.change_pct:+.1f}%")
            lines.append(f"- Baseline: {r.baseline_mean:.4f} → Current: {r.current_mean:.4f}")
            lines.append(f"- p-value: {r.p_value:.4f}")
            lines.append(f"- Severity: **{r.severity}**")

        if not comparison.get('regressions'):
            lines.append(f"")
            lines.append(f"No regressions detected.")

        return "\n".join(lines)
