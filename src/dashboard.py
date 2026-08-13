import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class LeaderboardEntry:
    """Single entry in model leaderboard."""
    model_name: str
    prompt_variant: str
    overall_score: float
    metric_scores: Dict[str, float]
    n_evaluations: int
    rank: int = 0
    latency_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class LeaderboardDashboard:
    """Leaderboard dashboard comparing models and prompt variants."""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.entries: List[LeaderboardEntry] = []
        self.history: List[Dict] = []
    
    def add_entry(self, model_name: str, prompt_variant: str,
                 scores: Dict[str, float], n_evaluations: int,
                 latency_ms: float = 0, metadata: Dict = None):
        """Add or update leaderboard entry."""
        # Check if entry exists
        existing = None
        for entry in self.entries:
            if entry.model_name == model_name and entry.prompt_variant == prompt_variant:
                existing = entry
                break
        
        overall = np.mean(list(scores.values())) if scores else 0
        
        if existing:
            existing.metric_scores = scores
            existing.overall_score = overall
            existing.n_evaluations = n_evaluations
            existing.latency_ms = latency_ms
            if metadata:
                existing.metadata = metadata
        else:
            entry = LeaderboardEntry(
                model_name=model_name,
                prompt_variant=prompt_variant,
                overall_score=overall,
                metric_scores=scores,
                n_evaluations=n_evaluations,
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
            self.entries.append(entry)
        
        self._update_ranks()
    
    def _update_ranks(self):
        """Update rankings based on overall score."""
        self.entries.sort(key=lambda x: x.overall_score, reverse=True)
        for i, entry in enumerate(self.entries):
            entry.rank = i + 1
    
    def get_leaderboard(self, top_n: int = None) -> List[Dict]:
        """Get current leaderboard."""
        entries = self.entries[:top_n] if top_n else self.entries
        
        return [
            {
                'rank': e.rank,
                'model': e.model_name,
                'prompt': e.prompt_variant,
                'overall': e.overall_score,
                'metrics': e.metric_scores,
                'n_evals': e.n_evaluations,
                'latency': e.latency_ms,
            }
            for e in entries
        ]
    
    def compare_models(self, model_a: str, model_b: str) -> Dict[str, Any]:
        """Compare two models head-to-head."""
        entries_a = [e for e in self.entries if e.model_name == model_a]
        entries_b = [e for e in self.entries if e.model_name == model_b]
        
        if not entries_a or not entries_b:
            return {'error': 'Model not found'}
        
        avg_a = np.mean([e.overall_score for e in entries_a])
        avg_b = np.mean([e.overall_score for e in entries_b])
        
        # Metric-by-metric comparison
        all_metrics = set()
        for e in entries_a + entries_b:
            all_metrics.update(e.metric_scores.keys())
        
        metric_comparison = {}
        for metric in all_metrics:
            scores_a = [e.metric_scores.get(metric, 0) for e in entries_a]
            scores_b = [e.metric_scores.get(metric, 0) for e in entries_b]
            metric_comparison[metric] = {
                model_a: float(np.mean(scores_a)),
                model_b: float(np.mean(scores_b)),
                'winner': model_a if np.mean(scores_a) > np.mean(scores_b) else model_b,
            }
        
        return {
            'model_a': model_a,
            'model_b': model_b,
            'overall_a': float(avg_a),
            'overall_b': float(avg_b),
            'overall_winner': model_a if avg_a > avg_b else model_b,
            'metric_comparison': metric_comparison,
        }
    
    def save_snapshot(self):
        """Save current leaderboard snapshot."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'entries': self.get_leaderboard(),
        }
        self.history.append(snapshot)
        
        # Save to file
        path = os.path.join(self.output_dir, 'leaderboard_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def generate_html_report(self) -> str:
        """Generate HTML leaderboard report."""
        entries = self.get_leaderboard()
        
        rows_html = ""
        for e in entries:
            metrics_html = "".join([
                f'<td>{v:.3f}</td>' for v in e['metrics'].values()
            ])
            
            rank_class = 'gold' if e['rank'] == 1 else 'silver' if e['rank'] == 2 else 'bronze' if e['rank'] == 3 else ''
            
            rows_html += f"""
            <tr class="{rank_class}">
                <td>{e['rank']}</td>
                <td><strong>{e['model']}</strong></td>
                <td>{e['prompt']}</td>
                <td class="score">{e['overall']:.3f}</td>
                {metrics_html}
                <td>{e['n_evals']}</td>
                <td>{e['latency']:.0f}ms</td>
            </tr>
            """
        
        # Get all metric names
        all_metrics = set()
        for e in entries:
            all_metrics.update(e['metrics'].keys())
        metric_headers = "".join([f'<th>{m}</th>' for m in sorted(all_metrics)])
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Evals Leaderboard</title>
    <style>
        :root {{
            --bg: #0f172a; --card: #1e293b; --text: #e2e8f0;
            --accent: #6366f1; --gold: #fbbf24; --silver: #94a3b8; --bronze: #cd7f32;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }}
        h1 {{ text-align: center; margin-bottom: 2rem; color: var(--accent); }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; }}
        th {{ background: var(--accent); padding: 1rem; text-align: left; font-weight: 600; }}
        td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #334155; }}
        tr:hover {{ background: rgba(99, 102, 241, 0.1); }}
        .gold td:first-child {{ border-left: 4px solid var(--gold); }}
        .silver td:first-child {{ border-left: 4px solid var(--silver); }}
        .bronze td:first-child {{ border-left: 4px solid var(--bronze); }}
        .score {{ font-weight: bold; color: var(--accent); font-size: 1.1em; }}
        .footer {{ text-align: center; margin-top: 2rem; color: #64748b; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Evals Leaderboard</h1>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Prompt Variant</th>
                    <th>Overall</th>
                    {metric_headers}
                    <th>Samples</th>
                    <th>Latency</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <div class="footer">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""
        
        path = os.path.join(self.output_dir, 'leaderboard.html')
        with open(path, 'w') as f:
            f.write(html)
        
        return path
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate leaderboard summary statistics."""
        if not self.entries:
            return {}
        
        scores = [e.overall_score for e in self.entries]
        
        models = {}
        for e in self.entries:
            if e.model_name not in models:
                models[e.model_name] = []
            models[e.model_name].append(e.overall_score)
        
        prompts = {}
        for e in self.entries:
            if e.prompt_variant not in prompts:
                prompts[e.prompt_variant] = []
            prompts[e.prompt_variant].append(e.overall_score)
        
        return {
            'total_entries': len(self.entries),
            'total_models': len(models),
            'total_prompts': len(prompts),
            'best_model': max(models.keys(), key=lambda k: np.mean(models[k])),
            'best_prompt': max(prompts.keys(), key=lambda k: np.mean(prompts[k])),
            'score_distribution': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
            },
        }
