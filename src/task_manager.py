import json
import os
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field

@dataclass
class EvalTask:
    """Single evaluation task."""
    task_id: str
    dataset_name: str
    input_text: str
    expected_output: str = None
    context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: str = None
    difficulty: str = None
    tags: List[str] = field(default_factory=list)
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.task_id is None:
            self.task_id = self._generate_id()
    
    def _generate_id(self) -> str:
        content = f"{self.dataset_name}:{self.input_text}:{self.expected_output}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class EvalResult:
    """Evaluation result for a single task."""
    task_id: str
    model_name: str
    prompt_variant: str = 'default'
    model_output: str = None
    scores: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0
    token_usage: Dict[str, int] = field(default_factory=dict)
    timestamp: str = None
    error: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class TaskManager:
    """Manage evaluation task datasets."""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.datasets: Dict[str, List[EvalTask]] = {}
    
    def load_dataset(self, name: str, file_path: str = None) -> List[EvalTask]:
        """Load a dataset from file or memory."""
        if file_path and os.path.exists(file_path):
            return self._load_from_file(name, file_path)
        
        path = os.path.join(self.data_dir, f'{name}.json')
        if os.path.exists(path):
            return self._load_from_file(name, path)
        
        self.logger.warning(f"Dataset '{name}' not found")
        return []
    
    def _load_from_file(self, name: str, file_path: str) -> List[EvalTask]:
        """Load tasks from JSON file."""
        with open(file_path) as f:
            data = json.load(f)
        
        tasks = []
        for item in data:
            task = EvalTask(
                task_id=item.get('task_id'),
                dataset_name=name,
                input_text=item['input'],
                expected_output=item.get('expected', item.get('output')),
                context=item.get('context', []),
                metadata=item.get('metadata', {}),
                category=item.get('category'),
                difficulty=item.get('difficulty'),
                tags=item.get('tags', []),
            )
            tasks.append(task)
        
        self.datasets[name] = tasks
        self.logger.info(f"Loaded {len(tasks)} tasks from '{name}'")
        return tasks
    
    def save_dataset(self, name: str, tasks: List[EvalTask]):
        """Save dataset to JSON file."""
        path = os.path.join(self.data_dir, f'{name}.json')
        data = [asdict(t) for t in tasks]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.datasets[name] = tasks
        self.logger.info(f"Saved {len(tasks)} tasks to '{name}'")
    
    def create_dataset(self, name: str, tasks_data: List[Dict]) -> List[EvalTask]:
        """Create new dataset from list of dicts."""
        tasks = []
        for item in tasks_data:
            task = EvalTask(
                task_id=None,
                dataset_name=name,
                input_text=item['input'],
                expected_output=item.get('expected', item.get('output')),
                context=item.get('context', []),
                metadata=item.get('metadata', {}),
                category=item.get('category'),
                difficulty=item.get('difficulty'),
                tags=item.get('tags', []),
            )
            tasks.append(task)
        
        self.save_dataset(name, tasks)
        return tasks
    
    def filter_tasks(self, dataset_name: str,
                    category: str = None,
                    difficulty: str = None,
                    tags: List[str] = None,
                    limit: int = None) -> List[EvalTask]:
        """Filter tasks by criteria."""
        tasks = self.datasets.get(dataset_name, [])
        
        if category:
            tasks = [t for t in tasks if t.category == category]
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]
        if tags:
            tasks = [t for t in tasks if any(tag in t.tags for tag in tags)]
        if limit:
            tasks = tasks[:limit]
        
        return tasks
    
    def split_dataset(self, dataset_name: str,
                     train_pct: float = 0.7,
                     val_pct: float = 0.15,
                     test_pct: float = 0.15,
                     seed: int = 42) -> Dict[str, List[EvalTask]]:
        """Split dataset into train/val/test."""
        import random
        
        tasks = self.datasets.get(dataset_name, [])
        random.seed(seed)
        shuffled = tasks.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        splits = {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:],
        }
        
        for split_name, split_tasks in splits.items():
            self.datasets[f'{dataset_name}_{split_name}'] = split_tasks
        
        return splits
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistics about a dataset."""
        tasks = self.datasets.get(dataset_name, [])
        
        if not tasks:
            return {'name': dataset_name, 'n_tasks': 0}
        
        categories = {}
        difficulties = {}
        all_tags = set()
        
        for task in tasks:
            cat = task.category or 'uncategorized'
            categories[cat] = categories.get(cat, 0) + 1
            
            diff = task.difficulty or 'unknown'
            difficulties[diff] = difficulties.get(diff, 0) + 1
            
            all_tags.update(task.tags)
        
        avg_input_len = sum(len(t.input_text) for t in tasks) / len(tasks)
        avg_output_len = sum(len(t.expected_output or '') for t in tasks) / len(tasks)
        
        return {
            'name': dataset_name,
            'n_tasks': len(tasks),
            'categories': categories,
            'difficulties': difficulties,
            'unique_tags': len(all_tags),
            'avg_input_length': avg_input_len,
            'avg_output_length': avg_output_len,
        }
    
    def sample_tasks(self, dataset_name: str, n: int,
                    stratify_by: str = None, seed: int = 42) -> List[EvalTask]:
        """Sample tasks, optionally stratified."""
        import random
        random.seed(seed)
        
        tasks = self.datasets.get(dataset_name, [])
        
        if stratify_by == 'category':
            groups = {}
            for t in tasks:
                cat = t.category or 'default'
                groups.setdefault(cat, []).append(t)
            
            sampled = []
            for cat_tasks in groups.values():
                n_sample = min(n // len(groups), len(cat_tasks))
                sampled.extend(random.sample(cat_tasks, n_sample))
            return sampled
        
        return random.sample(tasks, min(n, len(tasks)))
