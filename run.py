import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation Framework')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Run evaluation')
    eval_parser.add_argument('dataset', help='Dataset name')
    eval_parser.add_argument('--model', '-m', required=True, help='Model function path')
    eval_parser.add_argument('--output', '-o', default='output', help='Output directory')
    eval_parser.add_argument('--n-samples', type=int, help='Number of samples')

    # RAGAS command
    ragas_parser = subparsers.add_parser('ragas', help='Run RAGAS evaluation')
    ragas_parser.add_argument('dataset', help='Dataset name')
    ragas_parser.add_argument('--rag-function', required=True, help='RAG function path')
    ragas_parser.add_argument('--output', '-o', default='output', help='Output directory')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model versions')
    compare_parser.add_argument('baseline', help='Baseline scores JSON')
    compare_parser.add_argument('current', help='Current scores JSON')
    compare_parser.add_argument('--output', '-o', default='output', help='Output directory')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate dashboard')
    dashboard_parser.add_argument('--entries', required=True, help='Entries JSON')
    dashboard_parser.add_argument('--output', '-o', default='output', help='Output directory')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.command == 'eval':
        run_eval(args)
    elif args.command == 'ragas':
        run_ragas(args)
    elif args.command == 'compare':
        run_compare(args)
    elif args.command == 'dashboard':
        run_dashboard(args)
    else:
        parser.print_help()

def run_eval(args):
    """Run evaluation."""
    from src.pipeline import EvalPipeline

    def dummy_model(prompt):
        return f"Response to: {prompt}"

    pipeline = EvalPipeline(output_dir=args.output)

    pipeline.task_manager.create_dataset(args.dataset, [
        {'input': 'What is ML?', 'expected': 'Machine learning is...'},
        {'input': 'Explain AI', 'expected': 'Artificial intelligence is...'},
    ])

    results = pipeline.run_full_evaluation(
        dataset_name=args.dataset,
        models={'demo_model': dummy_model},
    )

    print(f"\nEvaluation complete!")
    print(f"Results: {json.dumps(results.get('summary', {}), indent=2)}")

def run_ragas(args):
    """Run RAGAS evaluation."""
    from src.pipeline import EvalPipeline

    def dummy_rag(query):
        return {'answer': f'Answer to {query}', 'context': ['context1', 'context2']}

    pipeline = EvalPipeline(output_dir=args.output)

    pipeline.task_manager.create_dataset(args.dataset, [
        {'input': 'What is RAG?', 'expected': 'Retrieval augmented generation...'},
    ])

    results = pipeline.run_ragas_evaluation(args.dataset, dummy_rag)

    print(f"\nRAGAS evaluation complete!")
    print(f"Results: {json.dumps(results, indent=2)}")

def run_compare(args):
    """Compare model versions."""
    from src.pipeline import EvalPipeline

    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.current) as f:
        current = json.load(f)

    pipeline = EvalPipeline(output_dir=args.output)
    results = pipeline.run_regression_check(baseline, current)

    print(f"\nRegression check complete!")
    print(f"Status: {results['comparison']['overall_status']}")

def run_dashboard(args):
    """Generate dashboard."""
    from src.dashboard import LeaderboardDashboard

    with open(args.entries) as f:
        entries = json.load(f)

    dashboard = LeaderboardDashboard(output_dir=args.output)

    for entry in entries:
        dashboard.add_entry(
            model_name=entry['model'],
            prompt_variant=entry['prompt'],
            scores=entry['scores'],
            n_evaluations=entry.get('n_evals', 1),
        )

    path = dashboard.generate_html_report()
    print(f"\nDashboard generated: {path}")

if __name__ == '__main__':
    main()
