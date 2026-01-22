from pathlib import Path
import pandas as pd

from rich.console import Console

from src.tasks.base_tasks import BaseTask
from src.utils.metrics import save_score_csvs

def score(
    task: BaseTask,
    pred_path: str,
    gt_path: str,
    results_dir: str
):

    console = Console()

    pred_path = Path(pred_path)

    dir = Path(results_dir)
    dir.mkdir(exist_ok=True)

    console.rule(task.name)

    results = task.score(pred_path=pred_path, gt_path=gt_path)

    # Save main metrics CSV
    results_df = pd.DataFrame([results['metrics']])
    output_file = dir / "results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Save detailed CSVs
    save_score_csvs(
        results_dir=dir,
        page_scores=results['page_scores'],
        document_scores=results.get('document_scores'),
        structure_type=results['structure_type']
    )

    return True

