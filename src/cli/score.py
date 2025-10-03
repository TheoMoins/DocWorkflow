from pathlib import Path
import pandas as pd

from rich.console import Console

from src.tasks.base_tasks import BaseTask

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
    if not isinstance(results, pd.DataFrame):
        results_df = pd.DataFrame([results])
    else:
        results_df = results
        
    output_file = dir / "results.csv"  # ou un autre nom

    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")

    return True

