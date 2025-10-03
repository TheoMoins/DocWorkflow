from pathlib import Path
import pandas as pd

from rich.console import Console

from src.tasks.base_tasks import BaseTask

def visualize(
    task: BaseTask,
    task_name: str,
    pred_path: str,
    results_dir: str
):

    console = Console()

    pred_path = Path(pred_path)

    dir = Path(results_dir / "viz")
    dir.mkdir(exist_ok=True)

    console.rule(task.name)

    results = task.visualize(task_name=task_name, data_path=pred_path, output_dir=dir)

    print(f"\nDone!")

    return results

