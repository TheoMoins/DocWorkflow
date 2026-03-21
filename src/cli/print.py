from pathlib import Path
import pandas as pd

from rich.console import Console

from src.tasks.base_tasks import BaseTask

def visualize(
    task: BaseTask,
    task_name: str,
    pred_path: str,
    results_dir: str,
    source_data_path: str = None,
    json_format: str = False
):

    console = Console()

    pred_path = Path(pred_path)

    dir = Path(results_dir) / "viz"
    dir.mkdir(exist_ok=True)

    console.rule(task.name)
    
    img_dir = source_data_path if source_data_path else pred_path

    results = task.visualize(task_name=task_name, data_path=img_dir, xml_path=pred_path, 
                             output_dir=dir, json_format=json_format)

    print(f"\nDone!")

    return results
