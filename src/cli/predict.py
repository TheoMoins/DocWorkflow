from pathlib import Path
import pandas as pd

from rich.console import Console

from src.tasks.base_tasks import BaseTask

def predict(
    task: BaseTask,
    data_path: str,
    output: str,
    save_image: bool
):

    console = Console()

    data_path = Path(data_path)
    dir = Path(output)
    dir.mkdir(exist_ok=True)

    console.rule(task.name)

    _ = task.predict(data_path=data_path, output_dir=dir, save_image=save_image)

    print(f"\nResults saved to {dir}")

    return True

