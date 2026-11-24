from pathlib import Path
import pandas as pd

from rich.console import Console

from src.tasks.base_tasks import BaseTask
from src.utils.memory_monitor import MemoryLimitExceeded

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

    if save_image is None:
        save_image = task.config.get('save_image', True)

    try:
        _ = task.predict(data_path=data_path, output_dir=dir, save_image=save_image)
    except MemoryLimitExceeded as e:
        console.print(f"[bold red]Stopped to protect the machine: {e}[/bold red]")
        console.print("Already processed files are kept; relaunch the same command to resume.")
        raise SystemExit(1)

    print(f"\nResults saved to {dir}")

    return True

