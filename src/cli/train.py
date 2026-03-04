import numpy as np

from rich.console import Console
from pathlib import Path

from src.tasks.base_tasks import BaseTask

def train(
    task: BaseTask,
    data_path: str,
    seed: int = 42
):
    _ = np.random.default_rng(seed)

    train_path = Path(data_path)

    console = Console()

    console.rule(task.name)
    console.print(f"Training Algorithm...")
    console.print(f"  Task: {task.name}")
    console.print(f"  Data: {data_path}")
    
    task.train(train_path, seed)

    print("\nDone!")

    return True

