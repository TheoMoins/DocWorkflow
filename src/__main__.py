from pathlib import Path

import click

from src.cli.config import Config
from src.cli.score import score
from src.cli.train import train
from src.cli.predict import predict
from src.cli.print import visualize
import shutil


@click.group
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.pass_context
def cli(ctx, config):
    ctx.obj = Config(config)


@click.command
def test_installation():
    print("Looks good!")


@cli.command("predict")
@click.option(
    "-t", "--task", 
    type=click.Choice(['layout', 'line', 'htr', 'all']), 
    default='all',
    help="Task to score (default: all available tasks)"
)
@click.option(
    "-d", "--dataset", 
    type=click.Choice(['train', 'valid', 'test']), 
    default='test',
    help="Dataset to predict (default: test set)"
)
@click.option("-o", "--output", required=False, default="", type=click.Path(), help="Save results")
@click.option("--save_image", is_flag=True, help="Save the image with the prediction.")
@click.option("--cleanup_intermediate", is_flag=True, help="Delete intermediate results in full pipeline to save space.")
@click.pass_obj
def predict_command(config: Config, task: str, dataset: str, output: str, save_image: bool, cleanup_intermediate: bool):

    data_path = config.data[dataset]    
    if output == "":
        if not config.yaml.get("output_dir"):
            base_output = "./results/"
        else:
            base_output = config.yaml.get("output_dir") + "/"
        if config.yaml.get("run_name"):
            base_output = base_output + config.yaml.get("run_name") + "/"
    else:
        base_output = output + "/"


    if task == 'all':
        # Pipeline complète
        tasks_list = config.get_tasks()
        
        if not tasks_list:
            click.echo("Prediction cannot be done on any task.", err=True)
            return
        
        click.echo(f"RUNNING FULL PIPELINE")
        click.echo(f"Tasks configured: {', '.join(tasks_list)}")
        
        # Chaîner les prédictions : sortie de chaque étape = entrée de la suivante
        current_input = data_path
        previous_output = None  # Pour garder trace du dossier précédent
        
        for idx, task_name in enumerate(tasks_list):
            task_obj = getattr(config, f"{task_name}_task")
            task_output = Path(base_output) / task_name
            task_output.mkdir(parents=True, exist_ok=True)
            
            # Prédire
            predict(
                task=task_obj, 
                data_path=current_input, 
                output=task_output,
                save_image=True
            )
            
            click.echo(f"{task_name} completed")
            
            # Nettoyer les résultats intermédiaires si demandé
            if cleanup_intermediate and previous_output is not None and idx < len(tasks_list) - 1:
                # Ne pas supprimer le dernier output et ne pas supprimer si c'est les données originales
                if previous_output != data_path:
                    click.echo(f"Cleaning up intermediate results: {previous_output}")
                    shutil.rmtree(previous_output)
            
            # La sortie de cette étape devient l'entrée de la suivante
            previous_output = current_input
            current_input = task_output
        
        click.echo(f"Full pipeline complete!")
        click.echo(f"Final output: {current_input}")

    else:
        # Tâche unique
        task_obj = getattr(config, f"{task}_task")
        if task_obj is None:
            click.echo(f"Error: Task '{task}' not configured", err=True)
            return

        task_output = Path(base_output) / task
        task_output.mkdir(parents=True, exist_ok=True)

        predict(
            task=task_obj, 
            data_path=data_path, 
            output=task_output,
            save_image=save_image
        )


@cli.command("score")
@click.option(
    "-t", "--task", 
    type=click.Choice(['layout', 'line', 'htr', 'all']), 
    default='all',
    help="Task to score (default: all available tasks)"
)
@click.option(
    "-d", "--dataset", 
    type=click.Choice(['train', 'valid', 'test']), 
    default='test',
    help="Dataset to score (default: test set)"
)
@click.option("-p", "--pred_path", required=False, default="")
@click.option("-o", "--output", required=False, default="", type=click.Path(), help="Save results")
@click.pass_obj
def score_command(config: Config, task: str, pred_path: str, dataset: str, output: str):

    gt_path = config.data[dataset]    
    if output == "":
        if not config.yaml.get("output_dir"):
            output = "./results/"
        output = config.yaml.get("output_dir") + "/"
        if config.yaml.get("run_name"):
            output = output + "/" + config.yaml.get("run_name") + "/"
        output = Path(output + task)

    if task == 'all':
        # Score toutes les tâches configurées et possibles
        scoreable_tasks = config.get_scoreable_tasks(pred_path, gt_path)
        
        if not scoreable_tasks:
            click.echo("No tasks can be scored with the provided files.", err=True)
            return
                
        for task_name in scoreable_tasks:
            click.echo(f"\n{'='*50}")
            click.echo(f"Scoring {task_name}...")
            task_obj = getattr(config, f"{task_name}_task")

            if pred_path == "":
                if task_obj.config.yaml.get("input_file"):
                    pred_path = task_obj.config.yaml.get("input_file")
                else:
                    pred_path = output
        
            score(task = task_obj, 
                pred_path=pred_path, 
                gt_path=gt_path, 
                results_dir=output)    
    else:
        # Score une tâche spécifique
        task_obj = getattr(config, f"{task}_task")
        if task_obj is None:
            click.echo(f"Error: Task '{task}' not configured", err=True)
            return
    
        if pred_path == "":
            if task_obj.config.get("input_file"):
                pred_path = task_obj.config["input_file"]
            else:
                pred_path = output
        
        output.mkdir(parents=True, exist_ok=True)

        score(task = task_obj, 
              pred_path=pred_path, 
              gt_path=gt_path, 
              results_dir=output)
        
@cli.command("train")
@click.option("-t", "--task", type=click.Choice(['layout', 'line', 'htr']), required=True)
@click.option("-s", "--seed", required=False, default=42, type=click.INT, help="Random seed for generation")
@click.pass_obj
def train_command(config: Config, task: str, seed: int):
    task_obj = getattr(config, f"{task}_task")
    if task_obj is None:
        click.echo(f"Error: Task '{task}' not configured", err=True)
        return
    train(task_obj, config.data["train"], seed)


@cli.command("print")
@click.option(
    "-t", "--task", 
    type=click.Choice(['layout', 'line', 'htr']), 
    default='htr',
    help="Task to print (default: htr task)"
)
@click.option("-p", "--pred_path", required=True)
@click.option("-o", "--output", required=False, default="", type=click.Path(), help="Save results")
@click.pass_obj
def print_command(config: Config, task: str, pred_path: str, output: str):

    if output == "":
        if not config.yaml.get("output_dir"):
            output = "./results/"
        output = config.yaml.get("output_dir") + "/"
        if config.yaml.get("run_name"):
            output = output + "/" + config.yaml.get("run_name") + "/"
        output = Path(output + task)
    
    task_obj = getattr(config, f"{task}_task")
    if task_obj is None:
        click.echo(f"Error: Task '{task}' not configured", err=True)
        return

    visualize(task = task_obj, 
              task_name = task,
              pred_path=pred_path, 
              results_dir=output)

if __name__ == "__main__":
    cli()