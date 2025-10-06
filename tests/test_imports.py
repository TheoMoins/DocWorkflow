"""Tests pour vérifier que tous les imports nécessaires fonctionnent."""
import pytest
import sys


def test_core_imports():
    """Teste les imports de base Python."""
    try:
        import os
        import glob
        import shutil
        from pathlib import Path
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_scientific_imports():
    """Teste les imports scientifiques essentiels."""
    try:
        import numpy as np
        from PIL import Image
        assert np.__version__
        assert Image.__version__
    except ImportError as e:
        pytest.fail(f"Scientific import failed: {e}")


def test_xml_imports():
    """Teste les imports pour le traitement XML."""
    try:
        from lxml import etree as ET
        assert True
    except ImportError as e:
        pytest.fail(f"XML import failed: {e}")


def test_yolo_imports():
    """Teste les imports YOLO (pour layout)."""
    try:
        from ultralytics import YOLO, settings
        assert True
    except ImportError as e:
        pytest.fail(f"YOLO import failed: {e}")


def test_kraken_imports():
    """Teste les imports Kraken (pour line et HTR)."""
    try:
        from kraken.lib.vgsl import TorchVGSLModel
        from kraken.lib.xml import XMLPage
        from kraken import rpred
        from kraken.lib.models import load_any
        assert True
    except ImportError as e:
        pytest.fail(f"Kraken import failed: {e}")


def test_yaltai_imports():
    """Teste les imports YALTAi."""
    try:
        from yaltai.models.krakn import segment
        assert True
    except ImportError as e:
        pytest.fail(f"YALTAi import failed: {e}")


def test_metrics_imports():
    """Teste les imports pour les métriques."""
    try:
        from mean_average_precision import MetricBuilder
        import jiwer
        from jiwer import cer, wer
        assert True
    except ImportError as e:
        pytest.fail(f"Metrics import failed: {e}")


def test_visualization_imports():
    """Teste les imports pour la visualisation."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        assert True
    except ImportError as e:
        pytest.fail(f"Visualization import failed: {e}")


def test_cli_imports():
    """Teste les imports pour le CLI."""
    try:
        import click
        from rich.console import Console
        import yaml
        import pandas as pd
        import tabulate
        assert True
    except ImportError as e:
        pytest.fail(f"CLI import failed: {e}")


def test_torch_imports():
    """Teste les imports PyTorch."""
    try:
        import torch
        assert torch.__version__
        # Vérifie si CUDA est disponible (info, pas échec)
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
    except ImportError as e:
        pytest.fail(f"PyTorch import failed: {e}")


def test_docworkflow_imports():
    """Teste les imports du package DocWorkflow."""
    try:
        from src.tasks.base_tasks import BaseTask
        from src.tasks.layout.yolo_layout import YoloLayoutTask
        from src.tasks.line.kraken_line import KrakenLineTask
        from src.tasks.htr.kraken_htr import KrakenHTRTask
        from src.alto.alto_zones import extract_zones_from_alto
        from src.alto.alto_lines import extract_lines_from_alto
        from src.alto.yolalto import create_alto_xml
        from src.cli.config import Config
        from src.utils.visualisation import DocumentVisualizer
        assert True
    except ImportError as e:
        pytest.fail(f"DocWorkflow import failed: {e}")