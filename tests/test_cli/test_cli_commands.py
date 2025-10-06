"""Tests pour les commandes CLI."""
import pytest
from click.testing import CliRunner
from src.__main__ import cli


@pytest.fixture
def runner():
    """Crée un runner CLI."""
    return CliRunner()


def test_cli_help(runner):
    """Teste la commande help."""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'predict' in result.output
    assert 'score' in result.output
    assert 'train' in result.output


def test_cli_predict_missing_config(runner):
    """Teste predict sans configuration."""
    result = runner.invoke(cli, ['predict'])
    assert result.exit_code != 0


def test_cli_with_config(runner, sample_config_file):
    """Teste les commandes avec une configuration."""
    # Test avec une config mais sans données réelles
    result = runner.invoke(cli, ['-c', str(sample_config_file), 'predict', '--help'])
    assert result.exit_code == 0