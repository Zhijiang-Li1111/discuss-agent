import pytest


@pytest.fixture
def sample_config_dict():
    """Minimal valid config dict matching the YAML structure."""
    return {
        "discussion": {
            "min_rounds": 2,
            "max_rounds": 5,
            "model": "claude-sonnet-4-20250514",
        },
        "agents": [
            {"name": "Agent A", "system_prompt": "You are agent A."},
            {"name": "Agent B", "system_prompt": "You are agent B."},
        ],
        "host": {
            "convergence_prompt": "Judge convergence.",
            "summary_prompt": "Summarize the discussion.",
        },
        "tools": [
            {"path": "tests.helpers.FakeToolA"},
            {"path": "tests.helpers.FakeToolB"},
        ],
        "context": {
            "research_dir": "~/research-data/",
            "published_file": "PUBLISHED.md",
            "research_days": 2,
        },
    }


@pytest.fixture
def sample_config_yaml(tmp_path, sample_config_dict):
    """Write sample config dict to a YAML file and return the path."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(sample_config_dict, allow_unicode=True))
    return str(path)
