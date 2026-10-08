from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from blacksmoke.api.app import create_app
from blacksmoke.core.config import Config, load_config
from blacksmoke.core.container import Container, build_container
from blacksmoke.core.settings import PROJECT_ROOT, Settings
from tests.fakes import FakeProvider

CONFIG_DIR = PROJECT_ROOT / "config"
RAW_FILE = "app_reviews.csv"
TAGGED_FILE = "app_reviews_tagged.csv"

LONG = " This sentence pads the review so it passes the minimum length filter."


def _reviews() -> list[dict]:
    return [
        {
            "review_id": 1,
            "date_reviewed": "2025-01-01 10:00:00",
            "review_text": "Great prices and very easy to use." + LONG,
        },
        {
            "review_id": 2,
            "date_reviewed": "2025-01-02 10:00:00",
            "review_text": "The app crashes when I upload photos." + LONG,
        },
        {
            "review_id": 3,
            "date_reviewed": "2025-01-03 10:00:00",
            "review_text": "Refund took weeks and support ignored me." + LONG,
        },
        {"review_id": 4, "date_reviewed": "2025-01-04 10:00:00", "review_text": "Too short"},
        {
            "review_id": 5,
            "date_reviewed": "2025-01-05 10:00:00",
            "review_text": "Crashes again after the update, unusable." + LONG,
        },
    ]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    raw = pd.DataFrame(_reviews())
    raw.insert(1, "title", "")
    raw.to_csv(directory / RAW_FILE, sep=";", index=False)

    tagged = pd.DataFrame(_reviews())
    tagged["ai_positive_topics"] = [["affordable_prices"], [], [], [], []]
    tagged["ai_negative_topics"] = [
        [],
        ["frequent_app_crashes"],
        ["refund_issues"],
        [],
        ["frequent_app_crashes"],
    ]
    for column in ("ai_positive_topics", "ai_negative_topics"):
        tagged[column] = tagged[column].map(repr)
    tagged.to_csv(directory / TAGGED_FILE, sep="¤", index=False)
    return directory


@pytest.fixture
def config(data_dir: Path) -> Config:
    loaded = load_config(CONFIG_DIR)
    loaded.app = loaded.app.model_copy(
        update={"data_dir": data_dir, "output_dir": data_dir / "results"}
    )
    loaded.tasks.defaults.resilience = loaded.tasks.defaults.resilience.model_copy(
        update={"backoff_initial_s": 0.0, "backoff_max_s": 0.0}
    )
    return loaded


@pytest.fixture
def providers() -> dict[str, FakeProvider]:
    return {"openai": FakeProvider("openai"), "google": FakeProvider("google")}


@pytest.fixture
def container(config: Config, providers: dict[str, FakeProvider]) -> Container:
    builders = {name: (lambda p=provider: p) for name, provider in providers.items()}
    return build_container(Settings(_env_file=None), config=config, provider_builders=builders)


@pytest.fixture
def client(container: Container):
    with TestClient(create_app(container)) as test_client:
        yield test_client
