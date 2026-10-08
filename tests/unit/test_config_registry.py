import pytest

from blacksmoke.core.config import TaskConfig, load_config
from blacksmoke.core.errors import ConfigError, UnknownModelError
from blacksmoke.llm.pricing import PricingCatalog
from blacksmoke.llm.types import ModelRef
from blacksmoke.tasks.prompts import PromptBuilder, PromptRenderer
from blacksmoke.tasks.registry import load_builtin_tasks
from blacksmoke.tasks.schemas import ReviewTaskRequest
from tests.conftest import CONFIG_DIR, RAW_FILE


def test_shipped_config_is_valid():
    config = load_config(CONFIG_DIR)
    registry = load_builtin_tasks()
    builder = PromptBuilder(PromptRenderer(config.prompts_dir), config.app.prompt_delimiter)
    registry.validate(config.tasks, ["openai", "google"], PricingCatalog(config.models), builder)
    assert set(registry.names()) == set(config.tasks.tasks)


def test_params_merge_defaults_task_and_route_entry():
    tasks = load_config(CONFIG_DIR).tasks
    primary, fallback = tasks.route_for("create_reviews_summary")
    assert (primary.params.temperature, primary.params.top_p) == (0.2, 0.95)
    assert (fallback.params.temperature, fallback.params.top_p) == (0.1, None)
    assert primary.params.max_output_tokens == fallback.params.max_output_tokens == 500
    assert primary.params.timeout_s == 60


def test_invalid_resilience_override_is_a_config_error():
    tasks = load_config(CONFIG_DIR).tasks
    tasks.tasks["find_bugs_and_features"].resilience = {"transient_attempts": 0}
    with pytest.raises(ConfigError):
        tasks.resilience_for("find_bugs_and_features")


def test_validation_reports_yaml_registry_and_provider_mismatches():
    config = load_config(CONFIG_DIR)
    registry = load_builtin_tasks()
    config.tasks.tasks.pop("find_bugs_and_features")
    config.tasks.tasks["ghost_task"] = TaskConfig(route=[{"provider": "openai", "model": "gpt-4o"}])
    builder = PromptBuilder(PromptRenderer(config.prompts_dir), "##")
    with pytest.raises(ConfigError) as info:
        registry.validate(config.tasks, ["openai"], PricingCatalog(config.models), builder)
    message = str(info.value)
    assert "'find_bugs_and_features' is registered but missing" in message
    assert "'ghost_task' is in tasks.yaml but not registered" in message
    assert "unknown provider 'google'" in message


def test_model_override_becomes_primary_and_keeps_other_fallbacks(container):
    service = container.task_service
    request = ReviewTaskRequest(
        file_path=RAW_FILE, model_override=ModelRef(provider="openai", model="gpt-4o")
    )
    route = service.resolve_route("find_bugs_and_features", request)
    assert [str(t.ref) for t in route] == [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-2.5-flash",
    ]
    assert route[0].params.temperature == 0.2


def test_model_override_matching_a_fallback_is_not_tried_twice(container):
    request = ReviewTaskRequest(
        file_path=RAW_FILE, model_override=ModelRef(provider="google", model="gemini-2.5-flash")
    )
    route = container.task_service.resolve_route("find_bugs_and_features", request)
    assert [str(t.ref) for t in route] == ["google/gemini-2.5-flash", "openai/gpt-4o-mini"]
    assert route[0].params.provider_options == {"thinking_budget": 0}


def test_allow_fallback_false_keeps_only_primary(container):
    request = ReviewTaskRequest(file_path=RAW_FILE, allow_fallback=False)
    route = container.task_service.resolve_route("find_bugs_and_features", request)
    assert [str(t.ref) for t in route] == ["openai/gpt-4o-mini"]


def test_override_with_unknown_model_is_rejected(container):
    request = ReviewTaskRequest(
        file_path=RAW_FILE, model_override=ModelRef(provider="openai", model="gpt-unknown")
    )
    with pytest.raises(UnknownModelError):
        container.task_service.resolve_route("find_bugs_and_features", request)
