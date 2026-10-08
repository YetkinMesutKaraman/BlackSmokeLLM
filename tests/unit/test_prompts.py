import pytest

from blacksmoke.core.config import load_config
from blacksmoke.core.errors import ConfigError
from blacksmoke.tasks.base import AggregateTask, TaskContext
from blacksmoke.tasks.prompts import PromptBuilder, PromptRenderer, schema_hint
from blacksmoke.tasks.registry import context_for, load_builtin_tasks
from blacksmoke.tasks.schemas import BugsAndFeatures
from tests.conftest import CONFIG_DIR

CONFIG = load_config(CONFIG_DIR)
REGISTRY = load_builtin_tasks()
BUILDER = PromptBuilder(PromptRenderer(CONFIG.prompts_dir), "##")


@pytest.mark.parametrize("task_name", REGISTRY.names())
def test_every_task_prompt_renders_completely(task_name):
    task = REGISTRY.get(task_name)
    request = task.request_model.model_validate(task.example_request)
    context = context_for(task, CONFIG.tasks)
    if isinstance(task, AggregateTask):
        llm_request = BUILDER.aggregate(task, request, context, ["first review", "second review"])
        assert "##first review##second review##" in llm_request.user_prompt
    else:
        llm_request = BUILDER.per_review(task, request, context)("only review")
        assert "only review" in llm_request.user_prompt
    for prompt in (llm_request.system_prompt, llm_request.user_prompt):
        assert prompt and "{{" not in prompt and "{%" not in prompt


def test_tagging_prompt_lists_the_given_topics_and_constrains_the_schema():
    task = REGISTRY.get("tag_reviews_with_topics")
    request = task.request_model.model_validate(
        {
            "file_path": "x.csv",
            "positive_topics": ["fast_shipping"],
            "negative_topics": ["refund_issues"],
        }
    )
    llm_request = BUILDER.per_review(task, request, context_for(task, CONFIG.tasks))("review")
    assert "- fast_shipping" in llm_request.system_prompt
    assert "- refund_issues" in llm_request.system_prompt
    schema = llm_request.output_schema
    assert schema.model_validate({"positive_topics": ["fast_shipping"], "negative_topics": []})
    with pytest.raises(ValueError):
        schema.model_validate({"positive_topics": ["refund_issues"], "negative_topics": []})


def test_undefined_template_variable_fails_loudly(tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "system.md.j2").write_text("Topics: {{ topics }}")
    with pytest.raises(ConfigError, match="topics"):
        PromptRenderer(tmp_path).render("demo", "system", {})


def test_missing_template_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        PromptRenderer(tmp_path).render("absent", "user", {})


def test_schema_hint_matches_schema_field_names():
    assert schema_hint(BugsAndFeatures) == '{"bugs": ["string"], "feature_requests": ["string"]}'


def test_task_settings_reach_the_prompt():
    task = REGISTRY.get("extract_unsupervised_topics")
    request = task.request_model.model_validate(task.example_request)
    context = TaskContext(prompt=task.name, settings={"max_num_of_topics": 7})
    llm_request = BUILDER.aggregate(task, request, context, ["r"])
    assert "at most 7 topics" in llm_request.user_prompt
