from blacksmoke.llm import errors
from tests.conftest import RAW_FILE, TAGGED_FILE


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_list_tasks_exposes_routes(client):
    tasks = {t["name"]: t for t in client.get("/v1/tasks").json()}
    assert len(tasks) == 7
    assert tasks["tag_reviews_with_topics"]["kind"] == "per_review"
    assert tasks["recommend_actions"]["data_format"] == "tagged"
    assert tasks["find_bugs_and_features"]["route"][0] == {
        "provider": "openai",
        "model": "gpt-4o-mini",
    }


def test_aggregate_task_returns_envelope(client, providers):
    providers["openai"].default = {"bugs": ["crash on upload"], "feature_requests": []}
    response = client.post("/v1/tasks/find_bugs_and_features", json={"file_path": RAW_FILE})
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "find_bugs_and_features"
    assert body["output"] == {"bugs": ["crash on upload"], "feature_requests": []}
    assert body["served_by"] == {"provider": "openai", "model": "gpt-4o-mini"}
    assert body["num_of_reviews"] == 4
    assert body["usage"]["cost_usd"] > 0
    assert [a["outcome"] for a in body["attempts"]] == ["success"]


def test_fallback_is_visible_in_response(client, providers):
    providers["openai"].default = errors.AuthError("invalid key")
    providers["google"].default = {"paragraphs": ["one", "two"]}
    body = client.post("/v1/tasks/create_reviews_summary", json={"file_path": RAW_FILE}).json()
    assert body["served_by"]["provider"] == "google"
    assert [a["outcome"] for a in body["attempts"]] == ["auth_error", "success"]


def test_tagging_output_feeds_topic_scoped_tasks(client, providers):
    providers["openai"].default = {
        "positive_topics": [],
        "negative_topics": ["frequent_app_crashes"],
    }
    report = client.post(
        "/v1/tasks/tag_reviews_with_topics",
        json={"file_path": RAW_FILE, "negative_topics": ["frequent_app_crashes"]},
    ).json()
    assert report["succeeded"] == 4 and report["output_file"].startswith("results/")

    providers["openai"].default = {"recommended_actions": ["fix crashes"]}
    response = client.post(
        "/v1/tasks/recommend_actions",
        json={"file_path": report["output_file"], "topic_name": "frequent_app_crashes"},
    )
    assert response.status_code == 200
    assert response.json()["num_of_reviews"] == 4


def test_marketing_strategies_use_positive_topic(client, providers):
    providers["openai"].default = {"marketing_strategies": ["promote low prices"]}
    response = client.post(
        "/v1/tasks/recommend_marketing_strategies",
        json={"file_path": TAGGED_FILE, "topic_name": "affordable_prices"},
    )
    assert response.status_code == 200
    assert response.json()["num_of_reviews"] == 1
    assert "affordable_prices" in providers["openai"].calls[0][0].user_prompt


def test_unknown_dataset_is_404(client):
    response = client.post("/v1/tasks/find_bugs_and_features", json={"file_path": "missing.csv"})
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "DatasetNotFoundError"


def test_path_escape_is_422(client):
    response = client.post("/v1/tasks/find_bugs_and_features", json={"file_path": "../secret.csv"})
    assert response.status_code == 422


def test_topic_with_no_reviews_is_422(client):
    response = client.post(
        "/v1/tasks/extract_subtopics",
        json={"file_path": TAGGED_FILE, "topic_polarity": "positive", "topic_name": "nonexistent"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "NoReviewsError"


def test_legacy_fields_are_rejected(client):
    response = client.post(
        "/v1/tasks/find_bugs_and_features",
        json={"file_path": RAW_FILE, "provider_name": "openai", "llm_model_name": "gpt-4o"},
    )
    assert response.status_code == 422


def test_all_targets_failing_is_502_with_attempts(client, providers):
    providers["openai"].default = errors.ProviderUnavailableError("down")
    providers["google"].default = errors.AuthError("bad key")
    response = client.post("/v1/tasks/find_bugs_and_features", json={"file_path": RAW_FILE})
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "AllTargetsFailedError"
    assert [a["outcome"] for a in error["attempts"]] == ["unavailable", "auth_error"]


def test_providers_are_closed_on_shutdown(container, providers):
    from fastapi.testclient import TestClient

    from blacksmoke.api.app import create_app

    providers["openai"].default = {"bugs": [], "feature_requests": []}
    with TestClient(create_app(container)) as client:
        client.post("/v1/tasks/find_bugs_and_features", json={"file_path": RAW_FILE})
    assert providers["openai"].closed
