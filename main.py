from fastapi import FastAPI

from routers import (
    create_reviews_summary,
    extract_subtopics,
    extract_unsupervised_topics,
    find_bugs_and_features,
    tag_reviews_with_topics,
    recommend_actions,
)

app = FastAPI(title="LLM Review Analyzer API")

# Include routers for endpoints
app.include_router(create_reviews_summary.router)
app.include_router(extract_unsupervised_topics.router)
app.include_router(tag_reviews_with_topics.router)
app.include_router(find_bugs_and_features.router)
app.include_router(recommend_actions.router)
app.include_router(extract_subtopics.router)


@app.get("/")
async def root():
    return {"message": "Welcome to the LLM Review Analyzer API"}
