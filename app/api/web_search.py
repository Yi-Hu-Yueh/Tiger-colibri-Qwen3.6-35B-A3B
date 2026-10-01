from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.web_search import (
    MAX_QUERY_LENGTH,
    ROUTE_LOCAL_QWEN,
    ROUTE_WEATHER_DIRECT,
    ROUTE_WEATHER_QWEN,
    ROUTE_WEB_SEARCH,
    WebSearchError,
    build_weather_context,
    classify_query_route,
    extract_weather_locations,
    format_direct_weather_answer,
    ground_search_results,
    lookup_current_weather,
    search_web,
)

router = APIRouter(prefix="/api", tags=["web-search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    force_web_search: bool | None = None


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    route: str
    results: list[SearchResult]
    context: str
    direct_answer: str | None = None
    search_duration_seconds: float
    fetch_duration_seconds: float
    grounding_character_count: int


@router.post("/web-search", response_model=SearchResponse)
def web_search(request: SearchRequest) -> SearchResponse:
    query = request.query.strip()
    route = classify_query_route(query, force_web_search=bool(request.force_web_search))
    if request.force_web_search is None and route == ROUTE_LOCAL_QWEN:
        route = ROUTE_WEB_SEARCH
    if route == ROUTE_LOCAL_QWEN:
        return SearchResponse(
            query=query,
            route=route,
            results=[],
            context="",
            search_duration_seconds=0.0,
            fetch_duration_seconds=0.0,
            grounding_character_count=0,
        )

    if route in {ROUTE_WEATHER_DIRECT, ROUTE_WEATHER_QWEN}:
        locations = extract_weather_locations(query)
        if not locations:
            return SearchResponse(
                query=query,
                route=ROUTE_WEATHER_DIRECT,
                results=[],
                context="",
                direct_answer="請提供要查詢天氣的地點。" if any("\u3400" <= char <= "\u9fff" for char in query) else "Which location should I check?",
                search_duration_seconds=0.0,
                fetch_duration_seconds=0.0,
                grounding_character_count=0,
            )
        fetch_started = perf_counter()
        try:
            weather_reports = [lookup_current_weather(location) for location in locations]
        except WebSearchError as error:
            raise HTTPException(status_code=502, detail=f"Weather lookup failed: {error}") from error
        weather = weather_reports[0]
        source = weather["source"]
        summary = format_direct_weather_answer(query, weather)
        if route == ROUTE_WEATHER_DIRECT:
            return SearchResponse(
                query=query,
                route=route,
                results=[SearchResult(title=source["title"], url=source["url"], snippet=summary)],
                context="",
                direct_answer=summary,
                search_duration_seconds=0.0,
                fetch_duration_seconds=perf_counter() - fetch_started,
                grounding_character_count=0,
            )
        contexts: list[str] = []
        grounding_characters = 0
        for report in weather_reports:
            report_context, report_characters = build_weather_context(query, report)
            contexts.append(report_context)
            grounding_characters += report_characters
        context = "\n\n--- Additional location ---\n\n".join(contexts)
        return SearchResponse(
            query=query,
            route=route,
            results=[SearchResult(title=source["title"], url=source["url"], snippet=summary)],
            context=context,
            search_duration_seconds=0.0,
            fetch_duration_seconds=perf_counter() - fetch_started,
            grounding_character_count=grounding_characters,
        )

    if route != ROUTE_WEB_SEARCH:
        raise HTTPException(status_code=500, detail=f"Unsupported query route: {route}")
    search_started = perf_counter()
    try:
        results = search_web(query)
    except WebSearchError as error:
        raise HTTPException(status_code=502, detail=f"Web Search failed: {error}") from error
    search_duration = perf_counter() - search_started
    try:
        grounded = ground_search_results(query, results)
    except WebSearchError as error:
        raise HTTPException(status_code=502, detail=f"Web Search failed: {error}") from error
    return SearchResponse(
        query=query,
        route=route,
        results=grounded["results"],
        context=grounded["context"],
        search_duration_seconds=search_duration,
        fetch_duration_seconds=grounded["fetch_duration_seconds"],
        grounding_character_count=grounded["grounding_character_count"],
    )
