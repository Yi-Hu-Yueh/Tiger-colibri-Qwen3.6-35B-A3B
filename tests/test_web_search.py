from __future__ import annotations

from pathlib import Path
from urllib.request import Request

import pytest
from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.api import web_search as web_search_api
from app.main import app
from app.services import web_search

client = TestClient(app)


class FakeResponse:
    def __init__(self, payload: bytes, content_type: str, url: str = "https://example.com/page", content_length: str | None = None):
        self.payload = payload
        self.headers = {"Content-Type": content_type}
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size: int):
        return self.payload[:size]

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, response: FakeResponse):
        self.response = response

    def open(self, _request, timeout: float):
        assert timeout == web_search.FETCH_TIMEOUT_SECONDS
        return self.response


def test_search_normalizes_html_urls_and_caps_results(monkeypatch: pytest.MonkeyPatch):
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def text(self, query: str, **_kwargs):
            assert query == "current query"
            return [{"title": "Unsafe", "href": "javascript:alert(1)", "body": "ignored"}] + [
                {"title": f"<b>Result {index}</b>", "href": f"https://example.com/{index}", "body": "<em>Evidence</em>"}
                for index in range(6)
            ]

    monkeypatch.setattr(web_search, "DDGS", FakeDDGS)
    results = web_search.search_web(" current   query ")
    assert len(results) == 5
    assert results[0] == {"title": "Result 0", "url": "https://example.com/0", "snippet": "Evidence"}
    assert all(result["url"].startswith("https://") for result in results)
    assert all("<" not in result["title"] + result["snippet"] for result in results)


def test_search_result_url_is_ascii_safe_for_fetching():
    result = web_search.normalize_result(
        {"title": "Weather", "href": "https://example.com/颱風/消息?q=台中", "body": "Update"}
    )
    assert result is not None
    assert result["url"].isascii()
    assert "%E9%A2%B1%E9%A2%A8" in result["url"]


def test_search_failure_is_explicit(monkeypatch: pytest.MonkeyPatch):
    def fail(_query: str):
        raise web_search.WebSearchError("provider unavailable")

    monkeypatch.setattr(web_search_api, "search_web", fail)
    response = client.post("/api/web-search", json={"query": "latest evidence"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Web Search failed: provider unavailable"


def test_only_server_search_result_urls_may_be_fetched():
    with pytest.raises(web_search.WebFetchError, match="server-side search"):
        web_search.fetch_result_page("https://other.example/page", {"https://result.example/page"})


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/a", "http://localhost/a", "http://127.0.0.1/a", "http://[::1]/a", "http://192.168.1.1/a", "http://169.254.1.1/a"],
)
def test_private_local_and_non_http_destinations_are_rejected(url: str):
    with pytest.raises(web_search.WebFetchError):
        web_search.validate_fetch_url(url)


def test_resolved_private_address_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        web_search.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(web_search.socket.AF_INET, web_search.socket.SOCK_STREAM, 6, "", ("10.2.3.4", 443))],
    )
    with pytest.raises(web_search.WebFetchError, match="resolves to"):
        web_search.validate_fetch_url("https://internal.example/page")


def test_redirect_destination_is_revalidated():
    handler = web_search._SafeRedirectHandler()
    with pytest.raises(web_search.WebFetchError):
        handler.redirect_request(Request("https://example.com"), None, 302, "Found", {}, "http://127.0.0.1/private")


def test_fetch_rejects_binary_content_type(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(web_search, "validate_fetch_url", lambda url: url)
    url = "https://example.com/page"
    opener = FakeOpener(FakeResponse(b"%PDF", "application/pdf", url))
    with pytest.raises(web_search.WebFetchError, match="content type"):
        web_search.fetch_result_page(url, {url}, opener=opener)


def test_fetch_rejects_response_over_one_mib(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(web_search, "validate_fetch_url", lambda url: url)
    url = "https://example.com/page"
    payload = b"x" * (web_search.MAX_RESPONSE_BYTES + 1)
    opener = FakeOpener(FakeResponse(payload, "text/plain", url))
    with pytest.raises(web_search.WebFetchError, match="1 MiB"):
        web_search.fetch_result_page(url, {url}, opener=opener)


def test_html_visible_text_extraction_removes_page_chrome():
    html = b"""<html><body><nav>Navigation noise</nav><script>secret()</script><style>.x{}</style>
    <main><h1>Python release</h1><p>Python 3.14 was released with useful improvements.</p></main>
    <footer>Footer noise</footer></body></html>"""
    extracted = web_search.extract_visible_text(html)
    assert "Python release" in extracted
    assert "useful improvements" in extracted
    assert "Navigation noise" not in extracted
    assert "secret" not in extracted
    assert "Footer noise" not in extracted


def test_grounding_context_is_relevant_and_bounded(monkeypatch: pytest.MonkeyPatch):
    huge = "\n".join(f"Python release evidence paragraph {index} " + ("x" * 500) for index in range(20))
    monkeypatch.setattr(
        web_search,
        "fetch_result_page",
        lambda url, _allowed: (huge.encode(), "text/plain", url),
    )
    results = [
        {"title": f"Python {index}", "url": f"https://example.com/{index}", "snippet": "fallback"}
        for index in range(5)
    ]
    grounded = web_search.ground_search_results("Python release", results)
    assert len(grounded["results"]) == web_search.MAX_FETCHED_PAGES
    assert all(len(item["evidence"]) <= web_search.MAX_SOURCE_GROUNDING_CHARS for item in grounded["results"])
    assert grounded["grounding_character_count"] <= 6000
    assert len(grounded["context"]) < web_search.MAX_TOTAL_GROUNDING_CHARS


def test_weather_intent_and_location_detection():
    query = "台中今天的天氣如何？請直接告訴我目前氣溫、今日最高最低溫，並附來源。"
    assert web_search.is_weather_intent(query)
    assert web_search.is_weather_intent("What is the temperature in London?")
    assert web_search.is_weather_intent("What is today's high in Taichung?")
    assert not web_search.is_weather_intent("Explain Python dictionaries")
    assert web_search.extract_weather_location(query) == "台中"
    assert web_search.extract_weather_location("weather in Taipei today") == "Taipei"
    assert web_search.extract_weather_location("What is today's high in Taichung?") == "Taichung"
    assert web_search.extract_weather_locations("比較台中和高雄今天的天氣。") == ["台中", "高雄"]


def test_structured_weather_normalization():
    normalized = web_search.normalize_weather_data(
        {"results": [{"name": "Taichung", "admin1": "Taiwan", "country": "Taiwan", "latitude": 24.15, "longitude": 120.68}]},
        {
            "timezone": "Asia/Taipei",
            "current": {"time": "2026-10-01T12:00", "temperature_2m": 27.1, "apparent_temperature": 29.0, "weather_code": 2},
            "daily": {"temperature_2m_max": [30.2], "temperature_2m_min": [24.3], "precipitation_probability_max": [40], "weather_code": [2]},
        },
    )
    assert normalized["current_temperature_c"] == 27.1
    assert normalized["today_high_c"] == 30.2
    assert normalized["today_low_c"] == 24.3
    assert normalized["condition"] == "partly cloudy"


def test_general_search_api_returns_grounded_evidence(monkeypatch: pytest.MonkeyPatch):
    discovered = [{"title": "Result", "url": "https://example.com", "snippet": "snippet"}]
    monkeypatch.setattr(web_search_api, "search_web", lambda _query: discovered)
    monkeypatch.setattr(
        web_search_api,
        "ground_search_results",
        lambda _query, results: {
            "results": [{**results[0], "evidence": "direct factual evidence"}],
            "context": "direct grounded context [1]",
            "fetch_duration_seconds": 0.25,
            "grounding_character_count": 23,
        },
    )
    response = client.post("/api/web-search", json={"query": "latest Python release"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["context"] == "direct grounded context [1]"
    assert payload["results"][0]["url"] == "https://example.com"
    assert payload["fetch_duration_seconds"] == 0.25


def test_failed_page_fetch_uses_search_snippet(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        web_search,
        "fetch_result_page",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(web_search.WebFetchError("blocked")),
    )
    result = {"title": "Result", "url": "https://example.com", "snippet": "specific fallback evidence"}
    grounded = web_search.ground_search_results("specific", [result])
    assert grounded["results"][0]["evidence"] == "specific fallback evidence"
    assert "specific fallback evidence" in grounded["context"]


def test_weather_succeeds_when_ddgs_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(web_search_api, "search_web", lambda _query: (_ for _ in ()).throw(web_search.WebSearchError("offline")))
    monkeypatch.setattr(
        web_search_api,
        "lookup_current_weather",
        lambda _location: {
            "location": "台中市, 臺灣",
            "observation_time": "2026-10-01T12:00",
            "timezone": "Asia/Taipei",
            "current_temperature_c": 27.0,
            "apparent_temperature_c": 29.0,
            "condition": "partly cloudy",
            "today_high_c": 30.0,
            "today_low_c": 24.0,
            "precipitation_probability_percent": 30,
            "source": {"title": "Open-Meteo weather data", "url": "https://open-meteo.com/"},
        },
    )
    response = client.post("/api/web-search", json={"query": "台中天氣"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "WEATHER_DIRECT"
    assert payload["results"][0]["title"] == "Open-Meteo weather data"
    assert "27°C" in payload["direct_answer"]
    assert payload["context"] == ""


def _two_day_weather() -> dict:
    return {
        "location": "台中市, 臺灣",
        "observation_time": "2026-10-01T12:00",
        "timezone": "Asia/Taipei",
        "current_temperature_c": 27.0,
        "apparent_temperature_c": 29.0,
        "condition": "partly cloudy",
        "today_high_c": 30.0,
        "today_low_c": 24.0,
        "precipitation_probability_percent": 30,
        "daily_forecast": [
            {"date": "2026-10-01", "high_c": 30.0, "low_c": 24.0, "precipitation_probability_percent": 30, "condition": "partly cloudy"},
            {"date": "2026-10-02", "high_c": 27.0, "low_c": 22.0, "precipitation_probability_percent": 70, "condition": "moderate rain"},
        ],
        "hourly_forecast": [],
        "source": {"title": "Open-Meteo weather data", "url": "https://open-meteo.com/"},
    }


def test_direct_weather_routes_without_qwen_context_or_ddgs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(web_search_api, "lookup_current_weather", lambda _location: _two_day_weather())
    monkeypatch.setattr(web_search_api, "search_web", lambda _query: pytest.fail("DDGS must not run"))

    response = client.post(
        "/api/web-search",
        json={"query": "台中現在幾度？", "force_web_search": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "WEATHER_DIRECT"
    assert payload["direct_answer"] == "台中市, 臺灣，目前約 27°C。"
    assert payload["context"] == ""
    assert payload["results"][0]["title"].startswith("Open-Meteo")


def test_weather_comparison_uses_structured_context_without_ddgs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(web_search_api, "lookup_current_weather", lambda _location: _two_day_weather())
    monkeypatch.setattr(web_search_api, "search_web", lambda _query: pytest.fail("DDGS must not run"))

    response = client.post(
        "/api/web-search",
        json={"query": "台中今天和明天哪天比較適合戶外活動？", "force_web_search": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "WEATHER_QWEN"
    assert "Today:" in payload["context"]
    assert "Tomorrow:" in payload["context"]
    assert payload["direct_answer"] is None


def test_typhoon_current_information_routes_to_web_not_weather_provider(monkeypatch: pytest.MonkeyPatch):
    discovered = [{"title": "CWA update", "url": "https://example.com/cwa", "snippet": "current update"}]
    monkeypatch.setattr(web_search_api, "search_web", lambda _query: discovered)
    monkeypatch.setattr(web_search_api, "lookup_current_weather", lambda _location: pytest.fail("Open-Meteo is insufficient"))
    monkeypatch.setattr(
        web_search_api,
        "ground_search_results",
        lambda _query, results: {
            "results": results,
            "context": "current typhoon evidence [1]",
            "fetch_duration_seconds": 0.1,
            "grounding_character_count": 29,
        },
    )

    response = client.post(
        "/api/web-search",
        json={"query": "這週颱風是否會影響台中？請查最新資訊。", "force_web_search": False},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "WEB_SEARCH"


def test_weather_api_programming_question_stays_local(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(web_search_api, "search_web", lambda _query: pytest.fail("DDGS must not run"))
    monkeypatch.setattr(web_search_api, "lookup_current_weather", lambda _location: pytest.fail("Open-Meteo must not run"))

    response = client.post(
        "/api/web-search",
        json={"query": "Python 的 weather API 怎麼寫？", "force_web_search": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "LOCAL_QWEN"
    assert payload["results"] == []
    assert payload["context"] == ""


def test_normal_browser_chat_accepts_4096_without_search_injection(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    def stream(messages, max_tokens):
        observed.update(messages=messages, max_tokens=max_tokens)
        yield b'{"type":"done","content":"ok"}\n'

    monkeypatch.setattr(chat_api.chat_runtime, "stream_with_generation", lambda messages, max_tokens: (1, stream(messages, max_tokens)))
    payload = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4096}
    response = client.post("/api/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.headers["X-Generation-ID"] == "1"
    assert observed == {"messages": payload["messages"], "max_tokens": 4096}


def test_browser_chat_rejects_more_than_4096():
    response = client.post(
        "/api/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4097},
    )
    assert response.status_code == 422


def test_web_ui_defaults_and_safe_source_rendering():
    project_root = Path(__file__).resolve().parents[1]
    html = (project_root / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    javascript = (project_root / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert 'id="chat-web-search" type="checkbox"' in html
    assert "Force Web Search" in html
    assert 'value="4096" selected' in html
    assert "event.shiftKey" in javascript
    assert '$("#chat-form").requestSubmit()' in javascript
    assert 'link.textContent = source.title || url.hostname' in javascript
    assert 'link.rel = "noopener noreferrer"' in javascript
    assert 'if (requestError) $("#chat-error").textContent = requestError' in javascript
    assert "Grounding ${searchMetrics.groundingCharacters} chars" in javascript
    assert "Cannot reach the local FastAPI server" in javascript
    assert "waitForModelReady" in javascript
    assert javascript.index('fetch("/api/web-search"') < javascript.index('fetch("/api/chat/completions"')
    cancel_handler = javascript.split('$("#chat-cancel").addEventListener', 1)[1].split('$("#chat-input").addEventListener', 1)[0]
    assert cancel_handler.count('"/api/chat/cancel"') == 2
    search_cancel = cancel_handler.split('if (chatState === "SEARCHING")', 1)[1].split("} else {", 1)[0]
    generation_cancel = cancel_handler.split("} else {", 1)[1]
    assert search_cancel.index("controller.abort()") < search_cancel.index('api("/api/chat/cancel"')
    assert generation_cancel.index('api(') < generation_cancel.index("controller.abort()")
    assert "generationId === null ? undefined : {generation_id:generationId}" in generation_cancel
    assert 'response.headers.get("X-Generation-ID")' in javascript
    assert "activeChatController === requestController" in javascript
    assert javascript.index("if (!completed)") < javascript.rindex("chatHistory.push({role:\"user\", content:prompt}")
    source_renderer = javascript.split("function addChatSources", 1)[1].split('$("#chat-start")', 1)[0]
    assert ".innerHTML" not in source_renderer
    start_handler = javascript.split('$("#chat-start").addEventListener', 1)[1].split('$("#chat-stop").addEventListener', 1)[0]
    assert "chat-web-search" not in start_handler
