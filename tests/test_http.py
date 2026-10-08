import time

import pytest

from filing_signals import http


class FakeResponse:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    def raise_for_status(self):
        raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = 0

    def get(self, url, timeout):
        self.calls += 1
        return self.responses.pop(0)


def test_user_agent_must_declare_a_contact(monkeypatch):
    monkeypatch.setenv("EDGAR_USER_AGENT", "my project")
    with pytest.raises(http.ConfigError):
        http.user_agent_from_env()
    monkeypatch.setenv("EDGAR_USER_AGENT", "my project me@example.com")
    assert http.user_agent_from_env() == "my project me@example.com"


def test_rate_limiter_spaces_requests():
    limiter = http.RateLimiter(rate=50)
    start = time.monotonic()
    for _ in range(11):
        limiter.wait()
    assert time.monotonic() - start >= 10 / 50 * 0.95


def test_retries_transient_errors(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    session = FakeSession([FakeResponse(503), FakeResponse(429, headers={"Retry-After": "1"}),
                           FakeResponse(200, b"ok")])
    client = http.EdgarClient("ua a@b.c", rate=1000, session=session)
    assert client.get("u") == b"ok"
    assert session.calls == 3 and client.stats.retries == 2


def test_404_and_403_are_distinct_failures():
    client = http.EdgarClient("ua a@b.c", rate=1000, session=FakeSession([FakeResponse(404)]))
    with pytest.raises(http.NotFound):
        client.get("u")
    client = http.EdgarClient("ua a@b.c", rate=1000, session=FakeSession([FakeResponse(403)]))
    with pytest.raises(http.ConfigError):
        client.get("u")
