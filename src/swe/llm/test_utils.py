"""Common test utilities for LLM client testing."""

from typing import Any


class MockResponse:
    """Mock aiohttp response for testing."""

    def __init__(
        self,
        status: int = 200,
        json_data: dict[str, Any] | None = None,
        content_data: list[bytes] | None = None,
    ):
        self.status = status
        self._json_data = json_data
        self._content_data = content_data or []

    async def json(self) -> dict[str, Any]:
        if self._json_data is None:
            raise ValueError("No JSON data available")
        return self._json_data

    async def __aenter__(self) -> "MockResponse":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @property
    def content(self) -> "MockContent":
        return MockContent(self._content_data)


class MockContent:
    """Mock aiohttp content for testing streaming."""

    def __init__(self, data: list[bytes]) -> None:
        self._data = data.copy()  # Make a copy to avoid mutation issues

    def __aiter__(self) -> "MockContent":
        return self

    async def __anext__(self) -> bytes:
        if not self._data:
            raise StopAsyncIteration
        return self._data.pop(0)


class MockSession:
    """Mock aiohttp session for testing."""

    def __init__(self, response: MockResponse) -> None:
        self._response = response

    def post(self, *args: Any, **kwargs: Any) -> MockResponse:
        return self._response

    async def __aenter__(self) -> "MockSession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
