import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable


def _make_tool_call(tool_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _make_llm_handler(
    responder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        state: dict[str, Any] = {"calls": 0}

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.__class__.state["calls"] += 1
            body = responder(payload, self.__class__.state)

            encoded = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    return _Handler


def _run_agent_with_servers(
    question: str,
    responder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    *,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    handler = _make_llm_handler(responder)
    llm_server = HTTPServer(("127.0.0.1", 0), handler)
    llm_thread = threading.Thread(target=llm_server.serve_forever, daemon=True)
    llm_thread.start()

    env = os.environ.copy()
    env["LLM_API_KEY"] = "test-key"
    env["LLM_API_BASE"] = f"http://127.0.0.1:{llm_server.server_port}/v1"
    env["LLM_MODEL"] = "qwen3-coder-flash"
    if extra_env:
        env.update(extra_env)

    try:
        result = subprocess.run(
            [sys.executable, "agent.py", question],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )
    finally:
        llm_server.shutdown()
        llm_server.server_close()
        llm_thread.join(timeout=1)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_system_agent_reads_source_code_for_framework_question() -> None:
    def responder(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        messages = payload["messages"]

        if state["calls"] == 1:
            tool_names = {tool["function"]["name"] for tool in payload["tools"]}
            assert {"read_file", "list_files", "query_api"} <= tool_names
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                _make_tool_call(
                                    "call-1",
                                    "read_file",
                                    {"path": "backend/app/main.py"},
                                )
                            ],
                        }
                    }
                ]
            }

        assert any(
            message["role"] == "tool" and "FastAPI" in message["content"]
            for message in messages
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "answer": "The backend uses FastAPI.",
                                "source": "backend/app/main.py",
                            }
                        ),
                    }
                }
            ]
        }

    data = _run_agent_with_servers(
        "What Python web framework does this project's backend use?",
        responder,
        extra_env={"LMS_API_KEY": "test-lms-key"},
    )

    assert data["answer"] == "The backend uses FastAPI."
    assert data["source"] == "backend/app/main.py"
    assert [tool_call["tool"] for tool_call in data["tool_calls"]] == ["read_file"]


def test_system_agent_queries_api_with_backend_bearer_token() -> None:
    api_state: dict[str, Any] = {"auth_header": None}

    class _ApiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            api_state["auth_header"] = self.headers.get("Authorization")
            body = [
                {"id": 1, "title": "Lab 01"},
                {"id": 2, "title": "Task 01"},
                {"id": 3, "title": "Task 02"},
            ]
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    api_server = HTTPServer(("127.0.0.1", 0), _ApiHandler)
    api_thread = threading.Thread(target=api_server.serve_forever, daemon=True)
    api_thread.start()

    def responder(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        messages = payload["messages"]

        if state["calls"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                _make_tool_call(
                                    "call-1",
                                    "query_api",
                                    {"method": "GET", "path": "/items/"},
                                )
                            ],
                        }
                    }
                ]
            }

        assert any(
            message["role"] == "tool"
            and '"status_code": 200' in message["content"]
            and '"body_length": 3' in message["content"]
            for message in messages
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"answer": "There are 3 items in the database."}
                        ),
                    }
                }
            ]
        }

    try:
        data = _run_agent_with_servers(
            "How many items are currently stored in the database?",
            responder,
            extra_env={
                "LMS_API_KEY": "test-lms-key",
                "AGENT_API_BASE_URL": f"http://127.0.0.1:{api_server.server_port}",
            },
        )
    finally:
        api_server.shutdown()
        api_server.server_close()
        api_thread.join(timeout=1)

    assert data["answer"] == "There are 3 items in the database."
    assert [tool_call["tool"] for tool_call in data["tool_calls"]] == ["query_api"]
    assert api_state["auth_header"] == "Bearer test-lms-key"


def test_system_agent_can_probe_missing_auth_behavior() -> None:
    api_state: dict[str, Any] = {"auth_header": "unexpected"}

    class _ApiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            api_state["auth_header"] = self.headers.get("Authorization")
            encoded = json.dumps({"detail": "Not authenticated"}).encode("utf-8")
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    api_server = HTTPServer(("127.0.0.1", 0), _ApiHandler)
    api_thread = threading.Thread(target=api_server.serve_forever, daemon=True)
    api_thread.start()

    def responder(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        messages = payload["messages"]

        if state["calls"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                _make_tool_call(
                                    "call-1",
                                    "query_api",
                                    {"method": "GET", "path": "/items/"},
                                )
                            ],
                        }
                    }
                ]
            }

        assert any(
            message["role"] == "tool" and '"status_code": 403' in message["content"]
            for message in messages
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"answer": "Without the authentication header, the API returns 403."}
                        ),
                    }
                }
            ]
        }

    try:
        data = _run_agent_with_servers(
            "What HTTP status code does the API return when you request /items/ without an authentication header?",
            responder,
            extra_env={
                "LMS_API_KEY": "test-lms-key",
                "AGENT_API_BASE_URL": f"http://127.0.0.1:{api_server.server_port}",
            },
        )
    finally:
        api_server.shutdown()
        api_server.server_close()
        api_thread.join(timeout=1)

    assert data["answer"] == "Without the authentication header, the API returns 403."
    assert [tool_call["tool"] for tool_call in data["tool_calls"]] == ["query_api"]
    assert api_state["auth_header"] is None
