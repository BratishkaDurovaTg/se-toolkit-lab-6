import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable


def _make_tool_call(tool_id: str, name: str, path: str) -> dict[str, Any]:
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps({"path": path}),
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


def _run_agent_with_fake_llm(
    question: str,
    responder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    handler = _make_tool_llm_handler = _make_llm_handler(responder)
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    env = os.environ.copy()
    env["LLM_API_KEY"] = "test-key"
    env["LLM_API_BASE"] = f"http://127.0.0.1:{server.server_port}/v1"
    env["LLM_MODEL"] = "qwen3-coder-flash"

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
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_documentation_agent_reads_wiki_file_for_merge_conflict() -> None:
    def responder(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        messages = payload["messages"]

        if state["calls"] == 1:
            assert any(tool["function"]["name"] == "read_file" for tool in payload["tools"])
            assert any(tool["function"]["name"] == "list_files" for tool in payload["tools"])
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [_make_tool_call("call-1", "list_files", "wiki")],
                        }
                    }
                ]
            }

        if state["calls"] == 2:
            assert any(message["role"] == "tool" for message in messages)
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [_make_tool_call("call-2", "read_file", "wiki/git-workflow.md")],
                        }
                    }
                ]
            }

        assert any(
            message["role"] == "tool"
            and ("conflict" in message["content"].lower() or "merge" in message["content"].lower())
            for message in messages
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "answer": "Edit the conflicting file, choose the correct changes, then stage and commit.",
                                "source": "wiki/git-workflow.md#resolving-merge-conflicts",
                            }
                        ),
                    }
                }
            ]
        }

    data = _run_agent_with_fake_llm("How do you resolve a merge conflict?", responder)

    assert "wiki/git-workflow.md" in data["source"]
    tools_used = [tool_call["tool"] for tool_call in data["tool_calls"]]
    assert "read_file" in tools_used
    assert "list_files" in tools_used


def test_documentation_agent_lists_wiki_files() -> None:
    def responder(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        messages = payload["messages"]

        if state["calls"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [_make_tool_call("call-1", "list_files", "wiki")],
                        }
                    }
                ]
            }

        assert any(
            message["role"] == "tool" and "git-workflow.md" in message["content"]
            for message in messages
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "answer": "The wiki contains documentation files such as git-workflow.md, docker.md, and ssh.md.",
                                "source": "wiki",
                            }
                        ),
                    }
                }
            ]
        }

    data = _run_agent_with_fake_llm("What files are in the wiki?", responder)

    assert data["source"] == "wiki"
    tools_used = [tool_call["tool"] for tool_call in data["tool_calls"]]
    assert "list_files" in tools_used


def test_documentation_agent_blocks_path_traversal() -> None:
    def responder(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        messages = payload["messages"]

        if state["calls"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [_make_tool_call("call-1", "read_file", "../secret.txt")],
                        }
                    }
                ]
            }

        assert any(
            message["role"] == "tool" and "not allowed" in message["content"].lower()
            for message in messages
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "answer": "The requested path is outside the project directory and cannot be accessed.",
                                "source": "wiki/file-system.md#parent-directory-",
                            }
                        ),
                    }
                }
            ]
        }

    data = _run_agent_with_fake_llm("Can you read ../secret.txt?", responder)

    assert data["tool_calls"][0]["tool"] == "read_file"
    assert "not allowed" in data["tool_calls"][0]["result"].lower()


def test_documentation_agent_supports_textual_tool_call_format() -> None:
    def responder(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        messages = payload["messages"]

        if state["calls"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<function=list_files><parameter=path>wiki</parameter></function>",
                        }
                    }
                ]
            }

        assert any(message["role"] == "tool" for message in messages)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "answer": "The wiki contains multiple markdown files.",
                                "source": "wiki",
                            }
                        ),
                    }
                }
            ]
        }

    data = _run_agent_with_fake_llm("What files are in the wiki?", responder)

    assert data["source"] == "wiki"
    assert data["tool_calls"][0]["tool"] == "list_files"
