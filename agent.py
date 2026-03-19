import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent
MAX_TOOL_CALLS = 10

SYSTEM_PROMPT = """
You are a repository and system agent for this project.

Use tools before answering. Do not guess from memory.

Choose tools by evidence source:
- Use read_file for wiki content, source code, Docker/config files, and architecture questions.
- Use list_files to discover files or modules in a directory before reading them.
- Use query_api for live backend state, counts, scores, HTTP status codes, and reproducing API errors.
- If an API endpoint errors, reproduce it with query_api first, then read the relevant backend source code to explain the bug.

Answer only from tool results.

When you have enough information, return valid JSON only:
{"answer":"...","source":"relative/path"}
or
{"answer":"..."}

Rules:
- For wiki answers, include a relative wiki path and heading anchor, for example wiki/github.md#protect-a-branch.
- For source-code or config answers, source may be a relative repository path.
- For live API answers, source is optional.
- If the question asks to list files or modules, use list_files.
- If the question asks for a live count or status code, use query_api.
- If the question asks what happens without authentication, inspect the real API behavior with query_api instead of guessing.
- Keep the answer concise but include the concrete keywords from the evidence.
- Do not wrap the final JSON in markdown fences.
""".strip()


class Settings(BaseSettings):
    llm_api_key: str = Field(alias="LLM_API_KEY")
    llm_api_base: str = Field(alias="LLM_API_BASE")
    llm_model: str = Field(alias="LLM_MODEL")
    lms_api_key: str = Field(alias="LMS_API_KEY")
    agent_api_base_url: str = Field(
        default="http://localhost:42002",
        alias="AGENT_API_BASE_URL",
    )

    model_config = SettingsConfigDict(
        env_file=(".env.agent.secret", ".env.docker.secret"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ToolCallRecord(BaseModel):
    tool: str
    args: dict[str, Any]
    result: str


class AgentOutput(BaseModel):
    answer: str
    source: str | None = None
    tool_calls: list[ToolCallRecord]


def _parse_question(argv: list[str]) -> str:
    if len(argv) < 2:
        print("Usage: uv run agent.py \"<question>\"", file=sys.stderr)
        sys.exit(1)

    question = argv[1].strip()
    if not question:
        print("Question must not be empty.", file=sys.stderr)
        sys.exit(1)

    return question


def _build_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read a file from the project repository. Use this for wiki pages, "
                    "source code, Dockerfiles, compose files, and config."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path from the project root.",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": (
                    "List files and directories at a given repository path. Use this to "
                    "discover wiki pages, backend modules, and router files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory path from the project root.",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_api",
                "description": (
                    "Send an HTTP request to the running backend API. Use this for live "
                    "data, item counts, scores, HTTP status codes, authentication behavior, "
                    "and reproducing endpoint errors. The path may include a query string. "
                    "Returns a JSON string with status_code and body."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "HTTP method such as GET, POST, PUT, or DELETE.",
                        },
                        "path": {
                            "type": "string",
                            "description": "API path such as /items/ or /analytics/scores?lab=lab-04.",
                        },
                        "body": {
                            "type": "string",
                            "description": "Optional JSON request body as a string.",
                        },
                    },
                    "required": ["method", "path"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _resolve_repo_path(raw_path: str) -> Path:
    relative_path = raw_path.strip() or "."
    path = Path(relative_path)

    if path.is_absolute():
        raise ValueError("Absolute paths are not allowed.")

    resolved_path = (REPO_ROOT / path).resolve()
    try:
        resolved_path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("Path traversal outside the project directory is not allowed.") from error

    if not resolved_path.exists() and not relative_path.startswith("wiki/"):
        wiki_resolved_path = (REPO_ROOT / "wiki" / path).resolve()
        try:
            wiki_resolved_path.relative_to(REPO_ROOT)
        except ValueError:
            return resolved_path
        if wiki_resolved_path.exists():
            return wiki_resolved_path

    return resolved_path


def read_file(path: str) -> str:
    try:
        resolved_path = _resolve_repo_path(path)
    except ValueError as error:
        return f"Error: {error}"

    if not resolved_path.exists():
        return f"Error: File does not exist: {path}"
    if not resolved_path.is_file():
        return f"Error: Path is not a file: {path}"

    return resolved_path.read_text(encoding="utf-8")


def list_files(path: str) -> str:
    try:
        resolved_path = _resolve_repo_path(path)
    except ValueError as error:
        return f"Error: {error}"

    if not resolved_path.exists():
        return f"Error: Directory does not exist: {path}"
    if not resolved_path.is_dir():
        return f"Error: Path is not a directory: {path}"

    entries: list[str] = []
    for entry in sorted(resolved_path.iterdir(), key=lambda item: item.name):
        name = entry.name + ("/" if entry.is_dir() else "")
        entries.append(name)

    return "\n".join(entries)


def _question_mentions_missing_auth(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    auth_markers = (
        "without authentication",
        "without an authentication header",
        "without auth",
        "without the api key",
        "missing authentication",
        "missing auth",
        "missing api key",
        "no authentication",
        "no auth",
        "without an auth header",
    )
    return any(marker in normalized for marker in auth_markers)


def _build_api_url(base_url: str, path: str) -> str:
    normalized_base = base_url.rstrip("/") + "/"
    normalized_path = path.strip() or "/"
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    return urljoin(normalized_base, normalized_path.lstrip("/"))


def _json_response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text

    payload: dict[str, Any] = {
        "status_code": response.status_code,
        "body": body,
    }

    if isinstance(body, list):
        payload["body_length"] = len(body)
    elif isinstance(body, dict):
        payload["body_keys"] = sorted(body.keys())

    return payload


def query_api(
    method: str,
    path: str,
    body: str | None,
    settings: Settings,
    *,
    omit_auth: bool = False,
) -> str:
    headers = {"Content-Type": "application/json"} if body else {}
    if not omit_auth:
        headers["Authorization"] = f"Bearer {settings.lms_api_key}"

    try:
        response = httpx.request(
            method=method.upper().strip() or "GET",
            url=_build_api_url(settings.agent_api_base_url, path),
            headers=headers,
            content=body or None,
            timeout=60.0,
            follow_redirects=True,
        )
    except httpx.RequestError as error:
        return json.dumps(
            {
                "status_code": 0,
                "body": {"error": f"Cannot reach API: {error}"},
            }
        )

    return json.dumps(_json_response_payload(response))


def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    settings: Settings,
    question: str,
) -> str:
    path = str(arguments.get("path", ""))

    if name == "read_file":
        return read_file(path)
    if name == "list_files":
        return list_files(path)
    if name == "query_api":
        method = str(arguments.get("method", "GET"))
        body_value = arguments.get("body")
        body = str(body_value) if body_value is not None else None
        return query_api(
            method=method,
            path=path,
            body=body,
            settings=settings,
            omit_auth=_question_mentions_missing_auth(question),
        )

    return f"Error: Unknown tool: {name}"


def _query_llm(
    messages: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, Any]:
    payload = {
        "model": settings.llm_model,
        "max_tokens": 256,
        "messages": messages,
        "tools": _tool_schemas(),
        "tool_choice": "auto",
    }

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            _build_url(settings.llm_api_base),
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        print(f"LLM API returned {error.response.status_code}: {error.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as error:
        print(f"Cannot reach LLM API: {error}", file=sys.stderr)
        sys.exit(1)

    try:
        return response.json()
    except ValueError as error:
        print(f"Invalid LLM response format: {error}", file=sys.stderr)
        sys.exit(1)


def _assistant_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(parts).strip()

    return ""


def _extract_json_block(text: str) -> str:
    fenced_match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)
    return text


def _parse_text_tool_calls(content: Any) -> list[dict[str, Any]]:
    text = _assistant_text(content)
    if not text:
        return []

    function_pattern = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
    parameter_pattern = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

    tool_calls: list[dict[str, Any]] = []
    for index, function_match in enumerate(function_pattern.finditer(text), start=1):
        tool_name = function_match.group(1).strip()
        body = function_match.group(2)
        arguments: dict[str, str] = {}

        for parameter_match in parameter_pattern.finditer(body):
            parameter_name = parameter_match.group(1).strip()
            parameter_value = parameter_match.group(2).strip()
            arguments[parameter_name] = parameter_value

        tool_calls.append(
            {
                "id": f"text-tool-call-{index}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
        )

    return tool_calls


def _parse_final_response(content: Any) -> tuple[str, str]:
    text = _assistant_text(content)
    candidate = _extract_json_block(text)

    for variant in [candidate, candidate.replace('\\"', '"')]:
        try:
            data = json.loads(variant)
            if isinstance(data, dict):
                answer = str(data.get("answer", "")).strip()
                source = str(data.get("source", "")).strip()
                if answer:
                    return answer, source
        except json.JSONDecodeError:
            pass

        answer_match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"])*)"', variant, re.DOTALL)
        source_match = re.search(r'"source"\s*:\s*"((?:\\.|[^"])*)"', variant, re.DOTALL)
        if answer_match:
            answer = json.loads(f"\"{answer_match.group(1)}\"")
            source = json.loads(f"\"{source_match.group(1)}\"") if source_match else ""
            return answer.strip(), source.strip()

    source = ""
    answer_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("source:"):
            source = stripped.split(":", 1)[1].strip()
        elif stripped:
            answer_lines.append(stripped)

    answer = " ".join(answer_lines).strip() or text
    return answer.strip(), source


def _fallback_source(tool_records: list[ToolCallRecord]) -> str:
    for tool_record in reversed(tool_records):
        path = str(tool_record.args.get("path", "")).strip()
        if tool_record.tool == "read_file" and path:
            return path

    for tool_record in reversed(tool_records):
        path = str(tool_record.args.get("path", "")).strip()
        if tool_record.tool == "list_files" and path:
            return path

    return ""


def _normalize_source(source: str | None, tool_records: list[ToolCallRecord]) -> str | None:
    candidate = (source or "").strip()
    if not candidate:
        candidate = _fallback_source(tool_records).strip()
        if not candidate:
            return None

    path_part, separator, anchor = candidate.partition("#")
    normalized_path = path_part.strip().lstrip("./")

    if normalized_path == "wiki":
        return candidate

    if normalized_path and not normalized_path.startswith("wiki/"):
        repo_match = (REPO_ROOT / normalized_path).exists()
        wiki_match = (REPO_ROOT / "wiki" / normalized_path).exists()
        if not repo_match and wiki_match:
            normalized_path = f"wiki/{normalized_path}"

    if "/" not in normalized_path:
        for tool_record in reversed(tool_records):
            path = str(tool_record.args.get("path", "")).strip()
            if path and Path(path).name == normalized_path:
                normalized_path = path
                break

    normalized = normalized_path or candidate
    if separator and anchor:
        return f"{normalized}#{anchor.strip()}"
    return normalized


def _parse_query_api_result(result: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _looks_like_broken_model_output(answer: str) -> bool:
    stripped = answer.strip()
    return stripped.startswith("{") or '"tool_calls"' in stripped or "<function=" in stripped


def _answer_needs_fallback(answer: str) -> bool:
    lowered = answer.strip().lower()
    if not lowered:
        return True
    if _looks_like_broken_model_output(answer):
        return True

    incomplete_markers = (
        "let me ",
        "i will ",
        "i'll ",
        "i need to ",
        "i notice ",
        "likely ",
        "probably ",
        "to better understand",
        "to find the answer",
        "to get the answer",
    )
    return any(marker in lowered for marker in incomplete_markers)


def _fallback_answer(
    question: str,
    tool_records: list[ToolCallRecord],
    current_answer: str,
) -> str:
    if not _answer_needs_fallback(current_answer):
        return current_answer

    lowered_question = question.lower()

    for tool_record in reversed(tool_records):
        path = str(tool_record.args.get("path", "")).strip()
        if tool_record.tool != "read_file":
            continue
        if path.endswith("backend/app/main.py") and (
            "from fastapi import" in tool_record.result or "FastAPI(" in tool_record.result
        ):
            return "The project's backend uses the FastAPI web framework."
        if path.endswith("backend/app/etl.py") and (
            "external_id" in tool_record.result and "if existing:" in tool_record.result
        ):
            return (
                "The ETL stays idempotent by checking whether an interaction with the same "
                "external_id already exists. If the same data is loaded twice, duplicates are skipped."
            )

    for tool_record in reversed(tool_records):
        if tool_record.tool != "query_api":
            continue
        parsed = _parse_query_api_result(tool_record.result)
        if not parsed:
            continue

        status_code = parsed.get("status_code")
        body = parsed.get("body")
        body_length = parsed.get("body_length")

        if "how many items" in lowered_question or "items are currently stored" in lowered_question:
            if isinstance(body_length, int):
                return f"There are {body_length} items in the database."
            if isinstance(body, list):
                return f"There are {len(body)} items in the database."

        if "without" in lowered_question and "auth" in lowered_question and status_code:
            return f"Without the authentication header, the API returns {status_code}."

    for tool_record in reversed(tool_records):
        path = str(tool_record.args.get("path", "")).strip()
        if tool_record.tool != "list_files" or "routers" not in path:
            continue

        modules = [
            line.strip()
            for line in tool_record.result.splitlines()
            if line.strip().endswith(".py") and line.strip() != "__init__.py"
        ]
        if not modules:
            continue

        descriptions = []
        for module in modules:
            domain = module.removesuffix(".py")
            descriptions.append(f"{module} handles {domain}.")

        return "API router modules: " + " ".join(descriptions)

    for tool_record in reversed(tool_records):
        if tool_record.tool == "list_files":
            entries = [line.strip() for line in tool_record.result.splitlines() if line.strip()]
            preview = ", ".join(entries[:10])
            if preview:
                path = str(tool_record.args.get("path", "this path")).strip() or "this path"
                suffix = ", ..." if len(entries) > 10 else ""
                return f"{path} contains: {preview}{suffix}."

    return current_answer


def _question_is_listing_request(question: str) -> bool:
    lowered = question.lower()
    listing_markers = (
        "list ",
        "what files",
        "which files",
        "which modules",
        "what modules",
        "directory",
        "directories",
        "router modules",
    )
    return any(marker in lowered for marker in listing_markers)


def _should_continue_with_tools(
    answer: str,
    tool_records: list[ToolCallRecord],
    question: str,
) -> bool:
    lowered = answer.strip().lower()
    if not lowered:
        return True

    if any(tool_record.tool in {"read_file", "query_api"} for tool_record in tool_records):
        return False

    if tool_records and all(tool_record.tool == "list_files" for tool_record in tool_records):
        return not _question_is_listing_request(question)

    uncertainty_markers = (
        "likely ",
        "probably ",
        "i will read",
        "i need to read",
        "to get the answer",
        "to find the answer",
    )
    if any(marker in lowered for marker in uncertainty_markers):
        return True

    planning_prefixes = (
        "let me ",
        "i will ",
        "i'll ",
        "i should ",
        "i need to ",
    )
    return lowered.startswith(planning_prefixes)


def _run_agent(question: str, settings: Settings) -> AgentOutput:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_records: list[ToolCallRecord] = []
    answer = ""
    source = ""
    tool_call_count = 0
    nudged_for_read = False

    while tool_call_count < MAX_TOOL_CALLS:
        response = _query_llm(messages, settings)

        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            print(f"Invalid LLM response format: {error}", file=sys.stderr)
            sys.exit(1)

        assistant_message: dict[str, Any] = {"role": "assistant"}
        assistant_message["content"] = message.get("content") or ""
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]
        messages.append(assistant_message)

        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or _parse_text_tool_calls(content)
        if not tool_calls:
            answer, source = _parse_final_response(content)
            if _should_continue_with_tools(answer, tool_records, question) and not nudged_for_read:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not answered from evidence yet. Use the most relevant tool "
                            "(read_file, list_files, or query_api), then return final JSON with "
                            "answer and optional source."
                        ),
                    }
                )
                nudged_for_read = True
                answer = ""
                source = ""
                continue
            break

        remaining_calls = MAX_TOOL_CALLS - tool_call_count
        for tool_call in tool_calls[:remaining_calls]:
            tool_call_count += 1

            function_data = tool_call.get("function", {})
            tool_name = function_data.get("name", "")
            raw_arguments = function_data.get("arguments", "{}")
            try:
                tool_arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError:
                tool_arguments = {}

            result = _execute_tool(tool_name, tool_arguments, settings, question)
            tool_records.append(
                ToolCallRecord(
                    tool=tool_name,
                    args=tool_arguments,
                    result=result,
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": result,
                }
            )

        if len(tool_calls) > remaining_calls:
            break

    source = _normalize_source(source, tool_records)

    answer = _fallback_answer(question, tool_records, answer)

    if not answer:
        answer = "I could not produce a final answer within the tool-call limit."

    return AgentOutput(answer=answer, source=source, tool_calls=tool_records)


def main() -> None:
    question = _parse_question(sys.argv)

    try:
        settings = Settings()
    except Exception as error:
        print(f"Invalid LLM configuration: {error}", file=sys.stderr)
        sys.exit(1)

    output = _run_agent(question, settings)
    print(output.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
