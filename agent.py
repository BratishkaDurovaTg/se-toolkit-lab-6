import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent
MAX_TOOL_CALLS = 10

SYSTEM_PROMPT = """
You are a documentation agent for this repository.

Use the available tools to inspect the local project files before answering.
Prefer list_files to discover relevant wiki paths, then use read_file to inspect
the most relevant files.

Answer only from tool results. Do not rely on memory.

When you have enough information, return valid JSON with exactly these fields:
{"answer":"...","source":"wiki/path.md#section-anchor"}

Rules:
- The source must be a relative wiki path and a heading anchor when possible.
- For directory listing questions, source may be "wiki".
- Do not wrap the final JSON in markdown fences.
- Keep the final answer concise.
""".strip()


class Settings(BaseSettings):
    llm_api_key: str = Field(alias="LLM_API_KEY")
    llm_api_base: str = Field(alias="LLM_API_BASE")
    llm_model: str = Field(alias="LLM_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env.agent.secret",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ToolCallRecord(BaseModel):
    tool: str
    args: dict[str, Any]
    result: str


class AgentOutput(BaseModel):
    answer: str
    source: str
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
                "description": "Read a file from the project repository.",
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
                "description": "List files and directories at a given path in the repository.",
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


def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    path = str(arguments.get("path", ""))

    if name == "read_file":
        return read_file(path)
    if name == "list_files":
        return list_files(path)

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


def _looks_like_broken_model_output(answer: str) -> bool:
    stripped = answer.strip()
    return stripped.startswith("{") or '"tool_calls"' in stripped or "<function=" in stripped


def _fallback_answer(tool_records: list[ToolCallRecord], current_answer: str) -> str:
    if not _looks_like_broken_model_output(current_answer):
        return current_answer

    for tool_record in reversed(tool_records):
        if tool_record.tool == "list_files":
            entries = [line.strip() for line in tool_record.result.splitlines() if line.strip()]
            preview = ", ".join(entries[:10])
            if preview:
                path = str(tool_record.args.get("path", "this path")).strip() or "this path"
                suffix = ", ..." if len(entries) > 10 else ""
                return f"{path} contains: {preview}{suffix}."

    return current_answer


def _should_continue_with_tools(answer: str, tool_records: list[ToolCallRecord]) -> bool:
    lowered = answer.strip().lower()
    if not lowered:
        return True

    if any(tool_record.tool == "read_file" for tool_record in tool_records):
        return False

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
        if "content" in message:
            assistant_message["content"] = message["content"]
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]
        messages.append(assistant_message)

        tool_calls = message.get("tool_calls") or _parse_text_tool_calls(message.get("content", ""))
        if not tool_calls:
            answer, source = _parse_final_response(message.get("content", ""))
            if _should_continue_with_tools(answer, tool_records) and not nudged_for_read:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not answered the question yet. Read the most relevant wiki file "
                            "with read_file, then return final JSON with answer and source."
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

            result = _execute_tool(tool_name, tool_arguments)
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

    if not source:
        source = _fallback_source(tool_records)

    answer = _fallback_answer(tool_records, answer)

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
    print(output.model_dump_json())


if __name__ == "__main__":
    main()
