# Agent Architecture

## Overview

`agent.py` is now a CLI repository-and-system agent. It accepts the user question as the first command-line argument, calls an OpenAI-compatible LLM, lets the LLM use local tools, and prints one JSON object to stdout.

Current output fields:

- `answer`
- `source` (optional)
- `tool_calls`

## Configuration

The agent reads all runtime configuration from environment variables. Nothing important is hardcoded.

LLM settings:

- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_MODEL`

Backend settings for the system tool:

- `LMS_API_KEY`
- `AGENT_API_BASE_URL` with default `http://localhost:42002`

For local development, `.env.agent.secret` and `.env.docker.secret` are convenience files only. The autochecker can inject completely different values, so reading from the environment was an important Task 3 requirement.

## Tools

The agent exposes three function-calling tools:

- `read_file(path)` reads a repository file
- `list_files(path)` lists files and directories in a repository path
- `query_api(method, path, body?)` sends an HTTP request to the running backend and returns a JSON string with `status_code` and `body`

`read_file` and `list_files` enforce path safety by resolving each path against the repository root and rejecting traversal outside the project directory.

`query_api` authenticates with `Authorization: Bearer <LMS_API_KEY>` by default. It does not crash on HTTP `4xx` or `5xx`, because Task 3 includes questions where the agent must inspect a failing API response and then diagnose it from source code. The tool also adds small metadata such as `body_length` for list responses, which makes count questions more reliable.

## Tool Routing

The system prompt teaches the model to choose tools by evidence source:

- wiki or documentation questions -> `read_file`
- file/module discovery questions -> `list_files`
- live backend state, status codes, counts, and runtime failures -> `query_api`

If an API request fails, the intended behavior is to reproduce the error with `query_api` and then inspect the relevant backend file with `read_file`.

## Agentic Loop

1. Parse the question from the CLI.
2. Load settings from the environment.
3. Send the question, system prompt, and tool schemas to the LLM.
4. Execute returned tool calls.
5. Append tool results back as `tool` messages.
6. Repeat until the model returns a final answer or the 10-tool-call limit is hit.
7. Print valid JSON to stdout.

I kept the Task 2 loop and extended it instead of rewriting it. That made the Task 3 changes smaller and easier to debug.

## Benchmark Lessons

The benchmark exposed a few important problems that were not obvious from unit tests alone.

First, source formatting matters. An answer could be correct and still fail because the source was too loose, such as `github.md` instead of `wiki/github.md`. I added source normalization so wiki answers use repository-relative paths more consistently.

Second, the model sometimes stopped too early after `list_files`. For example, it could discover `ssh.md` and then answer with a guess instead of actually reading the file. I added continuation guards so non-listing questions keep going until the agent has stronger evidence.

Third, the configured Qwen-compatible API sometimes returned `content: null` with tool calls. Sending that `null` back in the next chat-completions request caused provider-side parameter errors. The fix was to normalize `None` to an empty string before storing assistant content.

Fourth, I added a few evidence-based fallbacks from tool results. For example, if the model stalls after reading `backend/app/main.py`, the agent can still recognize `FastAPI` from the imports. If it already listed `backend/app/routers`, it can still construct a stable answer about router modules from the filenames.

## Testing

Regression tests run `agent.py` as a subprocess against fake local servers. The tests verify:

- JSON output shape
- tool usage recorded in `tool_calls`
- safe path handling
- source-code questions triggering `read_file`
- API questions triggering `query_api`
- backend bearer authentication with `LMS_API_KEY`
- missing-auth probing behavior for status-code questions

## Current Eval Status

Latest confirmed local benchmark progress in this environment:

- the first uninterrupted full run reached `5/10`
- questions `0` through `4` were passing after the Task 3 fixes

After that, further uninterrupted end-to-end runs were blocked by an external `401 Not authenticated with Qwen` error from the configured LLM provider. Because of that provider-side failure, I could not honestly record a final uninterrupted `10/10` local run from this environment, even though the agent code, plan, and regression tests were updated to address the failures observed before the provider error.
