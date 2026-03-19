# Agent Architecture

## Overview

`agent.py` is a CLI documentation agent. It accepts a question as the first command-line argument, calls an OpenAI-compatible LLM, lets the LLM use local tools, and prints a single JSON object to stdout.

Current output fields:

- `answer`
- `source`
- `tool_calls`

## LLM Provider

The agent uses Qwen Code API deployed on the VM.

Current model:

- `qwen3-coder-flash`

## Configuration

The agent reads these values from environment variables, with `.env.agent.secret` used as a local convenience file:

- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_MODEL`

## Tools

The agent exposes two function-calling tools to the LLM:

- `read_file(path)` reads a repository file and returns its contents
- `list_files(path)` lists files and directories in a repository path

Both tools enforce path safety by resolving the requested path against the repository root and rejecting traversal outside the project directory.

## Agentic Loop

1. Parse the question from CLI arguments.
2. Load LLM configuration from the environment.
3. Send the question, system prompt, and tool schemas to the LLM.
4. If the LLM returns `tool_calls`, execute the tools locally.
5. Append tool results back to the conversation as `tool` messages.
6. Repeat until the model returns a final answer or the agent reaches the 10-tool-call limit.
7. Print valid JSON to stdout.

## System Prompt Strategy

The system prompt tells the model to:

- inspect repository files with tools before answering
- prefer `list_files` to discover relevant wiki paths
- use `read_file` to inspect relevant files
- answer only from tool results
- return final JSON with `answer` and `source`
- include a relative wiki path and heading anchor in `source` when possible

## Error Handling

- Missing CLI argument: print to stderr and exit non-zero.
- Missing or invalid environment variables: print to stderr and exit non-zero.
- Failed API request or malformed API response: print to stderr and exit non-zero.
- Tool errors return readable strings instead of crashing the agent.

## How To Run

Create the environment file:

```bash
cp .env.agent.example .env.agent.secret
```

Fill in:

- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_MODEL`

Run the agent:

```bash
uv run agent.py "How do you resolve a merge conflict?"
```

## Testing

Regression tests run `agent.py` as a subprocess against a fake local LLM server. The tests verify:

- the JSON output structure
- tool usage in `tool_calls`
- source references in `source`
- safe handling of blocked paths
