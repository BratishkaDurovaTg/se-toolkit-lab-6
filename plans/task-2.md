# Task 2 Plan

## Goal

Turn the Task 1 CLI into a documentation agent that can inspect the repository wiki with tools and answer questions using an agentic loop.

## Tool Schemas

Define two OpenAI-compatible function-calling tools:

- `read_file(path)` for reading a repository file
- `list_files(path)` for listing directory entries

Both schemas will expose a single required string parameter: `path`.

## Path Security

Resolve every requested path against the project root with `Path.resolve()`.
Reject:

- absolute paths
- any path that escapes the repository root
- wrong path type (file vs directory)

Return readable error strings instead of crashing.

## Agentic Loop

1. Send the user question, system prompt, and tool schemas to the LLM.
2. If the LLM returns `tool_calls`, execute each tool locally.
3. Append tool results as `tool` messages.
4. Repeat until the model returns a final text response or 10 tool calls are reached.

## Final Answer Format

Ask the model to return JSON with:

- `answer`
- `source`

Parse that final JSON and print the outer CLI result as:

- `answer`
- `source`
- `tool_calls`

## System Prompt Strategy

Tell the model to:

- use `list_files` to discover relevant wiki files
- use `read_file` to inspect them
- answer only from tool results
- include a relative wiki source path with a section anchor when possible

## Testing

Add regression tests that run `agent.py` as a subprocess against a fake local LLM server and verify:

- wiki questions trigger `read_file`
- directory questions trigger `list_files`
- tool outputs appear in `tool_calls`
- source references are returned
- path traversal is blocked
