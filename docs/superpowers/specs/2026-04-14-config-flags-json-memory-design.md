# Config Flags: JSON Generation and Memory — Design Spec

**Date:** 2026-04-14

## Overview

Add two optional boolean flags to `config.json` that let users disable structured JSON generation for translation and disable session memory. Both default to `true` for backward compatibility.

## Config Changes

`config.json` gains two new optional fields:

```json
{
  "json_enabled": true,
  "memory_enabled": true
}
```

- `json_enabled`: controls whether the translation LLM is asked to produce structured JSON output. Defaults to `true` if omitted.
- `memory_enabled`: controls whether session memory is loaded, used in prompts, and updated after each page. Defaults to `true` if omitted.

## `text_utils.py` Changes

### New parameter on `translate()`

```python
def translate(..., use_json: bool = True) -> str:
```

**When `use_json=True` (existing behavior):**
- Calls `get_formatted_user_prompt()` or `get_formatted_user_prompt_with_image()`
- Passes `format=Translation.model_json_schema()` to `call_llm()`
- Parses response via `Translation.model_validate_json()`
- Falls back via `fallback_translation()` if output is suspicious

**When `use_json=False`:**
- Calls new `get_formatted_user_prompt_plain()` helper
- No `format=` argument passed to `call_llm()`
- Response used directly after `clean_translated_text()`
- No fallback

### New helper: `get_formatted_user_prompt_plain()`

Uses `_build_prompt_base()` with a closing instruction of:

```
Output ONLY the translated text. No explanation, no commentary.
```

The two existing prompt functions (`get_formatted_user_prompt`, `get_formatted_user_prompt_with_image`) are unchanged.

## `inference.py` Changes

In `driver()`:

```python
json_enabled = config.get("json_enabled", True)
memory_enabled = config.get("memory_enabled", True)
```

- If `memory_enabled=False`: skip `load_memory()`, keep `session_memory = None`, skip `update_session_memory()` after each page.
- Pass `use_json=json_enabled` to every `translate()` call.

## Data Flow

```
config.json
  └─ json_enabled  ──► driver() ──► translate(use_json=...) ──► prompt + call_llm format
  └─ memory_enabled ──► driver() ──► load_memory / update_session_memory (skipped if False)
                                  └─ session_memory=None passed to translate()
```

## Error Handling

No new error handling needed. `config.get()` with defaults handles missing keys. The existing `clean_translated_text()` already sanitises raw LLM output for the plain-text path.

## Testing

- Existing tests cover the `use_json=True` path (no change needed).
- New unit tests for `translate()` with `use_json=False`: assert `call_llm` is called without `format`, assert result is the cleaned plain response, assert `fallback_translation` is never called.
- New unit test for `driver()` with `memory_enabled=False`: assert `load_memory` and `update_session_memory` are not called.
