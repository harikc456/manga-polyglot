# Memory Module Design

**Date:** 2026-04-12  
**Status:** Approved

## Overview

Add a session-scoped memory module to manga-polyglot that tracks named entities (characters, places, organizations), character gender, and a rolling story summary across all pages in a translation run. This enables consistent entity translations and correct pronoun usage throughout a session.

---

## Goals

- Consistent translation of character names, place names, and organization names across pages
- Correct pronoun usage via tracked character gender
- Rolling story summary injected as context to improve translation quality for later pages
- Human-readable memory file for debugging and inspection

---

## Data Model (`data_model.py`)

Two new Pydantic models:

```python
class CharacterEntry(BaseModel):
    original_name: str       # name as it appears in source (e.g. 田中)
    translated_name: str     # established translation (e.g. Tanaka)
    gender: str              # "male" | "female" | "unknown"
    notes: str = ""          # optional: role, speech register, etc.

class SessionMemory(BaseModel):
    characters: list[CharacterEntry] = []
    places: list[EntityEntry] = []
    organizations: list[EntityEntry] = []
    story_summary: str = ""         # rolling summary, ~150 words max
```

---

## Persistence

- **Format:** Markdown file at `{temp_dir}/memory.md`
- **On session start:** `driver()` checks for `{temp_dir}/memory.md`. If present, loads it into a `SessionMemory` instance (supports resuming interrupted runs). If absent, starts with an empty `SessionMemory`.
- **After each page:** The updated `SessionMemory` is serialized back to `{temp_dir}/memory.md`.

### Markdown format

```markdown
# Session Memory

## Story Summary
<rolling summary text>

## Characters
| Original | Translated | Gender | Notes |
|----------|------------|--------|-------|
| 田中 | Tanaka | male | protagonist |

## Places
| Original | Translated |
|----------|------------|
| 新宿 | Shinjuku |

## Organizations
| Original | Translated |
|----------|------------|
| 黒烏 | Black Crow Corp |
```

---

## Memory Update Flow (`text_utils.py`)

New function: `update_session_memory(translations, session_memory, model, temp_dir) -> SessionMemory`

**Called in `driver()` once per page**, after the translation loop for that page completes:

```python
session_memory = update_session_memory(translations, session_memory, llm_name, temp_dir)
```

**Inputs:**
- `translations`: list of `{"original": str, "translated": str}` pairs from the current page
- `session_memory`: current `SessionMemory` state
- `model`: LLM model name (same as used for translation)
- `temp_dir`: path for writing `memory.md`

**Behaviour:**
- Builds a prompt containing the current page's translations and the current memory state
- Calls the LLM using `SessionMemory.model_json_schema()` as the structured output format
- LLM is instructed to:
  - Identify new named entities (characters with gender, places, organizations)
  - Update existing entries if the LLM finds a better/corrected form
  - Produce a revised rolling story summary (max ~150 words)
- Returns the updated `SessionMemory`
- Serializes to `{temp_dir}/memory.md`

---

## Prompt Injection (`text_utils.py`)

`get_formatted_user_prompt()` and `get_formatted_user_prompt_with_image()` receive a new optional parameter:

```python
session_memory: SessionMemory = None
```

If `session_memory` is non-empty, a section is injected between the context block and the previous-translations block:

```
Known entities (use these translations consistently):
- Characters: Tanaka (male), Sakura (female, childhood friend)
- Places: Shinjuku, Black Crow HQ
- Organizations: Black Crow Corp

Story so far: <rolling summary>
```

**Prompt section order:**
1. Page context (lookback + current + lookahead)
2. Session memory (entities + story summary) ← new
3. Previous translations on this page
4. Text to translate + output instructions

---

## Files Changed

| File | Change |
|------|--------|
| `data_model.py` | Add `CharacterEntry`, `SessionMemory` models |
| `text_utils.py` | Add `update_session_memory()`, inject memory into prompt functions |
| `inference.py` | Load/pass `SessionMemory` in `driver()`, call `update_session_memory()` after each page |

---

## Out of Scope

- Persisting memory across different manga series or separate runs (beyond resume support)
- A UI or manual editor for the memory file
- Automatic gender correction based on downstream context
