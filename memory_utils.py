import os
import re
from data_model import CharacterEntry, EntityEntry, SessionMemory


def _escape_cell(value: str) -> str:
    return value.replace("|", "&#124;")


def serialize_memory(memory: SessionMemory, path: str) -> None:
    lines = ["# Session Memory", ""]
    lines.append("## Story Summary")
    lines.append(memory.story_summary or "")
    lines.append("")
    lines.append("## Characters")
    lines.append("| Original | Translated | Gender | Notes |")
    lines.append("|----------|------------|--------|-------|")
    for char in memory.characters:
        lines.append(f"| {_escape_cell(char.original_name)} | {_escape_cell(char.translated_name)} | {char.gender} | {_escape_cell(char.notes)} |")
    lines.append("")
    lines.append("## Places")
    lines.append("| Original | Translated |")
    lines.append("|----------|------------|")
    for place in memory.places:
        lines.append(f"| {_escape_cell(place.original)} | {_escape_cell(place.translated)} |")
    lines.append("")
    lines.append("## Organizations")
    lines.append("| Original | Translated |")
    lines.append("|----------|------------|")
    for org in memory.organizations:
        lines.append(f"| {_escape_cell(org.original)} | {_escape_cell(org.translated)} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_memory(path: str) -> SessionMemory:
    if not os.path.exists(path):
        return SessionMemory()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    memory = SessionMemory()
    sections = re.split(r'\n## ', content)

    for section in sections:
        header, _, body = section.partition('\n')
        header = header.lstrip('# ').strip()

        if header == "Story Summary":
            memory.story_summary = body.strip()

        elif header == "Characters":
            for row in _parse_table_rows(body):
                if len(row) >= 4:
                    memory.characters.append(CharacterEntry(
                        original_name=_unescape_cell(row[0]),
                        translated_name=_unescape_cell(row[1]),
                        gender=row[2],
                        notes=_unescape_cell(row[3]),
                    ))

        elif header == "Places":
            for row in _parse_table_rows(body):
                if len(row) >= 2:
                    memory.places.append(EntityEntry(original=_unescape_cell(row[0]), translated=_unescape_cell(row[1])))

        elif header == "Organizations":
            for row in _parse_table_rows(body):
                if len(row) >= 2:
                    memory.organizations.append(EntityEntry(original=_unescape_cell(row[0]), translated=_unescape_cell(row[1])))

    return memory


def _unescape_cell(value: str) -> str:
    return value.replace("&#124;", "|")


def _parse_table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('|') and not re.match(r'^\|[-|: ]+\|$', line):
            cols = [c.strip() for c in line.strip('|').split('|')]
            rows.append(cols)
    return rows[1:]  # skip header row


def format_memory_for_prompt(memory: SessionMemory) -> str:
    has_entities = any([memory.characters, memory.places, memory.organizations])
    if not has_entities and not memory.story_summary:
        return ""

    parts = []

    if has_entities:
        parts.append("Known entities (use these translations consistently):")
        if memory.characters:
            char_list = ", ".join(
                f"{c.translated_name} ({c.gender}{', ' + c.notes if c.notes else ''})"
                for c in memory.characters
            )
            parts.append(f"- Characters: {char_list}")
        if memory.places:
            parts.append(f"- Places: {', '.join(p.translated for p in memory.places)}")
        if memory.organizations:
            parts.append(f"- Organizations: {', '.join(o.translated for o in memory.organizations)}")

    if memory.story_summary:
        if parts:
            parts.append("")  # blank line separator
        parts.append(f"Story so far: {memory.story_summary}")

    return "\n".join(parts)
