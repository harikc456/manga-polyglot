import os
import re
import unicodedata
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


# Longer "names" are almost always phrases the model mistook for an entity.
MAX_ENTITY_NAME_LENGTH = 20
# Longer notes are the model reasoning in the field, not a note about the character.
MAX_NOTES_LENGTH = 80
# Japanese honorifics: 田中先輩 and 田中さん are the same character as 田中.
_HONORIFICS = ("先輩", "せんぱい", "先生", "さん", "くん", "君", "ちゃん", "様", "さま", "殿", "氏")


def normalize_name(text: str) -> str:
    """Comparison key for entity names: NFKC (full/half-width alike), no whitespace, case-insensitive."""
    return "".join(unicodedata.normalize("NFKC", text).split()).casefold()


def character_key(name: str) -> str:
    """normalize_name without a trailing honorific, as long as some name is left."""
    key = normalize_name(name)
    for honorific in _HONORIFICS:
        if key.endswith(honorific) and len(key) > len(honorific):
            return key[: -len(honorific)]
    return key


def _clean_notes(notes: str) -> str:
    notes = notes.strip()
    return notes if len(notes) <= MAX_NOTES_LENGTH else ""


def _valid_entity(original: str, translated: str, source_key: str) -> bool:
    key = normalize_name(original)
    return (
        bool(key)
        and bool(translated.strip())
        and len(key) <= MAX_ENTITY_NAME_LENGTH
        and key in source_key
    )


def _merge_entities(existing: list[EntityEntry], new: list[EntityEntry], source_key: str) -> list[EntityEntry]:
    merged = list(existing)
    seen = {normalize_name(e.original) for e in existing}
    for entry in new:
        key = normalize_name(entry.original)
        if key not in seen and _valid_entity(entry.original, entry.translated, source_key):
            merged.append(entry)
            seen.add(key)
    return merged


def merge_memory(memory: SessionMemory, update: SessionMemory, source_text: str) -> SessionMemory:
    """Add the update's new entities to memory; never drop or rename an existing one.

    An entity is accepted only if its original name occurs in source_text (the page's untranslated text).
    For a known character, the update may only fill in a gender that was unknown or notes that were empty.
    """
    source_key = normalize_name(source_text)

    characters = [c.model_copy() for c in memory.characters]
    by_key = {character_key(c.original_name): c for c in characters}
    for char in update.characters:
        key = character_key(char.original_name)
        known = by_key.get(key)
        if known is not None:
            if known.gender == "unknown":
                known.gender = char.gender
            if not known.notes:
                known.notes = _clean_notes(char.notes)
        elif _valid_entity(char.original_name, char.translated_name, source_key):
            by_key[key] = char.model_copy(update={"notes": _clean_notes(char.notes)})
            characters.append(by_key[key])

    return SessionMemory(
        characters=characters,
        places=_merge_entities(memory.places, update.places, source_key),
        organizations=_merge_entities(memory.organizations, update.organizations, source_key),
        story_summary=update.story_summary.strip() or memory.story_summary,
    )


def filter_memory(memory: SessionMemory, text: str) -> SessionMemory:
    """Keep only the entities whose original name occurs in text; the story summary is kept as is."""
    key = normalize_name(text)
    return SessionMemory(
        characters=[c for c in memory.characters if character_key(c.original_name) in key],
        places=[p for p in memory.places if normalize_name(p.original) in key],
        organizations=[o for o in memory.organizations if normalize_name(o.original) in key],
        story_summary=memory.story_summary,
    )


def format_glossary(memory: SessionMemory) -> list[str]:
    """One "original → translated" line per entity; characters carry gender and notes."""
    lines = [
        f"{c.original_name} → {c.translated_name} ({c.gender}{', ' + c.notes if c.notes else ''})"
        for c in memory.characters
    ]
    lines += [f"{e.original} → {e.translated}" for e in memory.places + memory.organizations]
    return lines


def format_memory_for_prompt(memory: SessionMemory) -> str:
    has_entities = any([memory.characters, memory.places, memory.organizations])
    if not has_entities and not memory.story_summary:
        return ""

    parts = []

    if has_entities:
        parts.append("Known entities (use these translations consistently):")
        sections = [
            ("Characters", SessionMemory(characters=memory.characters)),
            ("Places", SessionMemory(places=memory.places)),
            ("Organizations", SessionMemory(organizations=memory.organizations)),
        ]
        for title, section in sections:
            lines = format_glossary(section)
            if lines:
                parts.append(f"{title}:")
                parts.extend(f"- {line}" for line in lines)

    if memory.story_summary:
        if parts:
            parts.append("")  # blank line separator
        parts.append(f"Story so far: {memory.story_summary}")

    return "\n".join(parts)
