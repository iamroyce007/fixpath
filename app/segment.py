"""SIIS article text -> sections, sentences, atomic step candidates and Settings paths.

Everything here is deterministic. Sentence ids (S1, S2, ...) are stable for a given text,
so the LLM can refer to sentences by id and code maps them back to verbatim text.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

PREFIX_RE = re.compile(r"^[^\n]*?\([^()\n]*\)\s*:\s*")
HEADING_RE = re.compile(r"^\s*(#{1,6})\s*(.*)$")
STEP_HEADING_RE = re.compile(r"^\s*(?:step\s*\d+\s*[:.)-]|\d+\.\s+[A-Z])", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|(?<=[a-z][.!?])(?=[A-Z][a-z])")

IMPERATIVE_VERBS = {
    "add", "adjust", "back", "charge", "check", "choose", "clean", "clear", "close", "confirm",
    "connect", "contact", "disable", "disconnect", "drag", "enable", "ensure", "enter", "examine",
    "exit", "force", "go", "hold", "increase", "insert", "inspect", "install", "keep", "launch",
    "locate", "make", "navigate", "open", "perform", "place", "plug", "power", "press", "reconnect",
    "reinsert", "remove", "reset", "restart", "review", "scan", "schedule", "search", "select",
    "send", "set", "shine", "swipe", "switch", "tap", "touch", "try", "turn", "uninstall", "unplug",
    "update", "use", "verify", "visit", "wipe", "back up", "decrease", "reboot", "boot",
}
LEAD_INS = re.compile(
    r"^(?:(?:first|next|then|now|alternatively|finally|also|afterward|afterwards|please|simply|"
    r"just|and)\b[,]?\s*)+",
    re.I,
)
# "On devices with a Power button: Press ..." / "For Smart View: Swipe ..."
LABEL_RE = re.compile(r"^[A-Z][^:.]{0,60}:\s+(?=[A-Z])")
CONDITIONAL_RE = re.compile(r"^(?:if|to|when|once|after|for|before|while|in this case)\b[^,]*,\s*", re.I)
CLAUSE_SPLIT_RE = re.compile(r",?\s+(?:and\s+then|and|then)\s+|,\s+(?=(?:and\s+)?(?:then\s+)?[a-z]+\b)", re.I)

CRITICAL_RE = re.compile(
    r"\b(factory (data )?reset|reset to factory|restart|reboot|safe mode|firmware|software update|"
    r"system update|update (the |your )?(device )?software|wipe (all|data|the device)|master reset)\b",
    re.I,
)
MANUAL_RE = re.compile(
    r"\b(service cent(er|re)|customer support|contact|repair|replace|clean|cloth|charger|cable|"
    r"adapter|hdmi|flashlight|ejector|battery|inspect|examine|mouse|keyboard|monitor|screen protector|"
    r"physical damage|liquid|walk-in|mail-in|provider|pc|computer|remove any cases)\b",
    re.I,
)
PATH_ARROW_RE = re.compile(r"\bSettings\s*>\s*[^.]+", re.I)
PATH_START_RE = re.compile(r"\b(?:go to|navigate to(?: and open)?|open)\s+Settings\b", re.I)
PATH_HOP_RE = re.compile(
    r"\b(?:tap|select|touch)\s+(?:on\s+)?(?:the\s+)?(?:switch next to\s+)?([A-Z][\w'&/+-]*(?:\s+[\w'&/+-]+){0,4}?)"
    r"(?=\s*(?:,|\.|$|\s+and\b|\s+then\b|\s+to\b|\s+again\b|\s+when\b))"
)


@dataclass
class Sentence:
    id: str
    text: str
    section: str
    imperative: bool = False
    critical: bool = False
    manual: bool = False
    path: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)  # atomic step candidates


@dataclass
class Section:
    heading: str
    level: int
    sentence_ids: list[str] = field(default_factory=list)


@dataclass
class Article:
    title: str
    body: str
    sections: list[Section]
    sentences: list[Sentence]
    hash: str

    def by_id(self) -> dict[str, Sentence]:
        return {s.id: s for s in self.sentences}

    def step_sentences(self) -> list[Sentence]:
        return [s for s in self.sentences if s.steps]

    def to_dict(self) -> dict:
        return asdict(self)


def siis_hash(title: str, content: str) -> str:
    return hashlib.sha256(f"{title}\n{content}".encode("utf-8")).hexdigest()[:16]


def strip_prefix(content: str) -> str:
    return PREFIX_RE.sub("", content, count=1).strip()


def split_sentences(line: str) -> list[str]:
    return [p.strip() for p in SENT_SPLIT_RE.split(line) if p and p.strip()]


def _core(sentence: str) -> str:
    """Drop labels, lead-ins and conditional clauses so the imperative verb comes first."""
    s = sentence.strip().strip('"')
    for _ in range(3):
        before = s
        s = LABEL_RE.sub("", s)
        s = LEAD_INS.sub("", s)
        if not _starts_imperative(s):
            m = CONDITIONAL_RE.match(s)
            if m and _starts_imperative(s[m.end():]):
                s = s[m.end():]
        if s == before:
            break
    return s.strip()


def _first_words(s: str, n: int) -> str:
    return " ".join(re.findall(r"[A-Za-z']+", s.lower())[:n])


def _starts_imperative(s: str) -> bool:
    return _first_words(s, 1) in IMPERATIVE_VERBS or _first_words(s, 2) in IMPERATIVE_VERBS


def _finish(step: str) -> str:
    step = step.strip().strip(",;:").strip()
    if not step:
        return step
    step = step[0].upper() + step[1:]
    return step if step.endswith((".", "!", "?")) else step + "."


def atomic_steps(sentence: str) -> list[str]:
    """Split one imperative sentence into one-interaction steps.

    "Go to Settings, tap Display, and then tap Brightness." ->
    ["Go to Settings.", "Tap Display.", "Tap Brightness."]
    Only splits where every piece starts with an imperative verb, so meaning is never lost.
    """
    core = _core(sentence)
    if not core or not _starts_imperative(core) or core.endswith(":"):
        return []
    body = core.rstrip(".!? ")
    parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(body) if p and p.strip()]
    if len(parts) > 1 and all(_starts_imperative(p) and len(p.split()) >= 2 for p in parts):
        return [_finish(p) for p in parts]
    return [_finish(body)]


def settings_path(sentence: str) -> list[str]:
    m = PATH_ARROW_RE.search(sentence)
    if m:
        hops = [h.strip(" ,") for h in m.group(0).split(">")]
        return [re.split(r",|\band\b", h)[0].strip() for h in hops if h.strip()]
    if not PATH_START_RE.search(sentence):
        return []
    tail = sentence[PATH_START_RE.search(sentence).end():]
    hops = ["Settings"]
    for hop in PATH_HOP_RE.findall(tail):
        hop = re.sub(r"\s+(?:icon|option|menu)$", "", hop.strip())
        if hop and hop not in hops:
            hops.append(hop)
    return hops


def segment(title: str, content: str) -> Article:
    body = strip_prefix(content)
    sections: list[Section] = [Section(heading=title, level=0)]
    sentences: list[Sentence] = []
    n = 0
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        h = HEADING_RE.match(line)
        if h:
            if h.group(2).strip():
                sections.append(Section(heading=h.group(2).strip(), level=len(h.group(1))))
            continue
        if STEP_HEADING_RE.match(line) and len(line) < 80 and not line.endswith("."):
            sections.append(Section(heading=line, level=3))
            continue
        for text in split_sentences(line):
            if len(re.findall(r"[A-Za-z]", text)) < 3:
                continue
            n += 1
            sec = sections[-1]
            steps = atomic_steps(text)
            s = Sentence(
                id=f"S{n}",
                text=text,
                section=sec.heading,
                imperative=bool(steps),
                critical=bool(CRITICAL_RE.search(text)) or bool(CRITICAL_RE.search(sec.heading)),
                manual=bool(MANUAL_RE.search(text)),
                path=settings_path(text),
                steps=steps,
            )
            sec.sentence_ids.append(s.id)
            sentences.append(s)
    sections = [s for s in sections if s.sentence_ids]
    return Article(title=title, body=body, sections=sections, sentences=sentences,
                   hash=siis_hash(title, content))
