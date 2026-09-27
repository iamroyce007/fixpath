"""Complaint -> symptom fingerprint {component, symptom, trigger, condition, onset}.

Rules only, so it is fast and deterministic. The canonical string is used as the tier-2
cache key and to name the goal (topic) and title.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

DEVICE_RE = re.compile(
    r"\b(?:techcorp|samsung|galaxy|nexa|trifold|fold|flip|tab|ultra|plus|pro|x\d+|[as]\d{1,3}[a-z]?"
    r"(?:/[as]?\d{1,3}[a-z]?)*|g\d+)\b",
    re.I,
)
NUMBERED_RE = re.compile(r"(?:^|\s)\d+\.\s*[\"“]?")

SYNONYMS = [
    (r"\bdisplay\b", "screen"), (r"\bmonitor\b", "screen"), (r"\bblack\b", "blank"),
    (r"\bdark\b", "blank"), (r"\bwhite screen\b", "blank screen"), (r"\bno image\b", "blank"),
    (r"\bnothing (?:is )?(?:visible|loads|shows)\b", "blank"), (r"\bflash(?:es|ing)?\b", "flicker"),
    (r"\bflicker(?:s|ing)?\b", "flicker"), (r"\bcrack(?:ed|s)?\b", "crack"),
    (r"\blag(?:gy|s|ging)?\b", "lag"), (r"\bdelay(?:ed|s)?\b", "lag"), (r"\bslow\b", "lag"),
    (r"\bunresponsive\b", "not responding"), (r"\bdoesn'?t respond\b", "not responding"),
    (r"\bwon'?t\b", "will not"), (r"\bcan'?t\b", "cannot"), (r"\bdoesn'?t\b", "does not"),
    (r"\bmobile\b", "phone"), (r"\bsmartphone\b", "phone"), (r"\bhandset\b", "phone"),
]

COMPONENTS = [
    ("inner screen", r"\binner screen|\bfold(?:s|ed|ing)? (?:screen|display)|where it folds"),
    ("data transfer", r"\bdata transfer|\btransfer\w*|\bqr code"),
    ("email", r"\bemail|\bgmail"),
    ("charging", r"\bcharg\w*"),
    ("camera", r"\bcamera"),
    ("rotation", r"\brotat\w*"),
    ("floating menu", r"\bfloating|\bcircle|\bhover\w*"),
    ("touchscreen", r"\btouch\w*|\binputs?\b"),
    ("screen", r"\bscreen|\bdisplay"),
    ("power", r"\bturn (?:it )?on|\bpower\w*|\bboot\w*|\bstart up"),
]
SYMPTOMS = [
    ("crack", r"\bcrack"),
    ("half blank", r"\bhalf (?:blank|black)|one side"),
    ("flicker", r"\bflicker"),
    ("blank", r"\bblank|\bblue screen"),
    ("lag", r"\blag\b"),
    ("not responding", r"\bnot responding|\bdoes not respond|\bnot work\w*|\bstopped working"),
    ("distorted", r"\bdistort\w*"),
    ("small screen", r"\bsmall|\bnot fill|\bfull size|\bexpand"),
    ("unwanted overlay", r"\bremove it|\bfloating"),
    ("no power", r"\bwill not (?:turn on|start)|\bnot turn on"),
    ("poor visibility", r"\bhardly see|\bcannot see"),
]
TOPICS = {
    "crack": ("Cracked Screen", "Cracked screen repair"),
    "half blank": ("Partial Display", "Partial display failure"),
    "flicker": ("Screen Flicker", "Screen flicker fix"),
    "blank": ("Blank Screen", "Blank screen recovery"),
    "lag": ("Touchscreen Lag", "Touchscreen lag fix"),
    "not responding": ("Unresponsive Screen", "Unresponsive screen fix"),
    "distorted": ("Distorted Display", "Distorted display check"),
    "small screen": ("Screen Size", "Screen size settings"),
    "unwanted overlay": ("Floating Menu", "Floating menu removal"),
    "no power": ("Device Power", "Device power recovery"),
    "poor visibility": ("Screen Visibility", "Screen visibility issue"),
}
CONFIG_RE = re.compile(r"\b(i want to|how (?:do|can) i|how to|set up|change|remove it|turn off|enable)\b", re.I)
PROBLEM_RE = re.compile(r"\b(blank|flicker|crack|lag|not responding|distort|will not|cannot|broken|issue|problem)\b")
TRIGGER_RE = re.compile(r"\b(?:when(?:ever)?|while|after|every time)\s+([a-z ]{3,40}?)(?=[,.;]|\band\b|$)")
ONSET_RE = re.compile(r"\b(suddenly|by itself|on its own|after (?:an? )?update|new phone|new smartphone|"
                      r"right after|after about a month|again)\b")


DEVICE_VOCAB_RE = re.compile(
    r"\b(screen|display|touch\w*|phone|tablet|device|smartphone|mobile|battery|charg\w*|app|apps|"
    r"setting\w*|wi-?fi|bluetooth|camera|keyboard|notification\w*|volume|sound|button|restart|reboot|"
    r"crack\w*|flicker\w*|blank|black|lag\w*|frozen|freez\w*|crash\w*|update|email|gmail|transfer|"
    r"rotat\w*|brightness|pixel\w*|sensitivity|gesture\w*|navigation|sim|data|backup|storage|"
    r"turn on|turn off|power|boot\w*|signal|network|hotspot|fingerprint|lock)\b",
    re.I,
)


def is_device_query(query: str) -> bool:
    return bool(DEVICE_VOCAB_RE.search(query))


@dataclass(frozen=True)
class Fingerprint:
    component: str
    symptom: str
    trigger: str
    condition: str
    onset: str

    @property
    def key(self) -> str:
        return f"{self.component}|{self.symptom}"

    def to_dict(self) -> dict:
        return asdict(self) | {"key": self.key}


def strip_devices(text: str) -> str:
    text = DEVICE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(query: str) -> str:
    q = query.lower().replace("’", "'").replace("—", " ").replace("–", " ")
    q = NUMBERED_RE.sub(" ", q)
    q = strip_devices(q)
    for pat, rep in SYNONYMS:
        q = re.sub(pat, rep, q)
    q = re.sub(r"[^a-z0-9' ]+", " ", q)
    q = re.sub(r"\b(my|the|a|an|i|me|it|is|and|so|of|to|on|with|just)\b", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def split_complaints(query: str) -> list[str]:
    """'1. "A" 2. "B" 3. "C"' -> ["A", "B", "C"]; anything else -> [query]."""
    parts = [p.strip(' "“”') for p in re.split(r"(?:^|\s)\d+\.\s+", query) if p.strip(' "“”')]
    return parts if len(parts) > 1 else [query.strip()]


def _first(table, text: str, default: str) -> str:
    for name, rx in table:
        if re.search(rx, text):
            return name
    return default


def fingerprint(query: str) -> Fingerprint:
    n = normalize(query)
    full = query.lower()
    trig = TRIGGER_RE.search(normalize(query.replace(",", " , ")))
    onset = ONSET_RE.search(full)
    return Fingerprint(
        component=_first(COMPONENTS, n, "device"),
        symptom=_first(SYMPTOMS, n, "general"),
        trigger=(trig.group(1).strip() if trig else ""),
        condition="configuration" if CONFIG_RE.search(full) and not PROBLEM_RE.search(n) else "fault",
        onset=onset.group(1) if onset else "",
    )


def topic_for(fp: Fingerprint, fallback_title: str = "") -> tuple[str, str, str]:
    """-> (Topic In Title Case, sentence-case title of 2-3 words, kind)."""
    kind = "Configuration" if fp.condition == "configuration" or fp.symptom == "unwanted overlay" else "Troubleshooting"
    if fp.symptom in TOPICS:
        topic, title = TOPICS[fp.symptom]
        if fp.component == "inner screen" and fp.symptom in ("blank", "not responding"):
            topic, title = "Inner Screen", "Inner screen failure"
        elif fp.component == "data transfer" and fp.symptom == "blank":
            topic, title = "Data Transfer Display", "Data transfer screen"
        elif fp.component == "charging" and fp.symptom == "flicker":
            topic, title = "Charging Flicker", "Charging screen flicker"
        return topic, title, kind
    words = [w for w in re.findall(r"[A-Za-z]+", fallback_title) if w.lower() not in
             {"on", "a", "or", "your", "the", "smartphone", "tablet", "to", "with", "and", "use"}][:3]
    topic = " ".join(w.capitalize() for w in words) or "Device"
    title = " ".join(words).capitalize() if len(words) >= 2 else f"{topic} issues"
    return topic, title, kind
