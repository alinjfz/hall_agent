import re
from typing import Literal

from pydantic import BaseModel


class SafetyCheck(BaseModel):
    hazard: Literal["fire", "gas", "fire_equipment", "building_hazard", "dark_route",
                    "unsafe_door", "lockout", "identifiable_key"] | None
    evidence: str | None
    assessment: Literal["current", "not_current", "uncertain"] | None = None


# Explicit signals take precedence over routine triage, even without a model.
SIGNALS = {
    "fire": (
        r"\b(?:on fire|flames?|(?:frying pan|pan|oven|hob) (?:has )?ignited|"
        r"(?:there(?: is|'s)|I (?:see|smell)) smoke|"
        r"smoke (?:is )?(?:coming|pouring|in (?:the |my ))|fire (?:has )?(?:started|broken out)|"
        r"(?:fire|smoke) alarm (?:is )?(?:going off|sounding|ringing|went off))\b"
    ),
    "gas": (
        r"\b(?:gas leak|smell(?:s|ing)? (?:of )?gas|carbon monoxide (?:leak|poisoning)|"
        r"(?:rotten eggs?|sulphur|sulfur)[^.!?;]{0,45}\b(?:gas (?:hob|meter)|boiler|gas supply)|"
        r"(?:gas (?:hob|meter)|boiler)[^.!?;]{0,45}\b(?:rotten eggs?|sulphur|sulfur)|"
        r"(?:carbon monoxide|CO) alarm (?:is )?(?:going off|sounding|went off))\b"
    ),
    "fire_equipment": (
        r"\b(?:(?:fire|smoke|carbon monoxide) (?:alarm|detector)|fire door|fire exit|escape route|emergency light)\b"
        r"[^.!?;]{0,55}\b(?:broken|faulty|chirp\w*|beep\w*|block\w*|wedged|won't|will not|doesn't work|not working)\b"
        r"|\b(?:blocked|obstructed) (?:fire exit|escape route)\b"
    ),
    "building_hazard": (
        r"\b(?:flood(?:ing|ed)|burst pipe|water (?:is )?(?:pouring|spreading|(?:coming|leaking|dripping) through (?:the |my )?ceiling)|"
        r"(?:total|complete) power (?:loss|outage|failure)|"
        r"no power (?:anywhere|in (?:the|our|my) (?:whole |entire )?(?:flat|building))|"
        r"sparks?(?![ -]+plug\b)|sparking|(?:outlet|socket|plug)[^.!?;]{0,45}(?:glow\w*|melt\w*)|burning (?:smell|odor|odour)|smell(?:s)? (?:of )?burning|exposed wires?|electric shock|"
        r"ceiling (?:is )?(?:collapsing|caving in)|trapped in (?:a |the )?(?:lift|elevator)|"
        r"water [^.!?;]{0,35}(?:into|onto|through) (?:an? |the )?(?:socket|fuse box|consumer unit|electrical panel)|"
        r"broken glass [^.!?;]{0,25}(?:stairs|corridor|shared hallway))\b"
    ),
    "dark_route": (
        r"\b(?:stairs?|stairwell|corridor|shared hallway|(?:fire |emergency )?exit)\b"
        r"[^.!?;]{0,55}\b(?:pitch black|(?:completely |totally )?dark|no (?:working )?lights?|all (?:the )?lights (?:are )?out)\b"
    ),
    "unsafe_door": (
        r"\b(?:door|lock)\b[^.!?;]{0,40}\b(?:won't|will not|doesn't|cannot) "
        r"(?:latch|lock|secure|open from (?:the )?inside)\b|\b(?:trapped|locked) inside\b"
    ),
    "identifiable_key": (
        r"\b(?:lost|stolen|missing)\b[^.!?;]{0,70}\b(?:key|keyring|fob)\b"
        r"[^.!?;]{0,55}\b(?:room number|room-number|address|building name|identifying tag)\b"
        r"|\b(?:key|keyring|fob)\b[^.!?;]{0,60}\b(?:room number|room-number|address|building name|identifying tag)\b"
        r"[^.!?;]{0,40}\b(?:lost|stolen|missing)\b"
    ),
    "lockout": (
        r"\b(?:locked out|(?:can't|cannot) (?:get (?:back )?in(?! touch| contact)|enter|unlock (?:my|our|the) door)|"
        r"(?:fob|key card|keycard) (?:won't|will not) let me in)\b"
    ),
}


def noncurrent_mention(name, prefix, suffix):
    # Suppress mentions tied to a reference or hypothetical question, one match at a time.
    if name == "fire" and re.search(
        r"\b(?:photo(?:graph)?|picture|video)\b[^.!?;]{0,55}\b(?:shows?|depicts?)\s*$", prefix, re.I
    ):
        return True
    if name == "gas" and re.search(r"\b(?:what (?:to do|should I do)|what happens)\s+if\s+(?:I|we)\s*$",
                                   prefix, re.I):
        return not suffix.strip() or bool(re.fullmatch(r"\s+in (?:the )?future\s*", suffix, re.I))
    return name == "building_hazard" and bool(
        re.search(r"\b(?:diagram|manual|sign)\b[^.!?;]{0,30}\b(?:says?|reads?)\s+['\"]$", prefix, re.I)
        and re.match(r"['\"]\s+as a warning\b", suffix, re.I)
    )


def safety_signals(message):
    normalized = message.replace("’", "'")
    clauses = re.split(r"[.!?;\n]+|\b(?:but|however)\b", normalized, flags=re.I)
    confirmed_drill = re.search(
        r"\b(?:staff confirmed|confirmed by staff)\b[^.!?;]{0,35}\bdrill\b"
        r"|\bdrill\b[^.!?;]{0,35}\bconfirmed by staff\b", normalized, re.I
    )
    resolved = re.search(
        r"\b(?:staff|maintenance|engineers?)\b[^.!?;]{0,80}\b(?:confirmed|said)\b[^.!?;]{0,45}\bsafe\b",
        normalized, re.I,
    )
    historical = re.compile(r"\b(?:last (?:week|night|month)|yesterday|earlier|previously)\b", re.I)
    signals = set()
    for name, pattern in SIGNALS.items():
        for clause in clauses:
            for match in re.finditer(pattern, clause, re.I):
                if name == "fire" and confirmed_drill and "alarm" in match.group().lower():
                    continue
                start = normalized.find(clause)
                if (resolved and start < resolved.start() <= start + len(clause) + 80
                        and historical.search(clause)) and not re.search(
                    r"\b(?:now|again|still|currently)\b", clause, re.I
                ):
                    continue
                prefix = clause[:match.start()]
                if noncurrent_mention(name, prefix, clause[match.end():]):
                    continue
                if not re.search(r"\b(?:no|not|never|without)\s+(?:(?:any|a|an|signs? of|currently)\s+)*$", prefix, re.I):
                    signals.add(name)
                    break
    if "lockout" in signals:
        resolution = list(re.finditer(
            r"\b(?:I(?:'m| am)|we(?:'re| are)) (?:safely )?(?:back inside|back in now|inside now)\b",
            normalized, re.I
        ))
        if resolution and not re.search(SIGNALS["lockout"], normalized[resolution[-1].end():], re.I):
            signals.discard("lockout")
    return signals
