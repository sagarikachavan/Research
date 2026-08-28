"""
llm_ptt_parser.py
==================
Turns one PTT cell's free text into an ordered, classified list of items
using an LLM (OpenAI API) instead of regex.

    items, source = parse_ptt_items(machine, ptt_text, client, model=...)

`items` is a list of dicts:
    {
      "number":    "1.3.1",
      "title":     "Perform a port scan",
      "node_type": "State" | "Action",
      "status":    "completed" | "in_progress" | "to_do" | "unknown",
      "finding":   "<finding/result text, or None>",
    }

`source` tells you where the items came from:
    "llm"            - fresh LLM call, used directly
    "llm_cache"      - a previous fresh LLM call, read back from disk cache
    "fallback_regex" - the LLM call failed validation after all retries;
                        the deterministic regex parser in ptt_parser.py was
                        used instead so the row is never dropped

Why an LLM at all, and why keep a regex fallback
-------------------------------------------------
PTT cells are messy free text: status markers appear before or after the
payload (or not at all), some items are bare "{...}" data blocks nested
under a parent with no title of their own, and some items that carry a
"Findings: ..." payload (e.g. "Target IP - 10.129.X.X") are conceptually a
State, not an Action, purely by human judgement about what counts as
"something the pentester did". A single regex can't reliably make that
judgement call; an LLM can. But an API call can fail (network blip, rate
limit, model returns something that doesn't validate) and a 1900-row batch
job shouldn't crash or silently drop a row over one bad call, so any item
list that fails validation after `max_retries` attempts falls back to the
original regex parser (ptt_parser.parse_ptt + classify) instead.

Caching
-------
Every (model, machine, ptt_text) triple is cached to a JSON file under
.llm_cache/ keyed by its hash. build_input_json.py and generate_graphs.py
both parse the exact same CSV rows, so the cache means the *second* script
you run does zero additional API calls for rows already seen -- only the
first pays for them.
"""

import os
import json
import time
import hashlib
import pathlib

import ptt_parser  # deterministic fallback + is_valid_machine_name


# ========================================================================
# HYBRID MODE (parser structure + LLM classification, reconciled)
# ========================================================================
# Why hybrid, instead of picking one:
#   - Structure (splitting the PTT cell into numbered items; extracting
#     each item's title / status / payload) is NOT ambiguous in this
#     dataset -- ptt_parser._split_raw_items / _parse_item_block handle it
#     with a brace-balance scan, and it has been validated to never drop,
#     merge, or reorder an item. Letting an LLM redo that from scratch on
#     every row reintroduces exactly the failure modes a free-text
#     re-parse has: a skipped item, a merged item, a renumbered item, or a
#     title that quietly drifts from the source text. There's no upside
#     to paying for that risk when a deterministic pass already gets it
#     right for free.
#   - Classifying a payload-bearing item as State (contextual/identity
#     data) vs Action (something the pentester did) IS genuinely
#     ambiguous for titles the fixed regex/whitelist can't anticipate
#     (pentest terminology is huge and keeps growing). That's exactly the
#     kind of judgment call an LLM is good at, and exactly where it's
#     worth spending a call.
# So hybrid mode: run the deterministic parser for structure, then ask the
# LLM to classify ONLY the items that are actually ambiguous (has a
# payload, depth > 0) -- as a fixed-shape "here are the exact items,
# return State/Action for each `number`" call, not a free re-parse. The
# LLM cannot add, drop, merge, renumber, or reword an item because it
# never sees raw PTT text and its schema doesn't have room to -- see
# _validate_classification below, which requires the returned number set
# to equal the sent number set exactly.
#
# Reconciliation:
#   - Unambiguous items (no payload, or depth 0) are never sent to the
#     LLM at all -- they're always State, by both rule sets, so there's
#     nothing to adjudicate and no reason to spend a call.
#   - The identity/location whitelist (ptt_parser.NON_ACTION_LABEL_RE:
#     target IP, hostname, machine name, OS, domain, ...) is a HARD
#     override to State regardless of what the LLM returns. This is the
#     one class of error (turning "Target IP" into an Action) that the
#     dataset owner has said is unambiguously wrong, so it's enforced in
#     code rather than left to LLM judgment on a given call.
#   - Everywhere else, if the deterministic default and the LLM agree,
#     use it (no ambiguity). Where they disagree, the LLM's classification
#     wins (it has the semantic judgment the regex doesn't) -- but every
#     disagreement is logged to the hybrid run's diagnostics so the net
#     effect of the LLM pass is auditable in aggregate, instead of
#     requiring a per-row manual check.
# ========================================================================

CLASSIFY_SYSTEM_PROMPT = """You are a penetration-testing analyst reviewing items already extracted from a
"Pentesting Task Tree" (PTT). Each item below has a fixed number, title, and
payload (its "Findings" data) -- these are FIXED, do not change them. Your
only job is to decide, for each item, whether it is:

  "State"  - a contextual / informational fact ABOUT THE TARGET or the
             engagement -- its IP address, hostname, machine name, OS,
             domain, or a passively-reported field like SSH host key,
             HTTP server headers/titles, SSL certificate details, scan
             duration, service banner/version -- NOT something the
             pentester separately did. Example: idurar row 0 has "IP
             Address", "Host Status", "SSH Hostkey", "HTTP Server
             Headers", "HTTP Titles", "SSL Certificate (Port 443)", "OS",
             "Scan Duration" -- ALL of these are State, even though each
             carries a Findings payload, because none of them describes a
             distinct action the pentester took; they're all facts read
             off the target during one recon pass.
  "Action" - the title names a concrete action the pentester actually
             performed that produced this payload as a distinct result --
             "Perform a port scan", "Determine the services and versions",
             "Enumerate HTTP service", "Explore the directories",
             "Exploit the PHP files", "Check user privileges", "Obtain
             reverse shell", "Crack the password hash". Also covers a
             short noun phrase that clearly names a discrete task result
             rather than a passive report field, e.g. "Shares Enumerated"
             or "Decrypted Password" (something was actively done to get
             this, and it isn't a generic identity/report field).

You are only given items where a simple rule (leading-verb match) could
NOT already decide -- i.e. genuinely ambiguous cases. When unsure, prefer
"State": it is the safer default for a passively-reported fact, and the
worked idurar example above is the standard to calibrate against. Return
exactly one classification per item `number`, using the exact same
`number` strings you were given -- do not add, drop, merge, renumber, or
reinterpret items. Respond with JSON matching the schema only."""

CLASSIFY_JSON_SCHEMA = {
    "name": "ptt_classifications",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "string"},
                        "node_type": {"type": "string", "enum": ["State", "Action"]},
                    },
                    "required": ["number", "node_type"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["classifications"],
        "additionalProperties": False,
    },
}


def _validate_classification(obj, expected_numbers):
    if not isinstance(obj, dict) or not isinstance(obj.get("classifications"), list):
        raise ValueError("response missing a valid 'classifications' array")
    out = {}
    for c in obj["classifications"]:
        if not isinstance(c, dict):
            raise ValueError("classification entry is not an object")
        num = c.get("number")
        nt = c.get("node_type")
        if not isinstance(num, str) or not num.strip():
            raise ValueError("classification missing 'number'")
        if nt not in ("State", "Action"):
            raise ValueError(f"bad node_type {nt!r}")
        out[num.strip()] = nt
    got = set(out.keys())
    if got != expected_numbers:
        missing = expected_numbers - got
        extra = got - expected_numbers
        raise ValueError(
            f"classification number set mismatch (missing={sorted(missing)[:5]}, "
            f"extra={sorted(extra)[:5]})"
        )
    return out


def classify_ambiguous_items_with_llm(machine, ambiguous_items, client,
                                       model=None, max_retries=3):
    """Classify a fixed list of already-structurally-parsed items (each a
    dict with 'number', 'title', 'payload') as State/Action via LLM.

    Returns (dict number -> "State"/"Action", source) where source is
    "llm", "llm_cache", or "fallback_regex" (LLM call failed validation
    after all retries -- caller should keep the deterministic default for
    every item in this batch).
    """
    model = model or DEFAULT_MODEL
    if not ambiguous_items:
        return {}, "llm"

    expected_numbers = {it["number"] for it in ambiguous_items}
    payload_block = "\n".join(
        f"- number={it['number']!r} title={it['title']!r} payload={it['payload']!r}"
        for it in ambiguous_items
    )
    cache_key = _cache_key(f"classify:{model}", machine, payload_block)
    cached = _load_cache(cache_key, CACHE_DIR)
    if cached is not None:
        return cached["classifications"], "llm_cache"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Machine: {machine}\n\nItems:\n{payload_block}"},
                ],
                response_format={"type": "json_schema", "json_schema": CLASSIFY_JSON_SCHEMA},
            )
            raw = resp.choices[0].message.content
            result = _validate_classification(json.loads(raw), expected_numbers)
            _save_cache(cache_key, CACHE_DIR, {"classifications": result})
            return result, "llm"
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    print(f"  [warn] LLM classify failed for machine={machine!r} after {max_retries} "
          f"attempt(s) ({last_err!r}); kept deterministic classification for this batch.")
    return {}, "fallback_regex"


def parse_ptt_items_hybrid(machine, ptt_text, client, model=None,
                            max_retries=3, disagreement_log=None):
    """Structure from the deterministic parser, State/Action classification
    reconciled between the deterministic rule and an LLM pass over only the
    ambiguous (payload-bearing, depth>0) items. Returns (items, source)
    where items match the same schema build_graph_from_items expects.

    `disagreement_log`, if given a list, gets one dict appended per item
    where the LLM and the deterministic rule disagreed, so the net effect
    of the LLM pass is auditable across a whole run instead of requiring a
    per-row manual check.
    """
    parsed = ptt_parser.parse_ptt(ptt_text)

    ambiguous, resolved = [], {}
    for it in parsed:
        det = ptt_parser.classify(it)  # 'state' or 'action'
        if ptt_parser.is_ambiguous(it):
            # Neither the identity whitelist, the leading-verb rule, nor
            # the reporting-field denylist could decide -- worth an LLM
            # call. Everything else (identity fields, clear verb-led
            # actions, clear report-field nouns, no-payload/top-level
            # items) is resolved deterministically with high confidence
            # and never sent to the LLM.
            ambiguous.append(it)
        else:
            resolved[it["number"]] = "State" if det == "state" else "Action"

    llm_labels, source = ({}, "llm") if not ambiguous else \
        classify_ambiguous_items_with_llm(machine, ambiguous, client, model, max_retries)

    for it in ambiguous:
        det_label = "State" if ptt_parser.classify(it) == "state" else "Action"
        llm_label = llm_labels.get(it["number"])
        if llm_label is None:
            final = det_label  # LLM pass failed for this batch -- keep deterministic default
        else:
            final = llm_label
            if llm_label != det_label and disagreement_log is not None:
                disagreement_log.append({
                    "machine": machine, "number": it["number"], "title": it["title"],
                    "payload_preview": (it["payload"] or "")[:120],
                    "deterministic": det_label, "llm": llm_label,
                })
        resolved[it["number"]] = final

    items = []
    for it in parsed:
        items.append({
            "number": it["number"], "title": it["title"],
            "node_type": resolved[it["number"]], "status": it["status"],
            "finding": it["payload"],
        })
    return items, source

CACHE_DIR = pathlib.Path(__file__).parent / ".llm_cache"
DEFAULT_MODEL = os.environ.get("PTT_LLM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a penetration-testing analyst. You will be given the raw text of one
cell from a "Pentesting Task Tree" (PTT) column of a dataset. The cell is a
numbered, indented outline describing the state of a pentest engagement at
one point in time (numbers like "1", "1.1", "1.3.2", "3.4" ...). Convert it
into a FLAT, ORDERED list of "items", one per numbered line, each
classified as a graph node.

Classify every numbered item using this decision order -- apply the rules
top to bottom, stop at the first one that applies:

  1. No finding/data payload attached to this item at all (no "{...}"
     block, no data of any kind) -> node_type = "State". ALWAYS. Even if
     the title reads like a command (e.g. "2.1 Identify vulnerability -
     (to-do)" with nothing else) -- nothing has been produced yet, so it's
     just the current phase/sub-phase, not a completed Action.
  2. A payload IS present, and the item is really just contextual /
     informational data about the target or the engagement rather than a
     distinct action the pentester performed -- a machine name, target IP
     address, hostname, OS, domain, or a passively-reported field like
     "Host Status", "SSH Hostkey", "HTTP Server Headers", "HTTP Titles",
     "SSL Certificate (Port 443)", "Scan Duration" -> node_type = "State",
     and the payload stays attached to that same State node (do not also
     emit it as a separate Finding). Worked example: idurar row 0's
     "1.4 IP Address", "1.5 Host Status", "1.6 SSH Hostkey", "1.7 HTTP
     Server Headers", "1.8 HTTP Titles", "1.9 SSL Certificate (Port 443)",
     "1.10 OS", "1.11 Scan Duration" are ALL State -- every one of them
     carries a Findings payload, but none names a distinct action; they
     are all facts read off the target during one recon pass. When a
     payload-bearing item's title has no leading action verb and isn't
     itself a task result, default to State.
  3. A payload IS present, and the item's title clearly names a concrete
     action the pentester actually performed that produced this result --
     "Perform a port scan", "Determine the services and versions",
     "Enumerate HTTP service", "Explore the directories", "Exploit the PHP
     files", "Check user privileges", "Obtain reverse shell" -> node_type
     = "Action", and the payload becomes that item's `finding` (a separate
     Finding node downstream).

status: one of "completed", "in_progress", "to_do", "unknown" -- read from
  markers like (completed) / [completed], (to-do) / [to-do], (in progress)
  / [in progress], or {Status: ...}, wherever in the item they appear. If no
  status marker exists anywhere for that item, use "unknown".

finding: the result/data text attached to that item (commonly inside a
  "{...}" block, sometimes labelled "Findings:"), with the "Findings:"
  label itself stripped out of the returned text. Use null if the item has
  no such payload of its own.

Important edge cases:
  * Nesting depth alone never determines node_type -- judge each item by
    what it actually describes, per the rules above. A Target IP / IP
    address / machine-name item that has ITS OWN number (e.g. "1.4 Target
    IP: {Findings: 10.129.X.X}", "1.9.1 IP Address - {Findings: ...}") is
    ALWAYS its own separate State node -- never fold it into a sibling or
    parent just because it looks like a short data field.
  * The ONLY time you should fold an item into another is when it is a
    truly BARE data block with NO label/title text of its own whatsoever
    -- i.e. the line is nothing but a number followed directly by a
    "{...}" block, e.g. "1.3.2.1 {Target IP: ..., Findings: ...}" -- AND
    it sits exactly one numbering level deeper than the immediately
    preceding item, AND that preceding item itself has no payload yet. In
    that specific case only, merge the bare child's data into the
    `finding` field of that immediate parent instead of emitting a
    separate item. This is rare -- when in doubt, do NOT fold; give the
    item its own entry (rule 1 or 2 above will classify it correctly as a
    State either way).
  * Preserve document order. Every numbered item in the input maps to
    exactly one output item, except a bare child folded per the rule
    directly above.
  * Keep `title` short and clean: strip the leading item number, status
    markers, and the payload/braces out of it.

Respond with a JSON object matching the given schema only -- no commentary."""

ITEMS_JSON_SCHEMA = {
    "name": "ptt_items",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "string"},
                        "title": {"type": "string"},
                        "node_type": {"type": "string", "enum": ["State", "Action"]},
                        "status": {
                            "type": "string",
                            "enum": ["completed", "in_progress", "to_do", "unknown"],
                        },
                        "finding": {"type": ["string", "null"]},
                    },
                    "required": ["number", "title", "node_type", "status", "finding"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}


# ----------------------------------------------------------------------
# OpenAI client
# ----------------------------------------------------------------------
def get_openai_client():
    """Instantiate an OpenAI client from the OPENAI_API_KEY env var.

    Import is local so the rest of this module (and anything that only
    needs the regex fallback / cache reading) works even in environments
    where the `openai` package isn't installed.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export your OpenAI token first, e.g.:\n"
            "    export OPENAI_API_KEY=sk-...\n"
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit(
            "The 'openai' package isn't installed. Run:\n"
            "    pip install --upgrade openai\n"
        ) from e
    return OpenAI(api_key=api_key)


# ----------------------------------------------------------------------
# Disk cache
# ----------------------------------------------------------------------
def _cache_key(model, machine, ptt_text):
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update((machine or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((ptt_text or "").encode("utf-8"))
    return h.hexdigest()


def _cache_path(key, cache_dir):
    return pathlib.Path(cache_dir) / f"{key}.json"


def _load_cache(key, cache_dir):
    p = _cache_path(key, cache_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(key, cache_dir, payload):
    p = _cache_path(key, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # atomic, safe across the thread pool


# ----------------------------------------------------------------------
# Regex fallback (reuses the original parser in ptt_parser.py)
# ----------------------------------------------------------------------
def _fallback_items(ptt_text):
    parsed = ptt_parser.parse_ptt(ptt_text)
    items = []
    for it in parsed:
        node_type = "State" if ptt_parser.classify(it) == "state" else "Action"
        items.append({
            "number": it["number"], "title": it["title"], "node_type": node_type,
            "status": it["status"], "finding": it["payload"],
        })
    return items


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def _validate_items(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
        raise ValueError("response missing a valid 'items' array")

    out = []
    for it in obj["items"]:
        if not isinstance(it, dict):
            raise ValueError("item is not an object")
        for k in ("number", "title", "node_type", "status"):
            if not isinstance(it.get(k), str) or not it[k].strip():
                raise ValueError(f"item missing/invalid '{k}'")
        if it["node_type"] not in ("State", "Action"):
            raise ValueError(f"bad node_type {it['node_type']!r}")
        if it["status"] not in ("completed", "in_progress", "to_do", "unknown"):
            raise ValueError(f"bad status {it['status']!r}")
        finding = it.get("finding")
        if finding is not None and not isinstance(finding, str):
            raise ValueError("bad finding")
        out.append({
            "number": it["number"].strip(),
            "title": it["title"].strip(),
            "node_type": it["node_type"],
            "status": it["status"],
            "finding": finding.strip() if isinstance(finding, str) and finding.strip() else None,
        })

    if not out:
        raise ValueError("LLM returned an empty items list")
    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def parse_ptt_items(machine, ptt_text, client, model=DEFAULT_MODEL,
                     cache_dir=CACHE_DIR, max_retries=3, use_cache=True):
    ptt_text = ptt_text if isinstance(ptt_text, str) else ""
    if not ptt_text.strip():
        return [], "fallback_regex"

    key = _cache_key(model, machine, ptt_text)

    if use_cache:
        cached = _load_cache(key, cache_dir)
        if cached is not None:
            source = cached.get("source", "llm")
            return cached["items"], ("llm_cache" if source == "llm" else source)

    items, source, last_err = None, None, None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Machine: {machine}\n\nPTT cell:\n{ptt_text}"},
                ],
                response_format={"type": "json_schema", "json_schema": ITEMS_JSON_SCHEMA},
            )
            raw = resp.choices[0].message.content
            obj = json.loads(raw)
            items = _validate_items(obj)
            source = "llm"
            break
        except Exception as e:  # network error, rate limit, bad JSON, failed validation, ...
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    if items is None:
        items = _fallback_items(ptt_text)
        source = "fallback_regex"
        print(f"  [warn] LLM parse failed for machine={machine!r} after {max_retries} attempt(s) "
              f"({last_err!r}); used regex fallback instead.")

    if use_cache:
        _save_cache(key, cache_dir, {"items": items, "source": source})

    return items, source
