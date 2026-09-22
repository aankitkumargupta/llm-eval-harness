"""
Declare your own benchmark from the browser.

The Evaluate screen already walks someone from "I have data" to a runnable
*profile*. This is the same walk for a *benchmark*: the shape of the rows with
samples to copy, a spec generated from a handful of choices and validated
before it is shown, a row checker that names the line that is wrong, and a
writer that puts the two files where `harness bench` already looks.

It writes exactly two things and nothing else:

    config/benchmarks/<id>.yaml     the spec, every line commented
    data/benchmarks/<id>/items.jsonl   the rows

There is no third thing, no adapter module and no registry edit, because the
spec says `adapter: custom` and the generic adapter reads it
(`harness/bench/adapters/custom.py`). That is the whole point: I11 says adding
a benchmark is YAML, and anything that still needed Python meant the seam was
in the wrong place.

Nothing here runs a model or touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..bench.adapters.custom import SUPPORTED_FAMILIES
from ..bench.spec import BenchmarkSpec, SpecError
from .api import ApiError
from .scaffold import MAX_UPLOAD_CHARS, _check_name, _target

# --------------------------------------------------------------------------- #
#  The four shapes, and what each one needs
# --------------------------------------------------------------------------- #
SHAPES = [
    ("multiple_choice", "Multiple choice",
     "A question and a list of options. Scored on the option letter, with "
     "chance-adjusted accuracy beside the raw number."),
    ("short_answer", "Short answer",
     "A question and a reference answer. Scored on an exact match after case "
     "and punctuation are normalised; a number on either side is compared as "
     "a number."),
    ("math", "Number",
     "A question whose answer is a number. Scored by numeric equivalence, so "
     "1,000 and 1000 and 1000.0 are one answer."),
    ("classify", "Label",
     "A text and exactly one label from a list you declare. A reply that is "
     "not one of your labels is a format failure, not a wrong label."),
]

FIELDS = {
    "multiple_choice": [
        ("id", False, "Unique id for the row. The paired statistics join models on it. Generated when absent."),
        ("question", True, "What the model is asked."),
        ("options", True, "The choices, as a list of strings. Two or more."),
        ("answer", True, "The right option: its letter (B), its number counting from 1 (2), or the option text itself."),
        ("meta", False, "Free-form object. A `language` key gives a per-language reading."),
    ],
    "short_answer": [
        ("id", False, "Unique id for the row. Generated when absent."),
        ("question", True, "What the model is asked."),
        ("answer", True, "The reference answer."),
        ("meta", False, "Free-form object. A `language` key gives a per-language reading."),
    ],
    "math": [
        ("id", False, "Unique id for the row. Generated when absent."),
        ("question", True, "The problem."),
        ("answer", True, "The final number. Units and commas are normalised away before comparison."),
        ("meta", False, "Free-form object."),
    ],
    "classify": [
        ("id", False, "Unique id for the row. Generated when absent."),
        ("question", True, "The text to label."),
        ("answer", True, "The right label. Must be one of the labels you declare."),
        ("meta", False, "Free-form object."),
    ],
}

SAMPLES = {
    "multiple_choice": [
        {"id": "q001", "question": "Which authority hears a second appeal under the Right to Information Act, 2005?",
         "options": ["The Public Information Officer", "The first appellate authority",
                     "The State Information Commission", "The district court"],
         "answer": "C", "meta": {"language": "en"}},
        {"id": "q002", "question": "जन्म प्रमाणपत्र किस निकाय द्वारा जारी किया जाता है?",
         "options": ["नगर निगम", "उच्च न्यायालय", "भारतीय रिज़र्व बैंक", "निर्वाचन आयोग"],
         "answer": 1, "meta": {"language": "hi"}},
        {"id": "q003", "question": "A complaint about a broken street light is addressed to which body?",
         "options": ["The State Election Commission", "The Municipal Corporation",
                     "The Reserve Bank of India", "The High Court"],
         "answer": "The Municipal Corporation", "meta": {"language": "en"}},
    ],
    "short_answer": [
        {"id": "s001", "question": "Within how many days must a Public Information Officer reply to an application?",
         "answer": "30 days", "meta": {"language": "en"}},
        {"id": "s002", "question": "Which department issues a caste certificate in Maharashtra?",
         "answer": "The Revenue Department", "meta": {"language": "en"}},
        {"id": "s003", "question": "शिधापत्रिका कोणत्या विभागाकडून दिली जाते?",
         "answer": "अन्न व नागरी पुरवठा विभाग", "meta": {"language": "mr"}},
    ],
    "math": [
        {"id": "m001", "question": "A household receives 5 kg of grain per member each month. A family of four collects its grain for three months. How many kilograms is that?",
         "answer": "60"},
        {"id": "m002", "question": "An application fee is Rs 10 and a certified copy costs Rs 2 per page. What does an application with 15 pages cost in rupees?",
         "answer": "40"},
        {"id": "m003", "question": "A scheme pays Rs 1,200 a month. What is the annual amount in rupees?",
         "answer": "14400"},
    ],
    "classify": [
        {"id": "t001", "question": "The water connection in our lane has been cut for four days and nobody answers the ward office.",
         "answer": "water_supply", "meta": {"language": "en"}},
        {"id": "t002", "question": "रस्त्यावरील दिवे गेल्या आठवड्यापासून बंद आहेत.",
         "answer": "street_lighting", "meta": {"language": "mr"}},
        {"id": "t003", "question": "My pension has not been credited for two months.",
         "answer": "pension", "meta": {"language": "en"}},
    ],
}

#: Per shape: the extraction chain, the system prompt and the decoding budget.
#: These decide what a number means, so they are written into the spec in full
#: rather than defaulted silently somewhere in the code.
DEFAULTS = {
    "multiple_choice": {
        "chain": [{"kind": "regex", "pattern": r"Answer:\s*\(?([A-J])\)?"},
                  {"kind": "last_capital_letter"}],
        "system": ("Answer the multiple-choice question. Reply with the option "
                   "letter, ending your response with 'Answer: X'."),
        "max_tokens": 1024, "options_per_item": 4,
    },
    "short_answer": {
        "chain": [{"kind": "regex", "pattern": r"Answer:\s*(.+)"},
                  {"kind": "first_line"}],
        "system": ("Answer the question directly and briefly. Give the answer "
                   "only, with no explanation and no preamble."),
        "max_tokens": 1024,
    },
    "math": {
        "chain": [{"kind": "boxed"}, {"kind": "numeric"}],
        "system": ("Solve the problem. Reason if you need to, then end your "
                   "response with the final number on its own line as "
                   "'Answer: <number>'."),
        "max_tokens": 2048,
    },
    "classify": {
        "chain": [{"kind": "label_set"}, {"kind": "first_line"}],
        "system": ("Classify the text into exactly one label from the allowed "
                   "list. Output only the label, in lowercase, with nothing else."),
        "max_tokens": 512,
        "labels": ["water_supply", "street_lighting", "pension", "other"],
    },
}

NOTES = {
    "multiple_choice": [
        "Chance level is not cosmetic: 25% on four options is not '25% good'. The spec records 1/options, and the report prints accuracy above guessing beside the raw number.",
        "Write the answer however your file already has it, a letter, a number counting from 1, or the option text. All three resolve to the same option, and a row whose answer names no option is refused at load rather than scored zero.",
        "Sixty or more items before the paired test can separate models a few points apart; below that the honest output is 'not separable at this n'.",
    ],
    "short_answer": [
        "Scoring is exact after case, surrounding punctuation and repeated spaces are normalised. It is deliberately not fuzzy: a near miss counted as a hit is the same lie as a parse failure counted as a miss, pointing the other way.",
        "If your answers are long or free-form, this shape will under-count. Use a profile with the judge scorer instead, the Evaluate screen's 'Generate or transform' path.",
    ],
    "math": [
        "Units, commas and trailing full stops are normalised before comparison, so 'Rs 1,000.' and '1000' agree. Put only the number in `answer`.",
        "Reasoning models spend the token budget thinking before the visible answer. The budget here is 2048 for that reason; a small budget truncates before the first character and reads as accuracy zero.",
    ],
    "classify": [
        "Every answer must be one of your labels. A row with a label outside the set is refused at load, because it could never be reached and would score zero for every model.",
        "A reply that is not one of your labels is reported as a format failure, separately from a wrong label. The two need different fixes.",
        "Eight to ten rows per label is the floor for a readable per-label picture.",
    ],
}


def _shape(shape: str) -> str:
    if shape not in SAMPLES:
        raise ApiError(f"Unknown shape {shape!r}; one of {sorted(SAMPLES)}.", 400)
    return shape


# --------------------------------------------------------------------------- #
#  What the screen shows
# --------------------------------------------------------------------------- #
def describe(shape: str) -> dict:
    """The row format, sample rows, defaults and notes for one shape."""
    shape = _shape(shape)
    d = DEFAULTS[shape]
    return {
        "shape": shape,
        "shapes": [{"id": i, "title": t, "body": b} for i, t, b in SHAPES],
        "fields": [{"name": n, "required": r, "meaning": m} for n, r, m in FIELDS[shape]],
        "samples": SAMPLES[shape],
        "samples_jsonl": "\n".join(json.dumps(r, ensure_ascii=False)
                                   for r in SAMPLES[shape]) + "\n",
        "defaults": dict(d),
        "notes": NOTES[shape],
        "families": list(SUPPORTED_FAMILIES),
    }


# --------------------------------------------------------------------------- #
#  The spec
# --------------------------------------------------------------------------- #
def _yq(s) -> str:
    """JSON strings are valid YAML, so this quotes anything safely."""
    return json.dumps(s, ensure_ascii=False)


def spec_dict(bid: str, shape: str, options: dict | None = None) -> dict:
    """The spec as data, the single source both the YAML and the check use."""
    shape = _shape(shape)
    o = dict(options or {})
    d = DEFAULTS[shape]
    # The screen sends labels as one comma-separated string; the CLI and the
    # tests send a list. Iterating a string here would make every label one
    # character long, and every row's gold answer unreachable.
    raw_labels = o.get("labels") or d.get("labels") or []
    if isinstance(raw_labels, str):
        raw_labels = raw_labels.replace("\n", ",").split(",")
    labels = [str(x).strip() for x in raw_labels if str(x).strip()]

    if shape == "multiple_choice":
        n = int(o.get("options_per_item") or d["options_per_item"])
        chance = round(1.0 / n, 6) if n > 1 else 0.0
    elif shape == "classify":
        chance = round(1.0 / len(labels), 6) if len(labels) > 1 else 0.0
    else:
        chance = 0.0

    chain = [dict(link) for link in d["chain"]]
    if shape == "classify":
        for link in chain:
            if link.get("kind") == "label_set":
                link["labels"] = labels

    raw = {
        "id": bid,
        "version": int(o.get("version") or 1),
        "family": shape,
        "task": "classify" if shape == "classify" else "direct",
        "adapter": "custom",
        "source": {
            "kind": "local",
            "ref": o.get("data_path") or f"data/benchmarks/{bid}/items.jsonl",
            "split": "test",
            "licence": str(o.get("licence") or "in-house, not redistributed"),
            "commercial_use": bool(o.get("commercial_use", True)),
            "citation": str(o.get("citation") or ""),
        },
        "sampling": {"limit": o.get("limit"), "seed": int(o.get("seed") or 1729)},
        "prompt": {
            "few_shot": {"n": int(o.get("few_shot") or 0)},
            "system": o.get("system") or d["system"],
            "decoding": {"temperature": float(o.get("temperature") or 0.0),
                         "max_tokens": int(o.get("max_tokens") or d["max_tokens"]),
                         "top_p": 1.0},
        },
        "scoring": {"mode": "generative", "extraction": {"chain": chain},
                    "metric": "accuracy", "chance_level": chance},
        "reporting": {"primary_metric": "accuracy",
                      "also": ["extraction_failure_rate", "format_violation_rate",
                               "cost_usd", "latency_p95_ms"]},
    }
    if labels:
        raw["label_set"] = labels
    return raw


def spec_yaml(bid: str, shape: str, options: dict | None = None) -> str:
    """The spec as a commented YAML file, the thing actually written to disk."""
    bid = _check_name(bid)
    raw = spec_dict(bid, shape, options)
    src, smp, pr, sc = raw["source"], raw["sampling"], raw["prompt"], raw["scoring"]
    chain = "\n".join(f"      - {json.dumps(link, ensure_ascii=False)}"
                      for link in sc["extraction"]["chain"])
    lines = [
        f"# {bid}: a benchmark you declared, not one that shipped.",
        "# This file plus the JSONL it points at is the whole benchmark: the generic",
        "# adapter reads it, so there is no Python to write and nothing to register.",
        "# Bump `version` on ANY change below; it feeds spec_hash, and two runs with",
        "# different hashes are never compared.",
        f"id: {bid}",
        f"version: {raw['version']}",
        f"family: {raw['family']}                 # decides which metrics are meaningful",
        f"task: {raw['task']}",
        "adapter: custom                  # read by harness/bench/adapters/custom.py",
        "",
        "source:",
        "  kind: local                    # your file; nothing is fetched",
        f"  ref: {_yq(src['ref'])}",
        f"  split: {src['split']}",
        f"  licence: {_yq(src['licence'])}",
        f"  commercial_use: {str(src['commercial_use']).lower()}",
        f"  citation: {_yq(src['citation'])}",
        "",
        "sampling:",
        f"  limit: {'null' if smp['limit'] in (None, '') else int(smp['limit'])}   # null runs every row",
        f"  seed: {smp['seed']}                     # sampling is a pure function of (seed, file)",
        "",
        "prompt:",
        f"  few_shot: {{n: {pr['few_shot']['n']}}}",
        f"  system: {_yq(pr['system'])}",
        f"  decoding: {{temperature: {pr['decoding']['temperature']}, "
        f"max_tokens: {pr['decoding']['max_tokens']}, top_p: {pr['decoding']['top_p']}}}",
        "",
        "scoring:",
        f"  mode: {sc['mode']}",
        "  extraction:",
        "    chain:                       # first match wins; no match is a failure,",
        "                                 # reported beside accuracy, never inside it",
        chain,
        f"  metric: {sc['metric']}",
        f"  chance_level: {sc['chance_level']}            # accuracy above guessing uses this",
        "",
        "reporting:",
        f"  primary_metric: {raw['reporting']['primary_metric']}",
        f"  also: {json.dumps(raw['reporting']['also'])}",
    ]
    if raw.get("label_set"):
        lines += ["", f"label_set: {json.dumps(raw['label_set'], ensure_ascii=False)}"]
    return "\n".join(lines) + "\n"


def preview(bid: str, shape: str, options: dict | None = None) -> dict:
    """The YAML plus every problem with it, so the screen can validate as you type."""
    try:
        bid = _check_name(bid)
    except ApiError as e:
        return {"yaml": "", "errors": [e.message if hasattr(e, "message") else str(e)]}
    text = spec_yaml(bid, shape, options)
    errors: list[str] = []
    spec_hash = ""
    try:
        spec = BenchmarkSpec.from_dict(spec_dict(bid, shape, options))
        spec_hash = spec.spec_hash()
    except SpecError as e:
        errors = [ln.strip(" -") for ln in str(e).splitlines()[1:]] or [str(e)]
    return {"yaml": text, "errors": errors, "spec_hash": spec_hash,
            "spec_path": f"configs/benchmarks/{bid}.yaml",
            "data_path": f"data/benchmarks/{bid}/items.jsonl"}


# --------------------------------------------------------------------------- #
#  The rows
# --------------------------------------------------------------------------- #
def validate_jsonl(shape: str, text: str, *, labels: list[str] | None = None) -> dict:
    """Check the rows line by line. Content problems come back as a list naming
    the line rather than as an exception, so the whole file can be fixed once."""
    shape = _shape(shape)
    if len(text) > MAX_UPLOAD_CHARS:
        raise ApiError("File too large: 20 MB is the cap for a browser upload; "
                       "place larger files on disk and point `source.ref` at them.", 413)
    problems: list[str] = []
    seen: set[str] = set()
    rows = 0
    label_set = {str(x) for x in (labels or [])}
    langs: dict[str, int] = {}

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if len(problems) >= 20:
            problems.append("... more problems not shown")
            break
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {lineno}: invalid JSON ({e.msg} at column {e.colno})")
            continue
        if not isinstance(r, dict):
            problems.append(f"line {lineno}: each line must be one JSON object")
            continue
        rows += 1

        rid = str(r.get("id") or f"item_{lineno}")
        if rid in seen:
            problems.append(f"line {lineno}: duplicate id '{rid}' (paired statistics join on it)")
        seen.add(rid)

        if not isinstance(r.get("question"), str) or not r["question"].strip():
            problems.append(f"line {lineno}: 'question' must be a non-empty string")

        answer = r.get("answer")
        if answer is None or str(answer).strip() == "":
            problems.append(f"line {lineno}: 'answer' is required; a row with no reference answer cannot be scored")

        if shape == "multiple_choice":
            opts = r.get("options")
            if not isinstance(opts, list) or len(opts) < 2:
                problems.append(f"line {lineno}: 'options' must be a list of two or more strings")
            elif answer is not None:
                letters = [chr(ord("A") + i) for i in range(len(opts))]
                a = str(answer).strip()
                texts = [str(x).strip().casefold() for x in opts]
                reachable = (a.upper() in letters
                             or (a.isdigit() and (1 <= int(a) <= len(opts) or 0 <= int(a) < len(opts)))
                             or a.casefold() in texts)
                if not reachable:
                    problems.append(
                        f"line {lineno}: answer '{a}' is not an option letter, an option "
                        f"number, or one of the options; it could never be reached")
        elif shape == "classify":
            if label_set and str(answer).strip() not in label_set:
                problems.append(
                    f"line {lineno}: answer '{answer}' is not one of the labels "
                    f"{sorted(label_set)}; declare it above or fix the row")
        elif shape == "math" and answer is not None:
            probe = str(answer).replace(",", "").replace("$", "").strip().rstrip(".")
            try:
                float(probe)
            except ValueError:
                problems.append(
                    f"line {lineno}: answer '{answer}' is not a number. This shape "
                    f"compares numbers; use the short-answer shape for text.")

        if "meta" in r and not isinstance(r["meta"], dict):
            problems.append(f"line {lineno}: meta must be an object")
        lang = str(((r.get("meta") or {}) if isinstance(r.get("meta"), dict) else {}).get("language", "") or "")
        if lang:
            langs[lang] = langs.get(lang, 0) + 1

    if rows == 0:
        problems.append("the file has no rows")
    return {"shape": shape, "rows": rows, "problems": problems,
            "languages": langs, "ok": not problems}


# --------------------------------------------------------------------------- #
#  Writing
# --------------------------------------------------------------------------- #
def write_data(bid: str, shape: str, text: str, *, labels: list[str] | None = None,
               overwrite: bool = False, root: str | Path = ".") -> dict:
    """Validate, then write data/benchmarks/<id>/items.jsonl. Never over an
    existing file unless asked."""
    bid = _check_name(bid)
    check = validate_jsonl(shape, text, labels=labels)
    if not check["ok"]:
        raise ApiError("The file has problems; nothing was written:\n"
                       + "\n".join(check["problems"]), 400)
    rootp = Path(root)
    target = _target(rootp, bid, f"data/benchmarks/{bid}/items.jsonl")
    if target.exists() and not overwrite:
        raise ApiError(f"data/benchmarks/{bid}/items.jsonl already exists; "
                       f"tick overwrite to replace it.", 409)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text if text.endswith("\n") else text + "\n",
                      encoding="utf-8", newline="\n")
    return {"path": f"data/benchmarks/{bid}/items.jsonl", "rows": check["rows"],
            "languages": check["languages"]}


def scaffold(bid: str, shape: str, options: dict | None = None, *,
             with_samples: bool = True, overwrite: bool = False,
             root: str | Path = ".") -> dict:
    """Write the spec and, when asked, the sample rows, so it runs at once."""
    bid = _check_name(bid)
    shape = _shape(shape)
    text = spec_yaml(bid, shape, options)
    # Validate before writing anything: a spec on disk that does not load is a
    # benchmark that fails at run time instead of at creation time.
    BenchmarkSpec.from_dict(spec_dict(bid, shape, options))

    rootp = Path(root)
    spec_path = _target(rootp, bid, f"configs/benchmarks/{bid}.yaml")
    if spec_path.exists() and not overwrite:
        raise ApiError(f"configs/benchmarks/{bid}.yaml already exists; choose "
                       f"another id or tick overwrite.", 409)
    written: list[str] = []
    if with_samples:
        data_path = _target(rootp, bid, f"data/benchmarks/{bid}/items.jsonl")
        if data_path.exists() and not overwrite:
            raise ApiError(f"data/benchmarks/{bid}/items.jsonl already exists; "
                           f"tick overwrite to replace it.", 409)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                     for r in SAMPLES[shape]),
                             encoding="utf-8", newline="\n")
        written.append(f"data/benchmarks/{bid}/items.jsonl")
        readme = _target(rootp, bid, f"data/benchmarks/{bid}/README.md")
        if not readme.exists():
            readme.write_text(
                f"# {bid}\n\nA benchmark declared from the Evaluate screen, shape "
                f"`{shape}`, with sample rows. Replace `items.jsonl` with your own "
                f"rows in the same format, then run:\n\n"
                f"    python main.py bench validate --benchmark {bid}\n"
                f"    python main.py bench run --benchmark {bid} --models <a,b> --limit 20\n",
                encoding="utf-8", newline="\n")
            written.append(f"data/benchmarks/{bid}/README.md")

    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(text, encoding="utf-8", newline="\n")
    written.insert(0, f"configs/benchmarks/{bid}.yaml")
    return {"id": bid, "shape": shape, "written": written, "yaml": text,
            "spec_path": f"configs/benchmarks/{bid}.yaml"}
