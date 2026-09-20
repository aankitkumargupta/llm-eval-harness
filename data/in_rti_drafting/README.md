# in_rti_drafting

Fifty-one fictional citizen situations, each with a reference Right to
Information application drafted for it. Built by `build_dataset.py`, which
is the source of truth; `evalset.jsonl` is its output.

**Illustrative only.** Every office, place, name, date and amount is
invented. The legal framework the references cite is the real Right to
Information Act, 2005 (Section 6(1) application, Section 6(3) transfer,
Section 7(1) thirty-day limit, Section 7(5) fee and BPL waiver, Section 8
exemptions, Section 19 first appeal), because a draft that does not cite it
is not usable. Nothing here is legal advice.

**Shape.** The situation is written as the citizen would describe it: a
pension that stopped, a road that broke up, a scholarship that never came,
a tender nobody can see. Roughly four in five are in English, the rest in
Hindi (Devanagari) or Hinglish. One situation calls for a first appeal
rather than a fresh application, and the reference reflects that.

**The reference.** Generated from a template so every gold draft has the
same required elements: the correct office, two to four specific
information points, the period, the citizenship statement, the fee, the
30-day clock, the transfer request, and a request to name the exemption and
the appellate authority if anything is refused. References are in English
for every item; a model that answers a Hindi situation in Hindi is judged
on content.

**Scoring.** A direct profile scored by the judge: `accuracy` (right
office, right records, against the reference), `completeness` (nothing
required is missing) and `answer_relevance` (this citizen's problem, not a
generic form).

**Regenerate.**

    python data/in_rti_drafting/build_dataset.py
