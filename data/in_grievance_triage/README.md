# in_grievance_triage

Ninety-six fictional citizen grievances, each labelled with the department
that acts on it first. Built by `build_dataset.py`, which is the source of
truth; `evalset.jsonl` is its output.

**Illustrative only.** Every grievance, name, place, ID and date is invented.
No real complaint, portal record or person appears here. The label set is a
plausible district-level routing, not any government's actual taxonomy.

**Shape.** Modelled on how grievances arrive on public portals and helplines
such as CPGRAMS: short, first-person, unedited. Language mix is roughly a
third English, a third Hindi in Devanagari, a third Hinglish (Hindi in Latin
script or mixed). Some items carry realistic typos and mixed script on
purpose. Seven items are deliberately ambiguous between two departments and
are labelled with the one that acts first; they are listed at the end of the
builder.

**Labels (11).** water_supply, electricity, roads_and_transport, sanitation,
ration_and_pds, pension_and_welfare, land_records, police_and_safety,
health_services, education, other. Eight to ten items per label.

**Scoring.** A classify profile: the model must reply with exactly one label;
the label parser scores it exact-match. A reply that is not a label is a
format failure, reported separately, never counted as a wrong department.
