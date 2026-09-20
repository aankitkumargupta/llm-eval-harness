"""
Expand the regulated_qa fixture from a 4-document demo into a corpus large
enough to separate five models.

Why this exists: the shipped fixture had 4 passages and 5 questions. Five
questions cannot separate anything, the smallest detectable gap at n=5 is
larger than the whole scale, so any leaderboard drawn from it would have been
noise wearing a ranking (failure mode #1). The comparison needs n in the
tens, at minimum.

What it is: a fictional Data Protection Act, forty-four sections, each a
single passage of 40–90 words carrying one or two concrete facts (a number,
a deadline, a threshold) so every question has a crisp gold answer. It is
deliberately synthetic: `example.gov`, so it can never be contaminated and
never mistaken for legal advice.

Invariants kept:
  * The original four sections and five items are preserved byte-for-byte,
    first, with their ids. Nothing that existed is renamed.
  * Every passage is under the profile's 200-word chunk size, so each document
    is exactly one chunk and its gold id is `{doc_id}#0`. A passage that split
    would make its gold id wrong for every model at once.
  * Idempotent: re-running does not duplicate. Duplicate item ids break the
    paired tests, which join on them.
  * Unanswerable items ask about sections that do not exist in the corpus, so
    abstention is the only correct answer, not a retrieval failure dressed
    up as one.

    python data/regulated_qa/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.jsonl"
EVALSET = HERE / "evalset.jsonl"
URI = "https://example.gov/dpa/s{n}"

# (section, text). Kept short and factual: one passage, one or two numbers.
SECTIONS: list[tuple[int, str]] = [
    (14, "Section 14 requires every data controller to maintain a breach register recording the date of discovery, the categories of data affected, the number of individuals involved and the remedial steps taken. Entries must be retained for five years and produced to the regulator on request within ten working days."),
    (15, "Section 15 requires the appointment of a Data Protection Officer by any controller that processes the personal data of more than 50000 individuals in a calendar year, or that processes health, biometric or financial data as a core activity regardless of scale. The officer's contact details must be published and notified to the regulator."),
    (16, "Section 16 sets the age of digital consent at 14 years. Processing the personal data of a child below that age on the basis of consent requires the verifiable consent of a parent or guardian, and the controller must make reasonable efforts to verify that the consent was given by the holder of parental responsibility."),
    (17, "Section 17 provides that consent may be withdrawn at any time and that withdrawal must be as easy as giving consent. A controller must stop the processing based on that consent within 7 days of withdrawal and must not make the provision of a service conditional on consent that is not necessary for the service."),
    (18, "Section 18 requires controllers to keep a written record of processing activities describing the purposes of processing, the categories of data subjects and data, the recipients, any transfers abroad and the retention period for each category. The record must be reviewed at least once every 12 months."),
    (19, "Section 19 lists six lawful bases for processing personal data: consent, performance of a contract, compliance with a legal obligation, protection of vital interests, performance of a public task, and legitimate interests. A controller must identify its lawful basis before processing begins and record it under Section 18."),
    (21, "Section 21 provides that a processor may act only on the documented instructions of the controller. A processor that determines its own purposes for the data becomes a controller for that processing and assumes all of a controller's obligations, including the penalties in Section 12."),
    (22, "Section 22 permits a processor to engage a sub-processor only with the prior written authorisation of the controller. The processor must give the controller at least 30 days notice of any intended change of sub-processor, during which the controller may object."),
    (23, "Section 23 permits the transfer of personal data to a foreign jurisdiction without further safeguards only where the regulator has issued an adequacy decision for that jurisdiction. Adequacy decisions are reviewed every four years and may be revoked at any time."),
    (24, "Section 24 provides that, absent an adequacy decision, personal data may be transferred abroad only under standard contractual clauses approved by the regulator, binding corporate rules, or the explicit informed consent of the individual. The safeguard used must be recorded in the Section 18 record of processing."),
    (25, "Section 25 requires a data protection impact assessment before any processing likely to result in a high risk to individuals, including systematic monitoring of a public area, large-scale processing of health or biometric data, and automated decisions with legal effect. The assessment must be completed before processing begins."),
    (26, "Section 26 prescribes the content of an impact assessment: a description of the processing and its purposes, an assessment of necessity and proportionality, an assessment of the risks to individuals, and the measures envisaged to address those risks. The assessment must be signed off by the Data Protection Officer where one is appointed."),
    (27, "Section 27 requires a controller to consult the regulator before beginning any processing that an impact assessment shows would present a high residual risk. The regulator must respond within 8 weeks, extendable once by a further 6 weeks for complex cases, and may prohibit the processing."),
    (28, "Section 28 grants individuals the right to obtain confirmation of whether their personal data is being processed and a copy of that data. A controller must respond within one month of receiving the request, extendable by two further months where requests are numerous or complex, provided the individual is told of the extension within the first month."),
    (29, "Section 29 provides that a copy of personal data under Section 28 must be supplied free of charge. A controller may charge a reasonable fee based on administrative cost, or refuse to act, only where a request is manifestly unfounded or excessive, in particular because of its repetitive character."),
    (30, "Section 30 grants individuals the right to have inaccurate personal data rectified without undue delay and, taking into account the purposes of processing, to have incomplete data completed. The controller must inform every recipient to whom the data was disclosed of the rectification unless this proves impossible or involves disproportionate effort."),
    (32, "Section 32 grants individuals the right to receive the personal data they provided to a controller in a structured, commonly used and machine-readable format, and to transmit it to another controller. The right applies only where processing is based on consent or contract and is carried out by automated means."),
    (33, "Section 33 grants individuals the right to object at any time to processing based on legitimate interests or the performance of a public task. The controller must stop unless it demonstrates compelling legitimate grounds that override the individual's interests. An objection to direct marketing must always be honoured."),
    (34, "Section 34 provides that an individual has the right not to be subject to a decision based solely on automated processing that produces legal effects or similarly significant effects. Such decisions are permitted only where necessary for a contract, authorised by law, or based on explicit consent, and the individual must be able to obtain human intervention."),
    (35, "Section 35 provides that information society services offered directly to a child must present their privacy information in language a child of the target age can understand. Profiling a child for marketing purposes is prohibited regardless of consent."),
    (36, "Section 36 requires controllers and processors to implement technical and organisational measures appropriate to the risk, including pseudonymisation and encryption where appropriate, the ability to restore availability after an incident, and a process for regularly testing the effectiveness of those measures at least annually."),
    (37, "Section 37 provides that personal data which was encrypted with a state-of-the-art algorithm, where the key was not compromised, is treated as unintelligible for the purposes of breach notification. A controller relying on this provision must still record the breach in the Section 14 register."),
    (38, "Section 38 provides that personal data must not be kept in an identifiable form for longer than is necessary for the purposes for which it was collected. Retention periods must be documented under Section 18 and reviewed at least once every 12 months, and data past its retention period must be deleted or anonymised within 60 days."),
    (39, "Section 39 provides that data which has been anonymised so that the individual is no longer identifiable by any means reasonably likely to be used is not personal data and falls outside this Act. Pseudonymised data, where re-identification remains possible with additional information, remains personal data."),
    (40, "Section 40 classifies breaches into three severity tiers. Tier 1 covers breaches affecting fewer than 100 individuals with no sensitive data. Tier 2 covers breaches affecting 100 to 9999 individuals or any sensitive data. Tier 3 covers breaches affecting 10000 or more individuals. The tier determines the notification and penalty rules that apply."),
    (41, "Section 41 requires a controller to notify the regulator of a Tier 2 or Tier 3 breach within 72 hours of becoming aware of it, and of a Tier 1 breach within 30 days. Where notification is late, the controller must give reasons for the delay. Notification to affected individuals is governed separately by Section 13."),
    (42, "Section 42 empowers the regulator to conduct audits of any controller or processor on 14 days written notice, or without notice where it has reasonable grounds to suspect a serious violation. During an audit the regulator may inspect premises, equipment and records and may require any person to answer questions."),
    (43, "Section 43 caps administrative fines for a violation of this Act at 4 percent of an organisation's worldwide annual turnover for the preceding financial year, or 2 million dollars, whichever is greater. Fines for record-keeping failures under Section 18 are capped at 2 percent or 1 million dollars."),
    (44, "Section 44 grants any person subject to a penalty or an order of the regulator the right to appeal to the Data Protection Tribunal within 28 days of the decision being served. An appeal does not suspend the order unless the Tribunal directs otherwise."),
    (46, "Section 46 makes it a criminal offence to knowingly or recklessly obtain, disclose or sell personal data without the consent of the controller, and to re-identify anonymised data without lawful authority. The offence carries a fine of up to 50000 dollars or imprisonment of up to two years, or both."),
    (47, "Section 47 protects any employee who reports a suspected violation of this Act to the regulator in good faith. Dismissal or detriment on the ground of such a report is void, and the employee may recover compensation of up to 12 months salary from the Tribunal."),
    (48, "Section 48 permits industry bodies to draw up codes of conduct specifying how this Act applies in their sector. A code takes effect only once approved by the regulator, and adherence to an approved code is taken into account in assessing compliance and in setting any penalty."),
    (49, "Section 49 establishes a voluntary certification scheme under which an accredited body may certify that a controller's processing complies with this Act. A certificate is valid for three years, may be renewed, and must be withdrawn by the certifying body if the conditions are no longer met."),
    (50, "Section 50 exempts an organisation with fewer than 25 employees from the record-keeping obligation in Section 18 and the officer appointment obligation in Section 15, unless its processing is likely to result in a high risk to individuals or includes sensitive data as a core activity."),
    (51, "Section 51 permits processing for scientific or historical research or statistical purposes without the individual's consent, provided appropriate safeguards are in place, the data is pseudonymised where the purposes can be fulfilled that way, and the results are not used to take decisions about any individual."),
    (52, "Section 52 exempts processing carried out for journalistic purposes from Sections 28 to 34 where the controller reasonably believes that publication is in the public interest and that compliance would be incompatible with the journalistic purpose. The exemption does not extend to Section 36 security obligations."),
    (53, "Section 53 provides that closed-circuit television in a public area is systematic monitoring for the purposes of Section 25. Recordings must be deleted within 31 days unless retained as evidence of an incident, and clear signage must inform individuals that recording is taking place and identify the controller."),
    (54, "Section 54 provides that direct marketing by electronic mail requires the prior consent of the recipient, except where the recipient is an existing customer, the marketing concerns similar products, and every message offers a simple means of opting out. An opt-out must take effect within 10 working days."),
    (55, "Section 55 provides that a controller may store information on, or gain access to information stored in, a user's terminal equipment only with the user's consent, except where strictly necessary to provide a service the user has requested. Consent obtained through pre-ticked boxes or continued browsing is not valid."),
    (56, "Section 56 classifies biometric data used to uniquely identify a person, genetic data, health data, and data revealing racial or ethnic origin, political opinions, religious beliefs or trade union membership as sensitive data. Processing sensitive data is prohibited unless one of the ten conditions in Schedule 1 applies."),
]

# (item_id, question, gold answer, gold section)
ANSWERABLE: list[tuple[str, str, str, int]] = [
    ("rq6", "How long must entries in the breach register be retained?", "Five years.", 14),
    ("rq7", "Within how many working days must the breach register be produced to the regulator on request?", "Within ten working days.", 14),
    ("rq8", "Above what number of individuals processed per year must a controller appoint a Data Protection Officer?", "More than 50000 individuals in a calendar year.", 15),
    ("rq9", "What is the age of digital consent under Section 16?", "14 years.", 16),
    ("rq10", "After consent is withdrawn, within how many days must the controller stop the processing?", "Within 7 days.", 17),
    ("rq11", "How often must the record of processing activities be reviewed?", "At least once every 12 months.", 18),
    ("rq12", "How many lawful bases for processing does Section 19 list?", "Six.", 19),
    ("rq13", "What happens to a processor that determines its own purposes for the data?", "It becomes a controller for that processing and assumes all of a controller's obligations.", 21),
    ("rq14", "How much notice must a processor give the controller before changing a sub-processor?", "At least 30 days notice.", 22),
    ("rq15", "How often are adequacy decisions reviewed?", "Every four years.", 23),
    ("rq16", "Absent an adequacy decision, name a safeguard under which data may be transferred abroad.", "Standard contractual clauses approved by the regulator, binding corporate rules, or the explicit informed consent of the individual.", 24),
    ("rq17", "When must a data protection impact assessment be completed?", "Before the processing begins.", 25),
    ("rq18", "Who must sign off an impact assessment where one is appointed?", "The Data Protection Officer.", 26),
    ("rq19", "How long does the regulator have to respond to a prior consultation under Section 27?", "8 weeks, extendable once by a further 6 weeks for complex cases.", 27),
    ("rq20", "How long does a controller have to respond to a subject access request?", "One month, extendable by two further months where requests are numerous or complex.", 28),
    ("rq21", "When may a controller charge a fee for a copy of personal data?", "Only where the request is manifestly unfounded or excessive, in particular because of its repetitive character.", 29),
    ("rq22", "Who must a controller inform when personal data is rectified?", "Every recipient to whom the data was disclosed, unless this proves impossible or involves disproportionate effort.", 30),
    ("rq23", "On which lawful bases does the right to data portability apply?", "Consent or contract, where processing is carried out by automated means.", 32),
    ("rq24", "Must an objection to direct marketing be honoured?", "Yes, an objection to direct marketing must always be honoured.", 33),
    ("rq25", "What must an individual be able to obtain in respect of a permitted solely automated decision?", "Human intervention.", 34),
    ("rq26", "Is profiling a child for marketing purposes permitted with consent?", "No, it is prohibited regardless of consent.", 35),
    ("rq27", "How often must the effectiveness of security measures be tested under Section 36?", "At least annually.", 36),
    ("rq28", "Under what condition is encrypted personal data treated as unintelligible for breach notification?", "Where it was encrypted with a state-of-the-art algorithm and the key was not compromised.", 37),
    ("rq29", "Within how many days must data past its retention period be deleted or anonymised?", "Within 60 days.", 38),
    ("rq30", "Is pseudonymised data still personal data under the Act?", "Yes, pseudonymised data remains personal data because re-identification remains possible with additional information.", 39),
    ("rq31", "How many individuals must be affected for a breach to be Tier 3?", "10000 or more individuals.", 40),
    ("rq32", "Which breach tier covers any breach involving sensitive data?", "Tier 2.", 40),
    ("rq33", "Within what period must the regulator be notified of a Tier 1 breach?", "Within 30 days.", 41),
    ("rq34", "How much written notice must the regulator give before a routine audit?", "14 days.", 42),
    ("rq35", "What is the maximum administrative fine as a percentage of worldwide annual turnover?", "4 percent of worldwide annual turnover for the preceding financial year, or 2 million dollars, whichever is greater.", 43),
    ("rq36", "What is the cap on fines for record-keeping failures under Section 18?", "2 percent of worldwide annual turnover or 1 million dollars.", 43),
    ("rq37", "Within how many days must an appeal be lodged with the Data Protection Tribunal?", "Within 28 days of the decision being served.", 44),
    ("rq38", "What is the maximum term of imprisonment for knowingly obtaining personal data without consent?", "Up to two years.", 46),
    ("rq39", "How much compensation may a whistleblowing employee recover from the Tribunal?", "Up to 12 months salary.", 47),
    ("rq40", "When does an industry code of conduct take effect?", "Only once approved by the regulator.", 48),
    ("rq41", "How long is a certificate issued under the certification scheme valid?", "Three years.", 49),
    ("rq42", "Below what number of employees is an organisation exempt from the record-keeping obligation?", "Fewer than 25 employees.", 50),
    ("rq43", "May research results under Section 51 be used to take decisions about an individual?", "No, the results must not be used to take decisions about any individual.", 51),
    ("rq44", "Does the journalism exemption extend to the security obligations in Section 36?", "No, the exemption does not extend to Section 36 security obligations.", 52),
    ("rq45", "Within how many days must CCTV recordings be deleted unless retained as evidence?", "Within 31 days.", 53),
    ("rq46", "Within how many working days must a marketing opt-out take effect?", "Within 10 working days.", 54),
    ("rq47", "Is consent obtained through a pre-ticked box valid under Section 55?", "No, consent obtained through pre-ticked boxes or continued browsing is not valid.", 55),
    ("rq48", "How many conditions in Schedule 1 permit the processing of sensitive data?", "Ten.", 56),
    ("rq49", "What is the fine for repeat offences of unauthorized disclosure within a two-year period?", "Fines up to 20000 dollars and suspension of data-handling licenses.", 12),
]

# Sections that do not exist in the corpus. Abstention is the only right answer.
UNANSWERABLE: list[tuple[str, str]] = [
    ("rq50", "What is the maximum fine under Section 60 for failing to register a cookie banner design?"),
    ("rq51", "How many members sit on the Data Protection Tribunal under Section 70?"),
    ("rq52", "What licence fee does Section 58 set for operating a data broker?"),
    ("rq53", "Under Section 99, how many days does a controller have to respond to a freedom-of-information request?"),
]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def main() -> None:
    corpus = _read(CORPUS)
    have = {d["doc_id"] for d in corpus}
    added_docs = 0
    for n, text in SECTIONS:
        words = len(text.split())
        assert words < 200, f"act{n} is {words} words; it would split into two chunks"
        did = f"act{n}"
        if did in have:
            continue
        corpus.append({"doc_id": did, "text": text, "source_uri": URI.format(n=n)})
        added_docs += 1
    _write(CORPUS, corpus)

    evalset = _read(EVALSET)
    have_items = {r["item_id"] for r in evalset}
    doc_ids = {d["doc_id"] for d in corpus}
    added_items = 0
    for iid, q, gold, sec in ANSWERABLE:
        if iid in have_items:
            continue
        assert f"act{sec}" in doc_ids, f"{iid} cites act{sec}, which is not in the corpus"
        evalset.append({"item_id": iid, "query": q, "item_type": "answerable",
                        "gold_answer": gold, "gold_passage_ids": [f"act{sec}#0"]})
        added_items += 1
    for iid, q in UNANSWERABLE:
        if iid in have_items:
            continue
        evalset.append({"item_id": iid, "query": q, "item_type": "unanswerable",
                        "gold_answer": None, "gold_passage_ids": []})
        added_items += 1
    _write(EVALSET, evalset)

    print(f"corpus : {len(corpus)} documents (+{added_docs})")
    print(f"evalset: {len(evalset)} items (+{added_items}); "
          f"{sum(r['item_type'] == 'answerable' for r in evalset)} answerable, "
          f"{sum(r['item_type'] == 'unanswerable' for r in evalset)} unanswerable")


if __name__ == "__main__":
    main()
