"""
in_grievance_triage: route a citizen grievance to the right department.

Modelled on the shape of grievances filed on public portals such as
CPGRAMS and state helplines: short, first-person, often in Hindi or
Hinglish, often with a typo, sometimes ambiguous between two departments.
Every grievance here is fictional. Names, places, IDs and dates are
invented; no real person's complaint appears. Illustrative only.

Labels (11): water_supply, electricity, roads_and_transport, sanitation,
ration_and_pds, pension_and_welfare, land_records, police_and_safety,
health_services, education, other.

Language mix: roughly a third English, a third Hindi (Devanagari), a third
Hinglish (Hindi in Latin script or mixed). A few items carry realistic
typos and mixed script on purpose; the label is still unambiguous to a
human reader. The seven items marked ambiguous are the ones where a
reasonable clerk could argue for two departments; they are labelled with
the department that acts first.

Idempotent: item ids are stable; re-running rewrites the same file.

    python data/in_grievance_triage/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "evalset.jsonl"

# (label, text). 8-9 per label, 92 in all.
ITEMS: list[tuple[str, str]] = [
    # --- water_supply ---------------------------------------------------- #
    ("water_supply", "No water supply in Sector 14, Ward 9 since Monday morning. Tanker was promised but never came."),
    ("water_supply", "हमारे मोहल्ले (गली नंबर 4, रामनगर) में पिछले पाँच दिनों से नल का पानी नहीं आ रहा है।"),
    ("water_supply", "Paani ka pressure itna kam hai ki upar ke floor tak paani nahi pahunchta. Please pipeline check karein."),
    ("water_supply", "The water we receive is muddy and smells bad. Two children in the building fell ill this week."),
    ("water_supply", "नई पानी की लाइन के कनेक्शन के लिए आवेदन दिया था, 40 दिन हो गए, कोई जवाब नहीं आया।"),
    ("water_supply", "Water meter reading wrong, bill shows 48,000 litres for a 2 person house. Please re-check meter."),
    ("water_supply", "Sadak ke beech main pipe phat gayi hai aur paani bah raha hai 3 din se. Koi dekhne nahi aaya."),
    ("water_supply", "Handpump near the anganwadi in Dhanpur village is broken; women are walking 2 km for water."),
    ("water_supply", "Sumer supply timing 6 to 7 am is too short, please extend water supply time in Block C."),
    # --- electricity ------------------------------------------------------ #
    ("electricity", "Power cut every evening 7 to 10 pm for the last two weeks in Gandhi Colony. No notice, no reason."),
    ("electricity", "बिजली का बिल इस महीने 9,400 रुपये आया है जबकि पिछले महीनों में 1,200 के आसपास आता था। मीटर की जाँच करवाएँ।"),
    ("electricity", "Transformer near bus stand sparks and makes loud noise at night, bahut khatarnaak hai, please replace."),
    ("electricity", "New connection applied 2 months ago, deposit paid, still no meter installed. Application no. EL-2291."),
    ("electricity", "हमारे गाँव में बिजली के तार बहुत नीचे लटक रहे हैं, बच्चों के लिए खतरा है।"),
    ("electricity", "Street lights on the main road of Vikas Nagar are off since Diwali. Dark and unsafe at night."),
    ("electricity", "Meter reader nahi aata, har baar average bill bhej dete hain. Actual reading ke hisaab se bill chahiye."),
    ("electricity", "Voltage fluctuation damaged our fridge and TV. Request compensation and line inspection."),
    # --- roads_and_transport ---------------------------------------------- #
    ("roads_and_transport", "Huge potholes on Station Road between the temple and the school; two accidents last week."),
    ("roads_and_transport", "पंचायत भवन से मुख्य सड़क तक की कच्ची सड़क बरसात में पूरी तरह कीचड़ हो जाती है, पक्की सड़क बनवाएँ।"),
    ("roads_and_transport", "City bus route 27 has stopped coming to our stop after 8 pm. Office workers are stranded."),
    ("roads_and_transport", "Speed breaker ke upar koi paint nahi hai, raat ko dikhta nahi, do-pahiya wale gir rahe hain."),
    ("roads_and_transport", "The footpath outside the district hospital is dug up for 3 months and never repaired."),
    ("roads_and_transport", "गली में नाली की खुदाई के बाद सड़क वापस नहीं बनाई गई, गाड़ी निकालना मुश्किल है।"),
    ("roads_and_transport", "Auto drivers at the railway station refuse to go by meter and overcharge, no prepaid booth."),
    ("roads_and_transport", "Bridge on the Kalindi nala has a crack, heavy vehicles still using it. Please inspect urgently."),
    # --- sanitation ------------------------------------------------------- #
    ("sanitation", "Garbage has not been collected from our lane for 9 days. Dogs and flies everywhere."),
    ("sanitation", "हमारी गली की नाली बंद है और गंदा पानी घरों के सामने जमा हो रहा है।"),
    ("sanitation", "Public toilet near the market is locked since two weeks, log sadak par hi jaate hain."),
    ("sanitation", "Overflowing sewer manhole on Lake View Road, sewage entering ground floor houses."),
    ("sanitation", "कूड़ा उठाने वाली गाड़ी हफ्ते में सिर्फ एक बार आती है, रोज़ाना आनी चाहिए।"),
    ("sanitation", "Dead animal lying near the school gate since yesterday, please remove, bachhe udhar se guzarte hain."),
    ("sanitation", "Open dumping ground next to the housing colony is being burnt at night, smoke enters homes."),
    ("sanitation", "Fogging for mosquitoes has not been done this season, dengue cases rising in Ward 3."),
    # --- ration_and_pds --------------------------------------------------- #
    ("ration_and_pds", "Ration shop dealer gives only 3 kg wheat instead of 5 kg per person and says stock is short."),
    ("ration_and_pds", "मेरा राशन कार्ड ई-केवाईसी के बाद भी पोर्टल पर निष्क्रिय दिखा रहा है, दो महीने से अनाज नहीं मिला।"),
    ("ration_and_pds", "Ration wale ne biometric fail bata kar do baar wapas bhej diya, ab kya karein?"),
    ("ration_and_pds", "Applied to add my newborn daughter to the ration card in March; no update on the application."),
    ("ration_and_pds", "उचित मूल्य की दुकान महीने में सिर्फ 4 दिन खुलती है और उसी में भीड़ लग जाती है।"),
    ("ration_and_pds", "Dealer is charging Rs 2 per kg extra on rice, receipt nahi deta."),
    ("ration_and_pds", "My ration card shows the wrong village after migration; cannot draw grain at the new shop."),
    ("ration_and_pds", "Kerosene entitlement removed from our card without any notice; we have no LPG connection."),
    # --- pension_and_welfare ----------------------------------------------- #
    ("pension_and_welfare", "My mother's widow pension has not been credited for 4 months. Account is active, nothing received."),
    ("pension_and_welfare", "वृद्धावस्था पेंशन का आवेदन एक साल पहले किया था, जीवन प्रमाण भी जमा किया, अब तक स्वीकृत नहीं हुआ।"),
    ("pension_and_welfare", "Disability certificate hai phir bhi divyang pension form reject ho gaya, reason nahi bataya."),
    ("pension_and_welfare", "The scholarship amount for my son's class 11 was sanctioned but never transferred to the account."),
    ("pension_and_welfare", "पेंशन की राशि 1,000 से घटाकर 500 कर दी गई है, बिना किसी सूचना के।"),
    ("pension_and_welfare", "Maternity benefit ki second kisht 6 mahine se pending hai, aanganwadi wale kehte hain upar se atka hai."),
    ("pension_and_welfare", "Housing scheme instalment stopped after the first payment; house is half built."),
    ("pension_and_welfare", "Life certificate submitted in November but pension portal still shows it as pending."),
    # --- land_records ------------------------------------------------------ #
    ("land_records", "The online land record shows my late father's name; mutation to my name is pending for 18 months."),
    ("land_records", "खसरा नंबर 112 में हमारी ज़मीन का रकबा गलत दर्ज है, 0.8 हेक्टेयर की जगह 0.3 दिखा रहा है।"),
    ("land_records", "Patwari ne naksha dene se mana kar diya bina reason ke, 3 baar chakkar laga chuka hoon."),
    ("land_records", "Neighbour has encroached 2 feet into our plot; demarcation request filed in July, no survey yet."),
    ("land_records", "जमीन की रजिस्ट्री हो गई पर दाखिल-खारिज के लिए कार्यालय पैसे माँग रहा है।"),
    ("land_records", "Certified copy of the sale deed applied 5 weeks ago; office says record room is closed."),
    ("land_records", "Bhulekh portal par mera naam galat spelling se hai, correction ka koi option nahi mil raha."),
    ("land_records", "Two different khatas show the same survey number; bank refuses loan until it is resolved."),
    # --- police_and_safety -------------------------------------------------- #
    ("police_and_safety", "My phone was stolen at the bus stand on 3 March; the police station refused to register an FIR."),
    ("police_and_safety", "रात में हमारी कॉलोनी में लगातार चोरियाँ हो रही हैं, पुलिस गश्त बढ़ाई जाए।"),
    ("police_and_safety", "Mohalle ke ladke roz raat ko sharaab pee kar hungama karte hain, koi karyavahi nahi hoti."),
    ("police_and_safety", "Complaint of domestic violence given at the women's help desk; no follow-up in 20 days."),
    ("police_and_safety", "ट्रैफिक पुलिस बिना रसीद के चालान के नाम पर पैसे ले रही है, चौराहा नंबर 5 पर।"),
    ("police_and_safety", "Eve teasing outside the girls' college every evening, we have written twice, nothing done."),
    ("police_and_safety", "Passport verification report pending at the local thana for 45 days, application on hold."),
    ("police_and_safety", "Illegal parking mafia collecting money outside the hospital, threatens people who refuse."),
    # --- health_services --------------------------------------------------- #
    ("health_services", "PHC in Rampur has had no doctor for three weeks; the pharmacist is treating patients."),
    ("health_services", "सरकारी अस्पताल में दवाइयाँ नहीं मिल रहीं, बाहर से खरीदने को कहा जाता है।"),
    ("health_services", "Ambulance number 108 pe call kiya, 1 ghante tak nahi aayi, patient ko auto mein le jana pada."),
    ("health_services", "Health card shows my treatment as approved but the hospital is demanding cash payment."),
    ("health_services", "टीकाकरण शिविर की तारीख बदल दी गई और किसी को सूचना नहीं दी गई, बच्चे बिना टीके के लौटे।"),
    ("health_services", "Lab reports at the district hospital take 10 days; private labs give them the same day."),
    ("health_services", "Ward mein ek bhi nurse raat ko nahi rehti, mareez ke rishtedaar hi sab karte hain."),
    ("health_services", "Anganwadi is not distributing the take-home ration for pregnant women this month."),
    # --- education ---------------------------------------------------------- #
    ("education", "The government primary school in Tikri has only one teacher for five classes."),
    ("education", "मिड-डे मील में हफ्ते से सिर्फ चावल दिया जा रहा है, दाल-सब्ज़ी नहीं।"),
    ("education", "School ka building ki chhat se paani tapakta hai, class 3 ke bachhe baramde mein baithte hain."),
    ("education", "Scholarship form portal keeps rejecting the caste certificate upload, deadline is next week."),
    ("education", "कक्षा 9 की मुफ्त किताबें अभी तक नहीं मिलीं, सत्र शुरू हुए तीन महीने हो गए।"),
    ("education", "Private school is demanding capitation fee for admission under the 25% quota seat."),
    ("education", "Transfer certificate is being withheld by the school until we pay a 'development fee'."),
    ("education", "Girls' toilet in the high school has been out of order for the whole year."),
    # --- other --------------------------------------------------------------- #
    ("other", "My income certificate application has been 'under process' on the e-district portal for 50 days."),
    ("other", "जन्म प्रमाण पत्र में बच्चे का नाम गलत छपा है, सुधार के लिए कहाँ आवेदन करें?"),
    ("other", "Aadhaar update centre in the block office is closed every time I go, koi timing nahi likhi hai."),
    ("other", "Stray cattle sit in the middle of the highway near the toll plaza; accidents almost daily."),
    ("other", "पंचायत की बैठक की सूचना समय पर नहीं दी जाती, ग्रामसभा में कोई नहीं पहुँच पाता।"),
    ("other", "Bank correspondent in our village charges Rs 50 for every withdrawal from the Jan Dhan account."),
    ("other", "Loudspeakers at the weekly market run till midnight; senior citizens cannot sleep."),
    ("other", "Marriage certificate issued with the wrong date; the registrar's office says it cannot be corrected online."),
    # --- ambiguous on purpose: labelled with the department that acts first --- #
    ("sanitation", "Blocked drain outside the ration shop floods the street every evening; shop is closing early because of it."),
    ("water_supply", "Water pipeline broke during road repair work; road contractor left and now the lane has no water."),
    ("health_services", "Mosquito breeding in the open drain behind the hospital; patients are getting dengue inside the ward."),
    ("police_and_safety", "Street lights off for a month on the road to the girls' hostel and there were two chain-snatching incidents."),
    ("pension_and_welfare", "Old-age pension stopped because the bank says the Aadhaar seeding failed; bank sends me to the block office and back."),
    ("land_records", "Panchayat built a drain across the corner of my plot without notice and the record still shows the full area as mine."),
    ("education", "School van charges doubled without notice and the driver is unlicensed; the school says it is not their vehicle."),
]

LABELS = ["water_supply", "electricity", "roads_and_transport", "sanitation",
          "ration_and_pds", "pension_and_welfare", "land_records",
          "police_and_safety", "health_services", "education", "other"]


def main() -> None:
    assert all(lab in LABELS for lab, _ in ITEMS)
    rows = []
    counts: dict[str, int] = {}
    for lab, text in ITEMS:
        counts[lab] = counts.get(lab, 0) + 1
        rows.append({"item_id": f"{lab}_{counts[lab]:02d}", "query": text,
                     "item_type": "answerable", "gold_answer": lab})
    ids = [r["item_id"] for r in rows]
    assert len(ids) == len(set(ids))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    print(f"wrote {len(rows)} grievances across {len(counts)} labels -> {OUT}")
    for lab in LABELS:
        print(f"  {lab:22s} {counts.get(lab, 0)}")


if __name__ == "__main__":
    main()
