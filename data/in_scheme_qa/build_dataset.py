"""
in_scheme_qa: citizens ask about government welfare schemes; the model must
answer from the scheme passages it is given, in the language of the
question, and say so when the passages do not answer.

Every scheme here is fictional. Names, amounts, eligibility rules,
deadlines and offices are invented so that the evaluation measures one
thing: whether the model answers from the retrieved passages rather than
from whatever it remembers about real schemes with similar names. Any
resemblance to a real scheme is coincidental and the rules here must not be
relied on. Illustrative only.

The corpus is 42 short passages (under 180 words each, so each is exactly
one retrieval chunk, `<doc_id>#0`). The evalset is 48 answerable questions
in English, Hindi and Hinglish, 6 hand-written unanswerable questions that
are on-topic but not covered, plus the harness's derived probes
(unanswerable variants, noise, injection, paraphrase) generated here with
the profile's settings so the file is a pure function of this script.

Idempotent: stable ids, same output every run.

    python data/in_scheme_qa/build_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from harness.eval.probes import ProbeConfig, build_probe_suite  # noqa: E402
from harness.profiles.loaders import load_evalset, save_evalset  # noqa: E402

CORPUS = HERE / "corpus.jsonl"
EVALSET = HERE / "evalset.jsonl"
BASE = HERE / "evalset_base.jsonl"

# Probe fractions mirror configs/profiles/in_scheme_qa.yaml. Keep in step.
PROBES = ProbeConfig(unanswerable=0.15, noise=0.10, injection=0.15,
                     paraphrase=0.10, positional=0.0, n_distractors=3, seed=0)

# (doc_id, text). All fictional.
DOCS: list[tuple[str, str]] = [
    ("sahyog_pension", "Sahyog Vridh Pension Top-up (illustrative scheme). The state adds Rs 500 per month to the old-age pension of every beneficiary aged 60 or above whose household is in the BPL list. The top-up is credited with the regular pension on the 5th of each month to the same bank account. No separate application is needed: the block office adds eligible pensioners automatically after the annual life certificate is submitted between 1 November and 31 December. Pensioners who miss the life certificate window lose the top-up for the following year and must apply for restoration at the block office with the certificate."),
    ("vidya_setu", "Vidya Setu Girls' Scholarship (illustrative scheme). Girls studying in Classes 9 to 12 in a government or aided school whose family income is below Rs 2.5 lakh a year receive Rs 6,000 per year, paid in two instalments of Rs 3,000 in September and February. Applications are made on the state scholarship portal by 31 August with the income certificate, the previous year's marksheet and the bank passbook of the student or her mother. Attendance below 75 percent in the previous year disqualifies the student for that year. Renewal is automatic if attendance and promotion conditions are met."),
    ("kisan_jal", "Kisan Jal Sinchai Subsidy (illustrative scheme). Farmers with land holdings of up to 5 hectares get a 55 percent subsidy on drip and sprinkler irrigation equipment, raised to 75 percent for small and marginal farmers holding up to 2 hectares. The maximum subsidy is Rs 80,000 per farmer. Applications are made on the agriculture department portal with the land record (khasra copy), Aadhaar and bank passbook. Equipment must be bought from an empanelled supplier after approval; purchases made before approval are not reimbursed. The subsidy is credited to the bank account within 60 days of the installation inspection."),
    ("gram_ujala", "Gram Ujala Streetlight Grant (illustrative scheme). Every gram panchayat may claim a grant of up to Rs 1.2 lakh per year for installing and maintaining LED streetlights. The panchayat passes a resolution in the gram sabha, submits it with a quotation from an empanelled vendor to the block office, and receives 80 percent of the grant in advance and 20 percent after the block engineer inspects the installation. Panchayats with unspent balance from the previous year receive only the difference. Repair of existing lights is an eligible expense; purchase of generators is not."),
    ("nari_udyam", "Nari Udyam Loan Scheme (illustrative scheme). Women aged 18 to 55 starting or expanding a business can get a bank loan of up to Rs 5 lakh with a 6 percent interest subsidy paid by the state for the first three years. There is no collateral for loans up to Rs 2 lakh. Applications go through the district industries centre with a project report, Aadhaar, a residence certificate and, for existing businesses, the last year's accounts. The district committee meets monthly and decides within 45 days. A woman who has defaulted on an earlier government-linked loan is not eligible."),
    ("shramik_suraksha", "Shramik Suraksha Accident Cover (illustrative scheme). Construction workers registered with the state labour welfare board are covered for Rs 5 lakh on accidental death and Rs 2.5 lakh on permanent disability. There is no premium; the cover is funded by the labour welfare cess. Registration costs Rs 25 and needs a 90-day work certificate from an employer or contractor, an age proof and a photograph. Claims are filed at the district labour office within one year of the accident with the FIR or accident report and the hospital record. The nominee named at registration receives the payment."),
    ("matru_poshan", "Matru Poshan Maternity Benefit (illustrative scheme). A pregnant woman registered at an anganwadi or government health centre receives Rs 6,000 for her first two live births, paid in three instalments: Rs 2,000 on registration of the pregnancy in the first trimester, Rs 2,000 after the second antenatal check-up, and Rs 2,000 after the child's first vaccinations are complete. Registration must be done within 150 days of the last menstrual period. Women in regular government employment are not eligible. The payment goes to the mother's own bank account."),
    ("yuva_kaushal", "Yuva Kaushal Training Stipend (illustrative scheme). Candidates aged 18 to 35 enrolled in a skill course of at least three months at a government ITI or an empanelled training centre receive Rs 1,500 per month during the course and a one-time Rs 3,000 placement bonus if employed within six months of completion. The stipend needs 80 percent attendance in the month. Applications are made at the training centre with Aadhaar, an educational certificate and a bank passbook. A candidate may receive the stipend for only one course in a lifetime."),
    ("awas_sahayata", "Awas Sahayata Housing Grant (illustrative scheme). Rural families without a pucca house and with annual income below Rs 1.5 lakh receive Rs 1.3 lakh to build a house of at least 25 square metres, paid in three stages: Rs 40,000 on sanction, Rs 60,000 after the plinth is inspected, and Rs 30,000 after the roof is cast. The family must own the plot or hold a patta for it. Applications are made to the gram panchayat, which forwards a priority list to the block office. Families who received a housing grant under any earlier scheme are not eligible."),
    ("swachh_gram", "Swachh Gram Toilet Incentive (illustrative scheme). A rural household that builds a twin-pit toilet with a water connection receives Rs 12,000 after the gram panchayat verifies construction and uploads a geotagged photograph. The household must not have received a toilet incentive earlier. Applications are made at the gram panchayat with Aadhaar and a bank passbook. Payment is made within 30 days of verification. Households in villages already declared open-defecation free are still eligible if they are newly formed or newly migrated."),
    ("balika_samriddhi", "Balika Janm Samriddhi (illustrative scheme). For a girl born on or after 1 April 2024 to a family with annual income below Rs 3 lakh, the state deposits Rs 25,000 in a fixed deposit in the girl's name. The maturity amount is paid when she turns 18, provided she has passed Class 10 and is unmarried at 18. Application must be made within one year of birth at the district women and child development office with the birth certificate, the income certificate and the parents' Aadhaar. A family may enrol at most two girls."),
    ("divyang_sahayak", "Divyang Sahayak Aids and Appliances (illustrative scheme). Persons with a disability certificate showing 40 percent or more disability receive free assistive devices: wheelchairs, tricycles, hearing aids, crutches and smartphones with screen readers for the visually impaired. Devices are issued at assessment camps held in every block twice a year, in March and September, and at the district hospital on any working day. Bring the disability certificate, Aadhaar and one photograph. A device can be replaced free after three years, or earlier if a doctor certifies it is unusable."),
    ("fasal_kharid", "Kisan Fasal Kharid Registration (illustrative scheme). To sell wheat or paddy at the minimum support price at a government procurement centre, a farmer must register on the procurement portal before the season opens: by 15 March for wheat and by 15 September for paddy. Registration needs the land record, Aadhaar and a bank passbook, and the crop sown must match the record. The portal issues a slot date; produce brought on another day is not accepted. Payment is made to the bank account within 72 hours of the weighment slip."),
    ("pashu_bima", "Pashu Dhan Bima (illustrative scheme). Livestock owners can insure cattle, buffaloes, goats and sheep for one year with the state paying 70 percent of the premium; the owner pays the rest, about Rs 150 per cow. The animal is ear-tagged at enrolment by the veterinary officer. On death, the owner must inform the veterinary office within 24 hours; a post-mortem certificate and the ear tag are needed for the claim. The sum insured is the market value assessed at enrolment, up to Rs 60,000 per animal. Claims are paid within 30 days."),
    ("chhatra_cycle", "Chhatra Cycle Yojana (illustrative scheme). Students entering Class 9 in a government school located more than 2 kilometres from their home receive a bicycle free of cost. The school prepares the list from admission records and the distance declared by the parent, verified by the headmaster. Bicycles are distributed at the school by 31 October. A student who leaves the school within one year must return the bicycle. Students who received a bicycle under any earlier scheme are not eligible again."),
    ("panchayat_library", "Gram Panchayat Library Grant (illustrative scheme). A gram panchayat that provides a room of at least 30 square metres receives Rs 2 lakh once for furnishing and Rs 30,000 per year for books, newspapers and a part-time librarian honorarium. The library must be open at least four hours a day, six days a week, and free to all residents. The annual grant is released after the panchayat uploads the previous year's utilisation certificate and visitor register summary on the panchayat portal by 30 April."),
    ("vendor_card", "Shehri Vendor Card (illustrative scheme). Street vendors in a municipal area apply for a vendor card at the municipal office with Aadhaar, a photograph and proof of vending for at least one year, such as an earlier challan, an association letter or a survey slip. The card costs Rs 100 and is valid for five years. It entitles the holder to a vending spot allotted by the town vending committee and protects against removal from that spot without 30 days' notice. Card holders may also apply for a working-capital loan of Rs 20,000 at a subsidised rate."),
    ("jal_sanchay", "Jal Sanchay Rainwater Harvesting Rebate (illustrative scheme). A property owner in a municipal area who installs a rainwater harvesting system certified by the municipal engineer gets a 10 percent rebate on property tax for five years. Applications are made at the ward office with the installation invoice and photographs; the engineer inspects within 21 days. For buildings on plots above 300 square metres, rainwater harvesting is mandatory and no rebate is available. The rebate starts from the financial year after certification."),
    ("bijli_sahayata", "Bijli Sahayata Electricity Subsidy (illustrative scheme). BPL households with a single-phase domestic connection pay a flat Rs 1 per unit for the first 100 units each month; the difference from the tariff is paid by the state to the distribution company. Consumption above 100 units is charged at the normal tariff for the whole bill. To enrol, the consumer submits the BPL card and the consumer number at the sub-division office or on the distribution company's app. The subsidy is withdrawn if the connection is found to be used for commercial purposes."),
    ("vidhwa_sahayata", "Vidhwa Sahayata Monthly Assistance (illustrative scheme). A widow aged 18 to 59 with annual family income below Rs 1 lakh receives Rs 1,000 per month, rising to Rs 1,500 if she has a child under 18. Applications are made at the block office or online with the husband's death certificate, an income certificate, Aadhaar and a bank passbook. The assistance stops on remarriage or at age 60, when the beneficiary is moved to the old-age pension without a fresh application. A life certificate is required every year in November."),
    ("senior_bus_pass", "Senior Citizen Bus Pass (illustrative scheme). Residents aged 60 and above travel free on all city and state transport corporation buses within the state, except air-conditioned and interstate services, which carry a 50 percent concession. The pass is issued at any bus depot on the same day with Aadhaar as age and address proof and one photograph, at a fee of Rs 50, and is valid for five years. A lost pass is replaced for Rs 100 at the issuing depot."),
    ("anna_suraksha", "Anna Suraksha Ration Entitlement (illustrative scheme). Priority ration card households receive 5 kilograms of foodgrain per member per month at Rs 2 per kilogram for wheat and Rs 3 per kilogram for rice, and Antyodaya households receive 35 kilograms per household regardless of size. Ration is distributed at the fair price shop from the 1st to the 12th of the month after biometric verification of any household member. Grain not collected in a month cannot be carried forward. Complaints against a dealer are made on the toll-free number 1967 or at the district supply office."),
    ("krishi_yantra", "Krishi Yantra Subsidy (illustrative scheme). Farmers get a 40 percent subsidy on tractors up to a subsidy of Rs 1 lakh, and 50 percent on power tillers, seed drills and threshers up to Rs 60,000 per implement. A farmer may claim the subsidy for one tractor in ten years and one implement of each kind in five years. Applications open on the agriculture portal in May and are drawn by lottery when they exceed the district allocation. The machine must be bought from a registered dealer within 45 days of the permit and the invoice uploaded for the subsidy to be released."),
    ("solar_rooftop", "Solar Rooftop Rebate (illustrative scheme). Domestic consumers installing a grid-connected rooftop solar system of 1 to 3 kilowatts receive a rebate of Rs 18,000 per kilowatt, and Rs 9,000 per kilowatt for capacity between 3 and 10 kilowatts. The system must be installed by a vendor empanelled with the distribution company and the net meter must be installed before the rebate is claimed. Applications are made on the distribution company's solar portal; the rebate is credited within 30 days of the net-meter commissioning report. Systems installed before application are not eligible."),
    ("sadak_maintenance", "Gramin Sadak Maintenance Grant (illustrative scheme). Every gram panchayat receives Rs 25,000 per kilometre per year for maintaining rural roads built under any government scheme that are more than five years old. The panchayat prepares a maintenance plan in the gram sabha and executes it through the panchayat's own works committee or a local contractor. Pothole repair, drain cleaning and shoulder repair are eligible; new construction is not. The block engineer inspects twice a year, and the next year's grant is withheld if the previous year's work is not certified."),
    ("shg_fund", "Mahila Swayam Sahayata Group Revolving Fund (illustrative scheme). A women's self-help group that has completed six months of regular meetings and savings receives a revolving fund of Rs 15,000, and a group that has repaid its first bank loan on time receives a community investment fund of Rs 60,000. The funds are lent to members at an interest rate the group decides, not above 12 percent a year. The group applies through the block mission office with its meeting register, savings passbook and bank statement. Groups with fewer than 10 members are not eligible."),
    ("free_coaching", "Free Coaching for Competitive Examinations (illustrative scheme). Students from families with annual income below Rs 3 lakh who have passed Class 12 with at least 60 percent marks can attend free coaching for state civil services, banking and railway examinations at the district coaching centre. Batches of 100 start in July and January; selection is by a screening test held in June and December. Students from outside the district town receive a hostel allowance of Rs 2,000 per month. A student may attend only one batch."),
    ("tribal_hostel", "Tribal Students' Hostel Admission (illustrative scheme). Students from Scheduled Tribe communities studying in Classes 6 to 12 or in a college more than 8 kilometres from their home can apply for a seat in a government tribal hostel. Applications are made at the hostel or the tribal welfare office by 15 June with the caste certificate, the previous marksheet and a residence certificate. Seats are allotted by merit within the community, with 30 percent reserved for girls. Boarding, lodging, books and uniforms are free, and each student receives Rs 500 per month for personal expenses."),
    ("chikitsa_sahayata", "Mukhya Mantri Chikitsa Sahayata (illustrative scheme). Families with annual income below Rs 3 lakh can receive up to Rs 3 lakh towards treatment of cancer, kidney failure, heart surgery, organ transplant or serious burns at any empanelled hospital in the state. The hospital applies on the family's behalf on the scheme portal with the treatment estimate, the income certificate and Aadhaar before treatment begins, except in emergencies, where the application may follow within 7 days. The amount is paid to the hospital, not the family. Assistance may be claimed once in three years per family."),
    ("sasta_dawa", "Sasta Dawa Kendra (illustrative scheme). Generic medicines are sold at 50 to 80 percent below branded prices at Sasta Dawa Kendras located in every district hospital, community health centre and at private outlets licensed by the state. No card or registration is needed; any person may buy with a prescription. Kendras must stock at least 200 essential generics and display the price list. A person who wants to open a Kendra applies to the state medical services corporation with a pharmacist's licence and a shop of at least 12 square metres, and receives a one-time grant of Rs 2 lakh for furniture and initial stock."),
    ("kanya_vivah", "Kanya Vivah Sahayata (illustrative scheme). A family with annual income below Rs 1.5 lakh receives Rs 51,000 on the marriage of a daughter aged 18 or above, paid to the bride's bank account. The application is made at the block office within 90 days after the marriage with the marriage certificate, the bride's age proof, the income certificate and Aadhaar. A family may claim for at most two daughters. Marriages registered more than 90 days after the ceremony are not eligible."),
    ("startup_seed", "Graduate Startup Seed Grant (illustrative scheme). A person who graduated within the last three years and registers a company or LLP in the state can apply for a seed grant of up to Rs 10 lakh, released in two halves against milestones. Applications are made on the startup portal with the degree certificate, the incorporation certificate and a business plan; a panel interviews shortlisted applicants within 60 days. The grant cannot be used for buying land or vehicles. A startup that has already raised more than Rs 50 lakh from investors is not eligible."),
    ("fisher_boat", "Fisherfolk Boat and Net Subsidy (illustrative scheme). Registered fisherfolk get a 50 percent subsidy, up to Rs 1.5 lakh, on a motorised boat, and 60 percent, up to Rs 15,000, on nets. Registration with the fisheries department needs proof of fishing as a livelihood, such as a cooperative society membership. Applications are made at the district fisheries office by 31 July each year; the subsidy is drawn by lottery when applications exceed funds. The boat must be insured, and the subsidy is recovered if it is sold within five years."),
    ("weaver_yarn", "Handloom Weaver Yarn Subsidy (illustrative scheme). Weavers holding a handloom identity card get a 10 percent subsidy on cotton and silk yarn bought from the state yarn depot, up to Rs 12,000 per weaver per year. The subsidy is deducted at the depot on presenting the card; no separate claim is needed. Cards are issued by the district handloom office with proof of a working loom, verified by a field visit. Master weavers who employ others may claim up to Rs 40,000 on their workers' behalf."),
    ("digital_literacy", "Digital Literacy Certificate Programme (illustrative scheme). Any resident aged 14 to 60 who has not passed Class 8 or has no computer training can attend a free 20-hour course at a common service centre covering phone and computer basics, online payments and government portals. On passing an online test, the person receives a certificate and the centre receives Rs 300 per certified trainee. One person per household is trained in the first round; a second member may enrol after the first round closes. Enrol at any common service centre with Aadhaar."),
    ("old_age_home", "Government Old Age Home Admission (illustrative scheme). Persons aged 60 and above who have no family able to support them, or who are destitute, may be admitted to a government old age home free of cost. Application is made to the district social welfare officer with an age proof, an income certificate or a destitution certificate from the tehsildar, and a medical fitness certificate. A person with a serious infectious disease or needing continuous medical care is referred to a hospital instead. Residents receive food, clothing, medical check-ups and Rs 300 per month as pocket money."),
    ("sports_scholarship", "State Sports Scholarship (illustrative scheme). Players aged 12 to 25 who have won a medal at a state championship receive Rs 2,000 per month, and those with a national medal Rs 5,000 per month, for two years from the date of the medal. Applications are made to the district sports officer within six months of the medal with the certificate, the age proof and a bank passbook. The scholarship continues only if the player participates in the next state championship in the same discipline. A player may hold only one sports scholarship at a time."),
    ("disaster_exgratia", "Disaster Relief Ex-gratia (illustrative scheme). Families who lose a member in a notified natural disaster receive Rs 4 lakh, and those whose house is fully destroyed receive Rs 1.2 lakh in rural areas and Rs 1.5 lakh in urban areas. Partial house damage is compensated at Rs 32,000 for pucca houses and Rs 15,000 for kutcha houses. The tehsildar's assessment team records the loss within 15 days; families with a recorded loss need not apply. Crop loss of more than 33 percent is compensated at Rs 8,500 per hectare for irrigated land and Rs 6,800 for unirrigated land, up to 2 hectares."),
    ("senior_tax_rebate", "Senior Citizen Property Tax Rebate (illustrative scheme). Owners aged 65 and above who occupy their own residential property get a 30 percent rebate on property tax, and widows and persons with disabilities of any age get 40 percent. The rebate applies to one property per person. Application is made once at the ward office with the age proof, the title document and a self-declaration of occupation; the rebate then continues each year. It is not available on property let out for rent, and false declaration leads to recovery with a penalty of twice the rebate."),
    ("grievance_escalation", "Grievance Escalation Procedure (illustrative). A complaint filed on the state grievance portal is assigned to the concerned officer, who must respond within 15 working days. If the complainant marks the reply unsatisfactory, the complaint escalates to the district head of the department, who must respond within 10 more working days, and then to the departmental secretary. A complaint not answered within 30 working days at any level is reported to the chief minister's office. Complainants receive an SMS at each stage and may check status with the complaint number on the portal or on the helpline 181."),
    ("edistrict_timelines", "e-District Certificate Service Timelines (illustrative). Certificates applied for at any common service centre or on the e-District portal are issued within fixed limits: income certificate 7 working days, caste certificate 15 working days, domicile certificate 7 working days, birth and death certificates 3 working days when the event is already registered, and a duplicate ration card 10 working days. The fee is Rs 30 per certificate, paid at the centre. If the limit is exceeded, the applicant may appeal to the sub-divisional officer, who must decide within 15 days, and the responsible official may be fined Rs 250 per day of delay up to Rs 5,000."),
    ("scholarship_documents", "Scholarship Portal Document Checklist (illustrative). All scholarships on the state portal require the same core documents uploaded as PDF under 500 KB each: Aadhaar of the student, an income certificate not older than one year, the previous year's marksheet, the current year's fee receipt or bonafide certificate, a bank passbook page showing the account number and IFSC, and one photograph. Caste-based scholarships also need the caste certificate. Applications missing any document are returned once for correction with a 15-day window; a second rejection is final for that year."),
    ("labour_registration", "Labour Welfare Board Registration (illustrative). Workers in construction, brick kilns, loading, domestic work and street vending aged 18 to 60 can register with the state labour welfare board at the district labour office or online. The fee is Rs 25 for registration and Rs 10 per year for renewal. Registration needs an age proof, a 90-day work certificate from an employer, contractor or trade union, Aadhaar and a bank passbook. Registered workers are eligible for the accident cover, a maternity benefit of Rs 10,000, a tool purchase grant of Rs 3,000 and scholarships for children. Registration lapses if not renewed for two consecutive years."),
]

# (language, question, gold answer, gold doc_id). All answerable from one passage.
QA: list[tuple[str, str, str, str]] = [
    ("en", "How much does the Sahyog Vridh Pension Top-up add to the old-age pension each month?", "Rs 500 per month, for BPL pensioners aged 60 or above.", "sahyog_pension"),
    ("en", "When must the life certificate be submitted to keep the Sahyog pension top-up?", "Between 1 November and 31 December each year.", "sahyog_pension"),
    ("hi", "विद्या सेतु छात्रवृत्ति में कितनी राशि मिलती है और किस्तें कब आती हैं?", "प्रति वर्ष 6,000 रुपये, 3,000 रुपये की दो किस्तों में, सितंबर और फरवरी में।", "vidya_setu"),
    ("en", "What is the last date to apply for the Vidya Setu Girls' Scholarship?", "31 August, on the state scholarship portal.", "vidya_setu"),
    ("hinglish", "Vidya Setu scholarship ke liye attendance kitni honi chahiye?", "At least 75 percent attendance in the previous year; below that the student is disqualified for that year.", "vidya_setu"),
    ("en", "What subsidy does a marginal farmer with 1.5 hectares get on drip irrigation under Kisan Jal Sinchai?", "75 percent, up to a maximum of Rs 80,000 per farmer.", "kisan_jal"),
    ("hi", "किसान जल सिंचाई सब्सिडी में अनुमोदन से पहले खरीदे गए उपकरण का क्या होता है?", "अनुमोदन से पहले की गई खरीद की प्रतिपूर्ति नहीं होती; उपकरण अनुमोदन के बाद पैनलबद्ध आपूर्तिकर्ता से खरीदना होता है।", "kisan_jal"),
    ("en", "How much of the Gram Ujala streetlight grant is paid in advance?", "80 percent in advance; the remaining 20 percent after the block engineer inspects the installation.", "gram_ujala"),
    ("hinglish", "Gram Ujala grant se generator kharid sakte hain kya?", "No. Generators are not an eligible expense; repair of existing lights is.", "gram_ujala"),
    ("en", "Up to what loan amount is no collateral needed under the Nari Udyam Loan Scheme?", "Loans up to Rs 2 lakh need no collateral.", "nari_udyam"),
    ("hi", "नारी उद्यम ऋण योजना में जिला समिति कितने दिनों में निर्णय लेती है?", "समिति हर महीने बैठती है और 45 दिनों के भीतर निर्णय लेती है।", "nari_udyam"),
    ("en", "What does Shramik Suraksha pay on the accidental death of a registered construction worker?", "Rs 5 lakh to the nominee named at registration; Rs 2.5 lakh on permanent disability.", "shramik_suraksha"),
    ("hinglish", "Shramik Suraksha ka claim kitne time ke andar file karna hota hai?", "Within one year of the accident, at the district labour office, with the FIR or accident report and the hospital record.", "shramik_suraksha"),
    ("en", "How is the Rs 6,000 Matru Poshan maternity benefit split?", "Three instalments of Rs 2,000: on pregnancy registration in the first trimester, after the second antenatal check-up, and after the child's first vaccinations.", "matru_poshan"),
    ("hi", "मातृ पोषण लाभ के लिए गर्भावस्था का पंजीकरण कब तक कराना होता है?", "अंतिम माहवारी की तारीख से 150 दिनों के भीतर।", "matru_poshan"),
    ("en", "What attendance is needed to receive the Yuva Kaushal stipend for a month?", "80 percent attendance in the month.", "yuva_kaushal"),
    ("en", "How is the Awas Sahayata housing grant of Rs 1.3 lakh paid out?", "In three stages: Rs 40,000 on sanction, Rs 60,000 after the plinth inspection, and Rs 30,000 after the roof is cast.", "awas_sahayata"),
    ("hi", "आवास सहायता अनुदान के लिए मकान का न्यूनतम आकार क्या है?", "कम से कम 25 वर्ग मीटर।", "awas_sahayata"),
    ("en", "How much is the Swachh Gram toilet incentive and when is it paid?", "Rs 12,000, paid within 30 days after the gram panchayat verifies construction with a geotagged photograph.", "swachh_gram"),
    ("hinglish", "Balika Janm Samriddhi mein kitna paisa jama hota hai aur kab milta hai?", "Rs 25,000 is deposited in a fixed deposit in the girl's name; the maturity amount is paid at 18 if she has passed Class 10 and is unmarried.", "balika_samriddhi"),
    ("en", "Within what time must a family apply for Balika Janm Samriddhi?", "Within one year of the girl's birth, at the district women and child development office.", "balika_samriddhi"),
    ("hi", "दिव्यांग सहायक योजना में उपकरण पाने के लिए कितने प्रतिशत दिव्यांगता चाहिए?", "दिव्यांगता प्रमाण पत्र में 40 प्रतिशत या अधिक।", "divyang_sahayak"),
    ("en", "When are the Divyang Sahayak assessment camps held in each block?", "Twice a year, in March and September; devices are also issued at the district hospital on any working day.", "divyang_sahayak"),
    ("en", "By what date must a farmer register to sell wheat at the procurement centre?", "By 15 March for wheat (15 September for paddy).", "fasal_kharid"),
    ("hinglish", "Fasal kharid ke baad payment kitne time mein aata hai?", "Within 72 hours of the weighment slip, to the bank account.", "fasal_kharid"),
    ("en", "What share of the premium does the state pay under Pashu Dhan Bima?", "70 percent; the owner pays the rest, about Rs 150 per cow.", "pashu_bima"),
    ("hi", "पशु धन बीमा में पशु की मृत्यु की सूचना कितने समय में देनी होती है?", "24 घंटे के भीतर पशु चिकित्सा कार्यालय को।", "pashu_bima"),
    ("en", "Who gets a bicycle under the Chhatra Cycle Yojana?", "Students entering Class 9 in a government school more than 2 kilometres from their home.", "chhatra_cycle"),
    ("en", "How many hours a day must a panchayat library be open to keep its grant?", "At least four hours a day, six days a week, free to all residents.", "panchayat_library"),
    ("hinglish", "Shehri Vendor Card ki fees kitni hai aur kitne saal valid hai?", "Rs 100, valid for five years.", "vendor_card"),
    ("en", "How much notice must be given before a vendor card holder is removed from an allotted spot?", "30 days' notice.", "vendor_card"),
    ("en", "What property tax rebate does rainwater harvesting earn under Jal Sanchay, and for how long?", "A 10 percent rebate for five years, starting the financial year after certification.", "jal_sanchay"),
    ("hi", "बिजली सहायता योजना में पहली 100 यूनिट की दर क्या है?", "1 रुपये प्रति यूनिट; 100 यूनिट से अधिक खपत होने पर पूरा बिल सामान्य दर पर लगता है।", "bijli_sahayata"),
    ("en", "How much does Vidhwa Sahayata pay a widow who has a child under 18?", "Rs 1,500 per month (Rs 1,000 without a child under 18).", "vidhwa_sahayata"),
    ("hinglish", "Senior citizen bus pass AC bus mein chalta hai kya?", "AC and interstate services are not free; they carry a 50 percent concession.", "senior_bus_pass"),
    ("en", "How much foodgrain does an Antyodaya household get per month under Anna Suraksha?", "35 kilograms per household regardless of size.", "anna_suraksha"),
    ("hi", "अन्न सुरक्षा में राशन किस तारीख से किस तारीख तक बंटता है?", "हर महीने की 1 से 12 तारीख तक, बायोमेट्रिक सत्यापन के बाद।", "anna_suraksha"),
    ("en", "How often can a farmer claim the Krishi Yantra tractor subsidy?", "One tractor in ten years, and one implement of each kind in five years.", "krishi_yantra"),
    ("en", "What is the Solar Rooftop Rebate for a 5 kilowatt system?", "Rs 18,000 per kilowatt for the first 3 kilowatts and Rs 9,000 per kilowatt for the next 2, so Rs 72,000.", "solar_rooftop"),
    ("hi", "ग्रामीण सड़क रखरखाव अनुदान में प्रति किलोमीटर कितनी राशि मिलती है?", "प्रति किलोमीटर प्रति वर्ष 25,000 रुपये, पाँच वर्ष से पुरानी सड़कों के लिए।", "sadak_maintenance"),
    ("en", "How much revolving fund does a self-help group get after six months of regular meetings?", "Rs 15,000; a group that has repaid its first bank loan on time gets a community investment fund of Rs 60,000.", "shg_fund"),
    ("hinglish", "Free coaching ke liye Class 12 mein kitne marks chahiye?", "At least 60 percent, with family income below Rs 3 lakh.", "free_coaching"),
    ("en", "What share of tribal hostel seats is reserved for girls?", "30 percent.", "tribal_hostel"),
    ("hi", "मुख्यमंत्री चिकित्सा सहायता में राशि किसे दी जाती है, परिवार को या अस्पताल को?", "अस्पताल को, परिवार को नहीं।", "chikitsa_sahayata"),
    ("en", "How often can a family claim Mukhya Mantri Chikitsa Sahayata?", "Once in three years per family.", "chikitsa_sahayata"),
    ("en", "Within how many days after a marriage must Kanya Vivah Sahayata be applied for?", "Within 90 days after the marriage, at the block office.", "kanya_vivah"),
    ("hi", "कन्या विवाह सहायता में कितनी राशि मिलती है और किसके खाते में?", "51,000 रुपये, दुल्हन के बैंक खाते में।", "kanya_vivah"),
    ("hinglish", "Startup seed grant land ya gaadi kharidne mein use kar sakte hain?", "No. The grant cannot be used for buying land or vehicles.", "startup_seed"),
    ("en", "What is the maximum boat subsidy for registered fisherfolk?", "50 percent, up to Rs 1.5 lakh; nets get 60 percent up to Rs 15,000.", "fisher_boat"),
    ("en", "How does a handloom weaver claim the yarn subsidy?", "It is deducted at the state yarn depot on presenting the handloom identity card; no separate claim is needed.", "weaver_yarn"),
    ("hi", "डिजिटल साक्षरता प्रमाण पत्र कार्यक्रम का कोर्स कितने घंटे का है?", "20 घंटे का निःशुल्क कोर्स, जन सेवा केंद्र पर।", "digital_literacy"),
    ("en", "How much pocket money do residents of a government old age home receive?", "Rs 300 per month.", "old_age_home"),
    ("hinglish", "State sports scholarship national medal par kitna milta hai?", "Rs 5,000 per month for two years from the date of the medal (Rs 2,000 for a state medal).", "sports_scholarship"),
    ("en", "How much ex-gratia is paid for a fully destroyed house in a rural area after a notified disaster?", "Rs 1.2 lakh in rural areas (Rs 1.5 lakh in urban areas).", "disaster_exgratia"),
    ("hi", "वरिष्ठ नागरिक संपत्ति कर छूट किराये पर दी गई संपत्ति पर मिलती है क्या?", "नहीं, किराये पर दी गई संपत्ति पर छूट नहीं मिलती।", "senior_tax_rebate"),
    ("en", "Within how many working days must an officer respond to a complaint on the grievance portal?", "15 working days; on escalation the district head gets 10 more working days.", "grievance_escalation"),
    ("hinglish", "Caste certificate e-District se kitne din mein milta hai?", "15 working days; income and domicile certificates take 7 working days.", "edistrict_timelines"),
    ("en", "What is the maximum file size for each scholarship portal document?", "500 KB per PDF.", "scholarship_documents"),
    ("hi", "श्रम कल्याण बोर्ड पंजीकरण के लिए कार्य प्रमाण पत्र कितने दिनों का चाहिए?", "90 दिनों का कार्य प्रमाण पत्र, नियोक्ता, ठेकेदार या ट्रेड यूनियन से।", "labour_registration"),
]

# On-topic questions the corpus does not answer. The right answer is to say so.
UNANSWERABLE: list[tuple[str, str]] = [
    ("en", "What is the income limit for the Sahyog Vridh Pension Top-up in urban areas?"),
    ("hi", "विद्या सेतु छात्रवृत्ति के लिए हेल्पलाइन नंबर क्या है?"),
    ("en", "How many Sasta Dawa Kendras are there in the state?"),
    ("hinglish", "Kisan Jal Sinchai subsidy mein solar pump bhi aata hai kya?"),
    ("en", "Which bank handles the Nari Udyam loans?"),
    ("hi", "अन्न सुरक्षा में दाल और तेल कितना मिलता है?"),
]


def main() -> None:
    docs = []
    for doc_id, text in DOCS:
        n = len(text.split())
        assert 40 <= n <= 180, f"{doc_id}: {n} words (must be one chunk)"
        docs.append({"doc_id": doc_id, "text": text,
                     "source_uri": f"illustrative://in_scheme_qa/{doc_id}"})
    CORPUS.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs),
                      encoding="utf-8")
    ids = {d["doc_id"] for d in docs}

    rows = []
    for i, (lang, q, a, doc) in enumerate(QA, start=1):
        assert doc in ids, f"qa {i}: unknown doc {doc}"
        rows.append({"item_id": f"sq{i:02d}", "query": q, "item_type": "answerable",
                     "gold_answer": a, "gold_passage_ids": [f"{doc}#0"],
                     "meta": {"language": lang}})
    for j, (lang, q) in enumerate(UNANSWERABLE, start=1):
        rows.append({"item_id": f"su{j:02d}", "query": q, "item_type": "unanswerable",
                     "gold_answer": None, "gold_passage_ids": [],
                     "meta": {"language": lang}})
    BASE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")

    # Derived probes, generated with the profile's settings so the evalset is
    # a pure function of this file (and the builder stays idempotent).
    base_items = load_evalset(str(BASE))
    suite = build_probe_suite(base_items, PROBES)
    save_evalset(base_items + suite.items, str(EVALSET))
    langs = {lang: sum(1 for r in rows if r["meta"]["language"] == lang)
             for lang in ("en", "hi", "hinglish")}
    print(f"wrote {len(docs)} passages, {len(rows)} base items {langs}, "
          f"{len(suite.items)} probes {suite.counts} -> {EVALSET}")


if __name__ == "__main__":
    main()
