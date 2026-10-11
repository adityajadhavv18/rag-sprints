"""Generate the synthetic medical corpus used from the Hybrid sprint onward.

Everything here is FICTIONAL: Halverd Valley Medical Center (HVMC), its drugs, lab codes and the
Kestrel IV-220 pump do not exist. It is test data for retrieval, not medical guidance.

Built to stress both retrieval paths:
  - exact strings that embeddings blur: near-identical codes (HV-RX-2047 vs HV-RX-2074, E-17 vs E-71,
    LC-0412 vs LC-0421) and look-alike invented drug names (Lumidex vs Lumadex)
  - clinical concepts that keyword search misses when a question is phrased in plain language
    ("blood sugar crashed" vs "hypoglycemia", "poor kidney function" vs "renal impairment")

Data lives in the tables below; the script only renders them. Deterministic: same input, same files.

Run from rag_sprints/:  uv run scripts/generate_medical_corpus.py
"""

import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "medical"

DISCLAIMER = ("SYNTHETIC DOCUMENT: fictional facility, drugs and devices, created for retrieval testing. "
              "Not medical guidance.")


# ---------- 1. Formulary ----------

# (code, name, drug class, status, indications, dosing, renal, interactions, monitoring, notes)
FORMULARY = {
    "Anticoagulants and reversal agents": [
        ("HV-RX-2010", "Heparivex", "low molecular weight heparin", "Tier 1, unrestricted.",
         "Prevention of venous thromboembolism (VTE) in hospitalized adults; treatment of deep vein thrombosis "
         "(DVT) and pulmonary embolism (PE).",
         "VTE prophylaxis: 40 mg subcutaneously once daily. Treatment: 1 mg/kg subcutaneously every 12 hours.",
         "CrCl below 30 mL/min: prophylaxis 30 mg once daily; treatment 1 mg/kg once daily.",
         "Additive bleeding risk with antiplatelet agents and Ketrolin.",
         "Platelet count every 2 to 3 days for the first 14 days to detect heparin-induced thrombocytopenia. "
         "Anti-Xa level (LC-0920) in patients over 150 kg or with CrCl below 30 mL/min.",
         "Do not expel the air bubble from the prefilled syringe before injection."),
        ("HV-RX-2047", "Corvatrel", "oral factor Xa inhibitor", "Tier 1, unrestricted.",
         "Stroke prevention in non-valvular atrial fibrillation; treatment of DVT and PE.",
         "15 mg by mouth once daily with the evening meal.",
         "CrCl 15 to 50 mL/min: 10 mg once daily. CrCl below 15 mL/min: avoid use.",
         "Avoid combination with strong CYP3A4 inhibitors such as Nephrazole, which roughly doubles Corvatrel "
         "exposure.",
         "CBC and serum creatinine at baseline, then renal function every 6 months.",
         "No specific reversal agent is stocked. For life-threatening bleeding, contact hematology for "
         "four-factor prothrombin complex concentrate."),
        ("HV-RX-2074", "Corvetrol", "oral direct thrombin inhibitor", "Tier 2, pharmacist verification required.",
         "Stroke prevention in non-valvular atrial fibrillation.",
         "110 mg by mouth twice daily. Capsules must not be opened or crushed.",
         "CrCl 30 to 50 mL/min: 75 mg twice daily. CrCl below 30 mL/min: contraindicated.",
         "P-glycoprotein inhibitors increase Corvetrol levels.",
         "aPTT (LC-0625) may be used as a qualitative indicator of drug presence.",
         "The reversal agent is Revexa (HV-RX-2081)."),
        ("HV-RX-2081", "Revexa", "monoclonal antibody fragment reversal agent",
         "Restricted: requires approval from the hematology or emergency department attending.",
         "Reversal of Corvetrol before emergency surgery or in life-threatening bleeding.",
         "5 g IV, given as two consecutive 2.5 g boluses no more than 15 minutes apart.",
         "No adjustment.",
         "Does not reverse Corvatrel or Heparivex.",
         "aPTT 2 to 4 hours after administration.",
         "Stored refrigerated in the emergency department automated dispensing cabinet. Restart anticoagulation "
         "within 24 hours once hemostasis is achieved."),
    ],
    "Anti-infectives": [
        ("HV-RX-1102", "Zephacillin", "extended-spectrum penicillin with beta-lactamase inhibitor",
         "Tier 1; infectious diseases (ID) approval required to continue beyond 72 hours.",
         "Empiric therapy for sepsis of unknown source, hospital-acquired pneumonia and intra-abdominal infection.",
         "4.5 g IV every 6 hours. The first dose is given over 30 minutes; later doses run as an extended "
         "infusion over 4 hours, which is the HVMC standard.",
         "CrCl 20 to 40 mL/min: 3.375 g every 6 hours. CrCl below 20 mL/min: 2.25 g every 6 hours.",
         "Increased nephrotoxicity when combined with Vancorin.",
         "Serum creatinine daily; CBC twice weekly.",
         "First-line agent in the sepsis order set (CP-07)."),
        ("HV-RX-1108", "Cefalyn", "third-generation cephalosporin", "Tier 1, unrestricted.",
         "Community-acquired pneumonia, pyelonephritis and bacterial meningitis.",
         "2 g IV once daily. Bacterial meningitis: 2 g IV every 12 hours.",
         "No adjustment.",
         "Do not co-administer with calcium-containing IV solutions in the same line.",
         "CBC weekly; watch for biliary sludge with prolonged courses.",
         "Cefalyn plus Azithrel is the standard ward regimen for community-acquired pneumonia."),
        ("HV-RX-1120", "Meronex", "carbapenem",
         "Restricted: ID approval required before the first dose, except in septic shock, where approval may be "
         "obtained within 24 hours.",
         "Infections with ESBL-producing organisms and severe nosocomial infections.",
         "1 g IV every 8 hours. Central nervous system infections: 2 g IV every 8 hours.",
         "CrCl 26 to 50 mL/min: 1 g every 12 hours. CrCl 10 to 25 mL/min: 500 mg every 12 hours.",
         "Lowers valproic acid levels by up to 90 percent; avoid the combination and choose an alternative "
         "antiepileptic.",
         "Seizure precautions in patients with central nervous system disease.",
         "Reserve for documented resistance to protect the hospital antibiogram."),
        ("HV-RX-1135", "Vancorin", "glycopeptide antibiotic with MRSA coverage",
         "Tier 1; dosed by pharmacy under the pharmacy-to-dose protocol.",
         "Suspected or confirmed MRSA infection; empiric sepsis coverage together with Zephacillin.",
         "Loading dose 25 mg/kg IV (maximum 3,000 mg), then maintenance dosing targeting an AUC of 400 to 600. "
         "Infuse no faster than 1 g per hour.",
         "Pharmacy adjusts the dose using drug levels.",
         "Nephrotoxicity is increased with Zephacillin and Ketrolin.",
         "Drug levels per LC-0910; serum creatinine daily.",
         "Flushing of the upper body during a fast infusion is an infusion reaction, not an allergy: slow the "
         "rate."),
        ("HV-RX-1150", "Azithrel", "macrolide antibiotic", "Tier 1, unrestricted.",
         "Community-acquired pneumonia (atypical coverage).",
         "500 mg IV or by mouth once daily for 3 days.",
         "No adjustment.",
         "QT prolongation with Lumidex, Halopran and Tarnaflox; obtain an ECG if combined.",
         "Baseline QTc in patients with cardiac disease.",
         "Switch from IV to oral once the patient tolerates oral intake."),
        ("HV-RX-1172", "Tarnaflox", "fluoroquinolone",
         "Restricted to patients with a documented beta-lactam allergy or to susceptibility-guided therapy.",
         "Pneumonia and complicated urinary tract infection.",
         "750 mg IV or by mouth once daily.",
         "CrCl 20 to 49 mL/min: 750 mg every 48 hours.",
         "Oral absorption is reduced by antacids, iron and magnesium; separate doses by at least 2 hours. "
         "Additive QT prolongation.",
         "Ask about tendon pain at every assessment.",
         "Boxed warning for tendon rupture, higher in patients over 60 and in those taking corticosteroids such "
         "as Prednavex. Avoid in myasthenia gravis."),
        ("HV-RX-1190", "Nephrazole", "triazole antifungal", "Tier 2.",
         "Invasive candidiasis.",
         "Loading dose 800 mg IV on day 1, then 400 mg once daily.",
         "CrCl below 50 mL/min: after the loading dose, give 200 mg once daily.",
         "Strong CYP3A4 inhibitor: avoid with Corvatrel.",
         "Liver enzymes (LC-0510, LC-0512) weekly.",
         "Oral and IV doses are interchangeable."),
    ],
    "Endocrine and glucose management": [
        ("HV-RX-3150", "Lumadex", "long-acting basal insulin analog",
         "Tier 1. High-alert medication: independent double check by two nurses required.",
         "Basal insulin for type 1 and type 2 diabetes.",
         "Insulin-naive adults: 0.2 units/kg subcutaneously once daily at bedtime.",
         "eGFR below 30: reduce the starting dose to 0.1 units/kg.",
         "Never mix in the same syringe as Glyvanta.",
         "Point-of-care glucose before meals and at bedtime.",
         "Do not hold basal insulin completely for patients who are NPO; give 80 percent of the usual dose "
         "instead."),
        ("HV-RX-3160", "Glyvanta", "rapid-acting insulin analog",
         "Tier 1. High-alert medication: independent double check by two nurses required.",
         "Mealtime and correction insulin.",
         "Give 0 to 15 minutes before meals. Correction scale: glucose 151 to 200 mg/dL, 2 units; 201 to 250, "
         "4 units; 251 to 300, 6 units; 301 to 350, 8 units; above 350, 10 units and notify the provider.",
         "No adjustment; titrate to glucose.",
         "Beta blockers such as Metavol can mask the warning signs of low glucose.",
         "Point-of-care glucose before meals and at bedtime.",
         "If the patient is not eating, give the correction dose only, never the mealtime dose."),
        ("HV-RX-3190", "Dextriva", "dextrose 50% injection", "Tier 1, floor stock.",
         "Hypoglycemia (blood glucose below 70 mg/dL) in patients unable to take oral carbohydrate who have IV "
         "access.",
         "25 g (50 mL) IV push over 2 to 5 minutes. May repeat once after 15 minutes if glucose remains below "
         "70 mg/dL.",
         "No adjustment.",
         "None clinically relevant.",
         "Confirm IV patency before administration: extravasation causes tissue necrosis.",
         "Part of the hypoglycemia protocol (CP-03)."),
        ("HV-RX-3195", "Glucarin", "intranasal glucagon", "Tier 1, floor stock.",
         "Severe hypoglycemia in a patient without IV access.",
         "3 mg into one nostril. May repeat after 15 minutes using a new device.",
         "No adjustment.",
         "None clinically relevant.",
         "Recheck glucose 15 minutes after the dose.",
         "Ineffective when glycogen stores are depleted, as in chronic alcohol use or starvation. Vomiting is "
         "common: place the patient on their side."),
        ("HV-RX-7050", "Thyrolex", "synthetic thyroid hormone", "Tier 1, unrestricted.",
         "Hypothyroidism.",
         "1.6 mcg/kg by mouth daily on an empty stomach. Older adults or patients with heart disease: start at "
         "25 mcg daily.",
         "No adjustment.",
         "Separate from iron and calcium supplements by 4 hours; Pantorex reduces absorption.",
         "TSH (LC-0870) 6 to 8 weeks after any dose change.",
         "Doses are not interchangeable between brands without TSH follow-up."),
        ("HV-RX-7065", "Prednavex", "intravenous corticosteroid", "Tier 1, unrestricted.",
         "Septic shock with an ongoing vasopressor requirement; adrenal insufficiency.",
         "Septic shock: 50 mg IV every 6 hours when the Norvessa dose has been 0.25 mcg/kg/min or higher for at "
         "least 4 hours. Taper once off vasopressors.",
         "No adjustment.",
         "Raises tendon rupture risk with Tarnaflox.",
         "Glucose every 6 hours because of steroid-induced hyperglycemia.",
         "Stress-dose steroids for adrenal insufficiency are ordered separately."),
    ],
    "Antiemetics and gastrointestinal": [
        ("HV-RX-3105", "Lumidex", "5-HT3 receptor antagonist antiemetic", "Tier 1, unrestricted.",
         "Postoperative and chemotherapy-induced nausea and vomiting.",
         "4 mg IV every 8 hours as needed. Maximum 16 mg per day.",
         "No adjustment.",
         "QT prolongation with Azithrel, Tarnaflox and Halopran; serotonin syndrome risk with serotonergic "
         "agents.",
         "ECG in patients with electrolyte abnormalities; correct potassium and magnesium first.",
         "Constipation is common."),
        ("HV-RX-3110", "Ondelix", "dopamine antagonist antiemetic and prokinetic", "Tier 1, unrestricted.",
         "Nausea, diabetic gastroparesis.",
         "10 mg IV every 6 hours. Do not exceed 5 days of therapy.",
         "CrCl below 40 mL/min: 5 mg every 6 hours.",
         "Additive extrapyramidal effects with Halopran; avoid in Parkinson disease.",
         "Watch for extrapyramidal symptoms such as acute dystonia and restlessness.",
         "Contraindicated in bowel obstruction."),
        ("HV-RX-5010", "Pantorex", "intravenous proton pump inhibitor", "Tier 1, unrestricted.",
         "Upper gastrointestinal bleeding; stress ulcer prophylaxis in selected ICU patients.",
         "40 mg IV once daily. Upper GI bleed: 80 mg IV bolus, then 40 mg IV every 12 hours.",
         "No adjustment.",
         "Reduces absorption of Thyrolex.",
         "Magnesium level with courses longer than 2 weeks.",
         "Stress ulcer prophylaxis is indicated only for ICU patients on mechanical ventilation for more than 48 "
         "hours or with coagulopathy (platelets below 50 or INR above 1.5). Discontinue at transfer out of the "
         "ICU."),
        ("HV-RX-5030", "Senolax", "stimulant laxative", "Tier 1, floor stock.",
         "Prevention of opioid-induced constipation.",
         "8.6 mg tablets: 2 tablets by mouth at bedtime, up to 4 tablets twice daily.",
         "No adjustment.",
         "None clinically relevant.",
         "Document bowel movements every shift.",
         "Started automatically with any scheduled opioid order under the HVMC bowel regimen. Hold for loose "
         "stools."),
    ],
    "Analgesia and sedation": [
        ("HV-RX-4010", "Morphalin", "opioid agonist", "High-alert medication; Schedule II.",
         "Severe acute pain.",
         "2 to 4 mg IV every 3 hours as needed. Opioid-naive older adults: 1 mg.",
         "CrCl below 30 mL/min: avoid, because an active metabolite accumulates; use Hydrovane instead.",
         "Additive respiratory depression with Calmazine, Pregavel and Sedoflux.",
         "Sedation score (POSS) and respiratory rate 30 minutes after each IV dose. Naloxar must be available "
         "on the unit.",
         "Start Senolax with any scheduled order."),
        ("HV-RX-4022", "Hydrovane", "opioid agonist", "High-alert medication; Schedule II.",
         "Severe acute pain.",
         "0.2 to 0.6 mg IV every 2 hours as needed.",
         "Preferred opioid in renal impairment. CrCl below 30 mL/min: reduce the dose by 50 percent.",
         "Additive respiratory depression with benzodiazepines and gabapentinoids.",
         "Sedation score (POSS) and respiratory rate 30 minutes after each IV dose.",
         "Hydrovane is about 5 times more potent than Morphalin per mg; confusing the two is the most common "
         "opioid conversion error at HVMC."),
        ("HV-RX-4090", "Naloxar", "opioid antagonist", "Tier 1, floor stock.",
         "Opioid-induced respiratory depression.",
         "0.04 mg IV every 2 minutes, titrated to a respiratory rate of at least 10 breaths per minute. The full "
         "0.4 mg dose is reserved for cardiopulmonary arrest.",
         "No adjustment.",
         "Reverses analgesia as well as respiratory depression.",
         "Monitor for at least 2 hours after the last dose, because Naloxar wears off before most opioids.",
         "See protocol CP-15."),
        ("HV-RX-4100", "Acetavol", "intravenous acetaminophen",
         "Restricted to patients unable to take oral medication; convert to oral within 24 hours.",
         "Mild to moderate pain and fever.",
         "1 g IV every 6 hours. Maximum 4 g per day from all sources. Reduce the maximum to 3 g per day in "
         "patients weighing under 50 kg, with chronic alcohol use, or with hepatic impairment.",
         "No adjustment.",
         "Count acetaminophen in combination products toward the daily maximum.",
         "Liver enzymes if therapy exceeds 5 days.",
         "Opioid-sparing first step in the HVMC pain ladder."),
        ("HV-RX-4115", "Ketrolin", "intravenous NSAID", "Tier 1, unrestricted.",
         "Moderate to severe acute pain.",
         "15 mg IV every 6 hours. Maximum 5 days of total therapy.",
         "CrCl below 30 mL/min or acute kidney injury: avoid.",
         "Bleeding risk with Heparivex, Corvatrel and Corvetrol; nephrotoxicity with Vancorin.",
         "Serum creatinine every 48 hours.",
         "Contraindicated with a history of gastrointestinal bleeding or after coronary artery bypass surgery."),
        ("HV-RX-4130", "Pregavel", "gabapentinoid", "Tier 1, unrestricted.",
         "Neuropathic pain.",
         "75 mg by mouth twice daily; maximum 300 mg twice daily.",
         "CrCl 30 to 60 mL/min: halve the dose. CrCl below 30 mL/min: 25 to 75 mg once daily.",
         "Respiratory depression when combined with opioids.",
         "Dizziness and sedation.",
         "Dizziness raises fall risk: apply fall precautions (CP-11)."),
        ("HV-RX-4205", "Sedoflux", "alpha-2 agonist sedative", "Restricted to the ICU.",
         "Light sedation of mechanically ventilated patients.",
         "0.2 to 1.4 mcg/kg/h continuous infusion. No loading dose.",
         "No adjustment.",
         "Additive bradycardia with Metavol.",
         "Continuous cardiac monitoring for bradycardia and hypotension.",
         "Target a RASS of 0 to -2."),
        ("HV-RX-4220", "Halopran", "antipsychotic",
         "Tier 1, for agitation from delirium after non-drug measures have failed.",
         "Agitation in delirium.",
         "0.5 to 1 mg IV every 30 minutes as needed; maximum 5 mg in 24 hours in older adults.",
         "No adjustment.",
         "QT prolongation with Lumidex and Azithrel; avoid in Parkinson disease and Lewy body dementia.",
         "Baseline QTc; hold if the QTc is above 500 ms.",
         "Not for sleep or for routine agitation without delirium."),
        ("HV-RX-4235", "Calmazine", "benzodiazepine", "Tier 1; Schedule IV.",
         "Alcohol withdrawal; acute seizures.",
         "Alcohol withdrawal: symptom-triggered dosing, 1 to 2 mg IV every hour while the CIWA-Ar score is 10 "
         "or more.",
         "No adjustment.",
         "Respiratory depression with opioids.",
         "CIWA-Ar score every hour during active withdrawal.",
         "Paradoxical agitation can occur in older adults. See protocol CP-17."),
    ],
    "Cardiovascular and vasopressors": [
        ("HV-RX-6010", "Furovex", "loop diuretic", "Tier 1, unrestricted.",
         "Fluid overload, acute heart failure.",
         "20 to 40 mg IV. Infuse no faster than 4 mg per minute to avoid ototoxicity.",
         "Higher doses may be required in renal impairment.",
         "Hypokalemia increases toxicity of QT-prolonging drugs.",
         "Potassium and magnesium daily; daily weights; strict intake and output.",
         "Give early in the day to avoid nighttime diuresis."),
        ("HV-RX-6025", "Spirolan", "aldosterone antagonist", "Tier 1, unrestricted.",
         "Heart failure with reduced ejection fraction; ascites.",
         "25 mg by mouth once daily.",
         "Hold if CrCl is below 30 mL/min.",
         "Hyperkalemia with Kaliveen and potassium supplements.",
         "Potassium within 3 days of starting.",
         "Hold if potassium is above 5.0 mmol/L."),
        ("HV-RX-6040", "Amlotra", "dihydropyridine calcium channel blocker", "Tier 1, unrestricted.",
         "Hypertension.",
         "5 mg by mouth once daily; maximum 10 mg once daily.",
         "No adjustment.",
         "Levels increased by strong CYP3A4 inhibitors.",
         "Blood pressure daily.",
         "Ankle swelling is the most common side effect."),
        ("HV-RX-6055", "Metavol", "beta blocker", "Tier 1, unrestricted.",
         "Hypertension, rate control in atrial fibrillation, post-myocardial infarction.",
         "25 mg by mouth twice daily.",
         "No adjustment.",
         "Masks low-glucose warning signs in patients on insulin; additive bradycardia with Sedoflux.",
         "Heart rate and blood pressure before each dose.",
         "Hold if heart rate is below 55 beats per minute or systolic blood pressure is below 95 mmHg. Do not "
         "stop abruptly."),
        ("HV-RX-6080", "Norvessa", "catecholamine vasopressor",
         "ICU and emergency department; may be started on a ward by the rapid response team pending transfer.",
         "First-line vasopressor for septic shock.",
         "Start at 0.05 mcg/kg/min; titrate by 0.02 mcg/kg/min every 5 minutes to a mean arterial pressure of "
         "65 mmHg or higher.",
         "No adjustment.",
         "None clinically relevant.",
         "Arterial line recommended; check the infusion site every hour if peripheral.",
         "Central line preferred. May infuse peripherally for up to 6 hours through a 20-gauge or larger "
         "catheter placed above the antecubital fossa. If extravasation occurs, stop the infusion, leave the "
         "catheter in place and call pharmacy for the antidote."),
        ("HV-RX-6085", "Vasoplex", "vasopressin analog", "ICU and emergency department.",
         "Second vasopressor in septic shock.",
         "Fixed dose of 0.03 units/min, not titrated. Added when Norvessa reaches 0.25 mcg/kg/min.",
         "No adjustment.",
         "None clinically relevant.",
         "Watch for digital ischemia.",
         "Wean Vasoplex last, after Norvessa is below 0.1 mcg/kg/min."),
    ],
    "Electrolytes and miscellaneous": [
        ("HV-RX-7010", "Kaliveen", "intravenous potassium chloride", "High-alert medication.",
         "Hypokalemia.",
         "Peripheral line: maximum 10 mEq per hour. Central line: maximum 20 mEq per hour with continuous "
         "cardiac monitoring. Never give as an IV push.",
         "CrCl below 30 mL/min: replace in 10 mEq steps and recheck.",
         "Hyperkalemia with Spirolan.",
         "Recheck potassium (LC-0412) 2 hours after every 40 mEq given.",
         "Oral replacement is preferred when potassium is 3.0 mmol/L or higher and the patient can swallow."),
        ("HV-RX-7020", "Magnora", "magnesium sulfate", "Tier 1, unrestricted.",
         "Hypomagnesemia.",
         "Magnesium 1.2 to 1.6 mg/dL: 2 g IV over 1 hour. Below 1.2 mg/dL: 4 g IV over 2 hours.",
         "CrCl below 30 mL/min: halve the dose.",
         "None clinically relevant.",
         "Deep tendon reflexes with high doses.",
         "Replace magnesium before potassium in refractory hypokalemia."),
        ("HV-RX-7035", "Fosferra", "intravenous iron", "Tier 2.",
         "Iron deficiency anemia when oral iron fails or is not tolerated.",
         "510 mg IV over 15 minutes; second dose 3 to 8 days later.",
         "No adjustment.",
         "Stop oral iron during IV therapy.",
         "Observe the patient for 30 minutes after the infusion for hypersensitivity reactions.",
         "Not given during an active bloodstream infection."),
        ("HV-RX-7080", "Allopra", "xanthine oxidase inhibitor", "Tier 1, unrestricted.",
         "Gout prophylaxis; prevention of tumor lysis syndrome.",
         "Start at 100 mg by mouth once daily.",
         "CrCl below 30 mL/min: 50 mg once daily.",
         "None clinically relevant.",
         "Uric acid every 2 to 4 weeks during titration.",
         "Do not start during an acute gout flare."),
    ],
}


# Distractor drugs: generated from class templates so the formulary is big enough that top-5 retrieval is a real
# test. Golden questions only target the hand-written entries above. Seeded, so the output never changes.
# (section, class, name stem, indications, dosing format, dose choices a/b/h, renal, interactions, monitoring, notes)
DISTRACTOR_CLASSES = [
    ("Cardiovascular and vasopressors", "ACE inhibitor", "pril", "Hypertension; heart failure; proteinuric kidney disease.",
     "{a} mg by mouth once daily; may increase to {b} mg once daily.", ([2.5, 5, 10], [20, 40], [24]),
     ["CrCl below 30 mL/min: start at half the usual dose.", "No adjustment; monitor potassium."],
     ["Hyperkalemia with Spirolan and Kaliveen.", "Never combine with a neprilysin inhibitor within 36 hours."],
     ["Potassium and creatinine 1 to 2 weeks after starting.", "Blood pressure daily during titration."],
     ["Dry cough may require a switch to an angiotensin receptor blocker.", "Hold the morning of surgery."]),
    ("Cardiovascular and vasopressors", "angiotensin receptor blocker", "sartan", "Hypertension; heart failure.",
     "{a} mg by mouth once daily; maximum {b} mg daily.", ([25, 40, 50], [100, 160, 320], [24]),
     ["No adjustment.", "Volume-depleted patients: start at the lowest dose."],
     ["Do not combine with an ACE inhibitor.", "Hyperkalemia with potassium supplements."],
     ["Potassium within 2 weeks of a dose change.", "Blood pressure and creatinine."],
     ["Preferred alternative when an ACE inhibitor causes cough.", "Hold for systolic pressure below 100 mmHg."]),
    ("Cardiovascular and vasopressors", "statin", "vastin", "Hyperlipidemia; secondary prevention of cardiovascular disease.",
     "{a} mg by mouth once daily in the evening.", ([10, 20, 40], [80], [24]),
     ["No adjustment.", "CrCl below 30 mL/min: maximum 10 mg daily."],
     ["Myopathy risk rises with strong CYP3A4 inhibitors.", "Levels raised by Nephrazole."],
     ["Lipid panel 6 to 12 weeks after starting.", "Creatine kinase only if muscle symptoms occur."],
     ["Report unexplained muscle pain.", "Continue through admission unless liver enzymes exceed 3 times normal."]),
    ("Cardiovascular and vasopressors", "cardioselective beta blocker", "olol", "Hypertension; angina; heart rate control.",
     "{a} mg by mouth twice daily; maximum {b} mg twice daily.", ([12.5, 25, 50], [100, 200], [12]),
     ["No adjustment.", "CrCl below 35 mL/min: maximum 50 mg daily."],
     ["Additive bradycardia with non-dihydropyridine calcium channel blockers.", "May mask hypoglycemia."],
     ["Heart rate and blood pressure before each dose.", "ECG if heart rate is below 50."],
     ["Hold if heart rate is below 50 beats per minute.", "Taper over 1 to 2 weeks when stopping."]),
    ("Cardiovascular and vasopressors", "positive inotrope", "butrex", "Low cardiac output; cardiogenic shock.",
     "Start at {a} mcg/kg/min and titrate to cardiac output; maximum {b} mcg/kg/min.", ([2, 2.5, 5], [15, 20], [1]),
     ["No adjustment.", "No adjustment; titrate to effect."],
     ["Arrhythmia risk with other catecholamines.", "Effect reduced by beta blockers."],
     ["Continuous ECG and blood pressure monitoring.", "Urine output every hour."],
     ["ICU only.", "Not a substitute for a vasopressor in septic shock."]),
    ("Anticoagulants and reversal agents", "P2Y12 antiplatelet agent", "grel", "Acute coronary syndrome; after coronary stenting.",
     "Loading dose {b} mg by mouth, then {a} mg once daily.", ([75, 90, 10], [300, 600, 180], [24]),
     ["No adjustment.", "No adjustment; avoid in dialysis patients."],
     ["Bleeding risk with anticoagulants and NSAIDs.", "Proton pump inhibitors may reduce effect."],
     ["CBC weekly while on dual antiplatelet therapy.", "Watch for signs of bleeding."],
     ["Do not stop within 6 months of a stent without cardiology approval.", "Hold 5 to 7 days before elective surgery."]),
    ("Anticoagulants and reversal agents", "low molecular weight heparin", "parin", "VTE prophylaxis in selected surgical patients.",
     "{a} units subcutaneously once daily.", ([2500, 5000], [10000], [24]),
     ["CrCl below 30 mL/min: use unfractionated heparin instead.", "CrCl below 30 mL/min: avoid."],
     ["Bleeding risk with antiplatelet agents.", "Additive bleeding with NSAIDs."],
     ["Platelets at baseline and on day 5.", "Platelets twice weekly."],
     ["Non-formulary alternative to Heparivex for orthopedic surgery only.", "Second-line to Heparivex."]),
    ("Endocrine and glucose management", "SGLT2 inhibitor", "gliflozin", "Type 2 diabetes; heart failure; chronic kidney disease.",
     "{a} mg by mouth once daily.", ([10, 25, 100], [300], [24]),
     ["eGFR below 20: do not start.", "eGFR below 30: not for glucose lowering."],
     ["Additive hypotension with diuretics.", "Raises ketoacidosis risk when insulin is reduced."],
     ["Ketones if the patient is unwell.", "Volume status and creatinine."],
     ["Hold 3 days before surgery because of euglycemic ketoacidosis risk.", "Hold while NPO."]),
    ("Endocrine and glucose management", "DPP-4 inhibitor", "gliptin", "Type 2 diabetes.",
     "{a} mg by mouth once daily.", ([5, 25, 50, 100], [100], [24]),
     ["CrCl below 45 mL/min: halve the dose.", "No adjustment."],
     ["Hypoglycemia when combined with sulfonylureas.", "None clinically relevant."],
     ["HbA1c every 3 months.", "Glucose before meals."],
     ["Usually stopped on admission and replaced with insulin.", "Report severe joint pain."]),
    ("Endocrine and glucose management", "sulfonylurea", "glizide", "Type 2 diabetes.",
     "{a} mg by mouth once daily with breakfast; maximum {b} mg daily.", ([1, 2.5, 5], [10, 20], [24]),
     ["CrCl below 30 mL/min: avoid.", "Start at the lowest dose in older adults."],
     ["Prolonged hypoglycemia with fluoroquinolones.", "Hypoglycemia masked by beta blockers."],
     ["Glucose before meals and at bedtime.", "HbA1c every 3 months."],
     ["Hold on admission; high hypoglycemia risk in hospital.", "Hold when not eating."]),
    ("Psychiatry and neurology", "selective serotonin reuptake inhibitor", "oxetine", "Depression; anxiety disorders.",
     "{a} mg by mouth once daily; maximum {b} mg daily.", ([10, 20, 50], [40, 60, 200], [24]),
     ["No adjustment.", "Hepatic impairment: halve the dose."],
     ["Serotonin syndrome with Lumidex, tramadol-like opioids and linezolid-like agents.", "Bleeding risk with antiplatelets."],
     ["Mood and suicidal thoughts in the first weeks.", "Sodium in older adults."],
     ["Hyponatremia is more common in older adults.", "Taper to avoid discontinuation symptoms."]),
    ("Psychiatry and neurology", "antiepileptic", "tiracetam", "Focal and generalized seizures; status epilepticus after a benzodiazepine.",
     "{a} mg IV or by mouth twice daily; loading dose {b} mg/kg IV for status epilepticus.", ([500, 750, 1000], [40, 60], [12]),
     ["CrCl 30 to 50 mL/min: maximum 750 mg twice daily.", "CrCl below 30 mL/min: maximum 500 mg twice daily."],
     ["None clinically relevant.", "Additive sedation with opioids."],
     ["Mood changes and irritability.", "Seizure frequency."],
     ["IV and oral doses are equivalent.", "Do not stop abruptly."]),
    ("Psychiatry and neurology", "atypical antipsychotic", "apine", "Schizophrenia; bipolar disorder; severe agitation.",
     "{a} mg by mouth at bedtime; maximum {b} mg daily.", ([2.5, 5, 25], [20, 30, 800], [24]),
     ["No adjustment.", "Start low in hepatic impairment."],
     ["QT prolongation with other QT-prolonging drugs.", "Sedation with benzodiazepines."],
     ["Glucose and lipids at baseline and 12 weeks.", "Weight weekly."],
     ["Orthostatic hypotension; rise slowly.", "Avoid in dementia-related psychosis."]),
    ("Allergy and respiratory", "second-generation antihistamine", "tadine", "Allergic rhinitis; urticaria.",
     "{a} mg by mouth once daily.", ([5, 10, 180], [20], [24]),
     ["CrCl below 30 mL/min: every other day.", "No adjustment."],
     ["Absorption reduced by fruit juice.", "None clinically relevant."],
     ["Sedation in older adults.", "Symptom relief."],
     ["Preferred over sedating antihistamines in older adults.", "Stop 5 days before allergy skin testing."]),
    ("Allergy and respiratory", "inhaled beta-2 agonist bronchodilator", "terol", "Asthma; chronic obstructive pulmonary disease.",
     "{a} puffs inhaled every {h} hours as needed.", ([1, 2], [8], [4, 6]),
     ["No adjustment.", "No adjustment."],
     ["Hypokalemia with high doses and diuretics.", "Effect blocked by non-selective beta blockers."],
     ["Heart rate and peak flow.", "Respiratory rate and oxygen saturation."],
     ["Use a spacer for metered-dose inhalers.", "Rinse mouth after use."]),
    ("Allergy and respiratory", "inhaled corticosteroid", "sonide", "Maintenance therapy for asthma and COPD.",
     "{a} mcg inhaled twice daily; maximum {b} mcg twice daily.", ([100, 200, 250], [400, 500], [12]),
     ["No adjustment.", "No adjustment."],
     ["Levels raised by strong CYP3A4 inhibitors.", "None clinically relevant."],
     ["Oral thrush.", "Symptom control."],
     ["Rinse mouth after each use to prevent thrush.", "Not for acute bronchospasm."]),
    ("Anti-infectives", "first-generation cephalosporin", "lexin", "Skin and soft tissue infection; surgical prophylaxis.",
     "{a} g IV every 8 hours; surgical prophylaxis {b} g IV within 60 minutes before incision.", ([1, 2], [2, 3], [8]),
     ["CrCl 10 to 30 mL/min: every 12 hours.", "CrCl below 35 mL/min: extend interval to every 12 hours."],
     ["None clinically relevant.", "Probenecid raises levels."],
     ["CBC weekly on long courses.", "Rash."],
     ["Patients over 120 kg receive 3 g for surgical prophylaxis.", "Safe in most penicillin-allergic patients without anaphylaxis history."]),
    ("Anti-infectives", "antiviral", "clovir", "Herpes simplex and varicella zoster infections.",
     "{a} mg/kg IV every 8 hours.", ([5, 10], [15], [8]),
     ["CrCl 25 to 50 mL/min: every 12 hours.", "CrCl below 25 mL/min: every 24 hours."],
     ["Nephrotoxicity with other nephrotoxic drugs.", "None clinically relevant."],
     ["Creatinine every 48 hours; keep patient well hydrated.", "Neurological status."],
     ["Dose on ideal body weight in obesity.", "Infuse over 1 hour."]),
    ("Anti-infectives", "aminoglycoside", "micin", "Gram-negative sepsis in combination; synergy for endocarditis.",
     "{a} mg/kg IV once daily.", ([5, 7], [7], [24]),
     ["Pharmacy adjusts the interval using levels.", "Avoid in CrCl below 20 mL/min."],
     ["Nephrotoxicity and ototoxicity with loop diuretics and Vancorin.", "Additive nephrotoxicity."],
     ["Level 6 to 14 hours after the first dose.", "Creatinine daily; hearing changes."],
     ["Limit courses to 72 hours when possible.", "Pharmacy-to-dose."]),
    ("Anti-infectives", "tetracycline-class antibiotic", "cycline", "Community-acquired pneumonia; skin infections; tick-borne illness.",
     "{a} mg IV or by mouth every 12 hours.", ([100], [200], [12]),
     ["No adjustment.", "No adjustment."],
     ["Absorption reduced by calcium, iron and antacids.", "Photosensitivity."],
     ["Liver enzymes on long courses.", "Rash and sun exposure."],
     ["Take with a full glass of water and stay upright for 30 minutes.", "Avoid in pregnancy."]),
    ("Antiemetics and gastrointestinal", "H2 receptor antagonist", "tidine", "Dyspepsia; alternative stress ulcer prophylaxis.",
     "{a} mg IV every 12 hours.", ([20, 40], [40], [12]),
     ["CrCl below 50 mL/min: halve the dose.", "CrCl below 50 mL/min: once daily."],
     ["Confusion in older adults.", "None clinically relevant."],
     ["Mental status in older adults.", "Platelets on long courses."],
     ["Second-line to Pantorex.", "Convert to oral when eating."]),
    ("Antiemetics and gastrointestinal", "5-HT3 receptor antagonist antiemetic", "setron", "Chemotherapy-induced nausea.",
     "{a} mg IV every {h} hours as needed; do not exceed {b} mg in 24 hours.", ([1, 8], [24, 32], [12, 24]),
     ["No adjustment.", "No adjustment."],
     ["QT prolongation with other QT-prolonging drugs.", "Serotonin syndrome with serotonergic drugs."],
     ["ECG if other QT-prolonging drugs are given.", "Bowel function."],
     ["Restricted to oncology; second-line to Lumidex.", "Oncology use only."]),
    ("Antiemetics and gastrointestinal", "osmotic laxative", "lax", "Constipation; bowel preparation.",
     "{a} g by mouth once daily dissolved in water.", ([17], [34], [24]),
     ["No adjustment.", "No adjustment."],
     ["None clinically relevant.", "None clinically relevant."],
     ["Stool frequency.", "Electrolytes with prolonged use."],
     ["Takes 1 to 3 days to work.", "Hold for diarrhea."]),
    ("Analgesia and sedation", "oral opioid agonist", "codone", "Moderate to severe pain.",
     "{a} mg by mouth every {h} hours as needed.", ([5, 10], [10], [4, 6]),
     ["Use with caution in renal impairment; extend the dosing interval.", "CrCl below 30 mL/min: start at the lowest dose."],
     ["Respiratory depression with benzodiazepines.", "Additive sedation with gabapentinoids."],
     ["Sedation score and respiratory rate after the first dose.", "Bowel function."],
     ["Start a bowel regimen.", "Do not crush extended-release forms."]),
    ("Analgesia and sedation", "local anesthetic", "caine", "Local infiltration; nerve blocks.",
     "Maximum {a} mg/kg per procedure without epinephrine; {b} mg/kg with epinephrine.", ([2, 3, 4.5], [7], [24]),
     ["No adjustment.", "No adjustment."],
     ["Toxicity is additive with other local anesthetics.", "None clinically relevant."],
     ["Watch for perioral numbness, tinnitus and seizures.", "Cardiac rhythm with large doses."],
     ["Lipid emulsion is stocked for local anesthetic toxicity.", "Aspirate before injecting."]),
    ("Analgesia and sedation", "skeletal muscle relaxant", "benzaprine", "Acute muscle spasm.",
     "{a} mg by mouth three times daily; maximum 3 weeks.", ([5, 10], [10], [8]),
     ["No adjustment.", "Avoid in hepatic impairment."],
     ["Additive sedation with opioids.", "Anticholinergic effects with antihistamines."],
     ["Sedation and falls.", "Urinary retention."],
     ["On the HVMC list of drugs to avoid in older adults.", "High fall risk."]),
    ("Electrolytes and miscellaneous", "bisphosphonate", "dronate", "Osteoporosis; hypercalcemia of malignancy.",
     "{a} mg IV once over 15 to 30 minutes.", ([4, 5], [5], [24]),
     ["CrCl below 35 mL/min: avoid.", "CrCl below 35 mL/min: contraindicated."],
     ["Hypocalcemia with loop diuretics.", "None clinically relevant."],
     ["Calcium and creatinine before each dose.", "Calcium 1 week after the dose."],
     ["Correct vitamin deficiencies first.", "Flu-like symptoms are common after the first infusion."]),
]

NAME_PREFIXES = ["Abra", "Belo", "Cira", "Dalo", "Elva", "Feno", "Gara", "Hexa", "Ilo", "Jova", "Kora", "Levi",
                 "Mira", "Nova", "Opra", "Pela", "Quina", "Rova", "Sila", "Tova", "Ulva", "Vela", "Wexa", "Xala",
                 "Yora", "Zana", "Brio", "Cala", "Dura", "Esta", "Fira", "Galo"]
DRUGS_PER_CLASS = 8


def distractor_drugs() -> dict[str, list[tuple]]:
    rng = random.Random(2026)
    taken_codes = {d[0] for drugs in FORMULARY.values() for d in drugs}
    taken_names = {d[1] for drugs in FORMULARY.values() for d in drugs}
    out: dict[str, list[tuple]] = {}
    for section, cls, stem, ind, dose_fmt, (a_opts, b_opts, h_opts), renal, inter, mon, notes in DISTRACTOR_CLASSES:
        for _ in range(DRUGS_PER_CLASS):
            name = rng.choice(NAME_PREFIXES) + stem
            while name in taken_names:
                name = rng.choice(NAME_PREFIXES) + stem
            code = f"HV-RX-{rng.randint(1000, 8999)}"
            while code in taken_codes:
                code = f"HV-RX-{rng.randint(1000, 8999)}"
            taken_names.add(name)
            taken_codes.add(code)
            dose = dose_fmt.format(a=rng.choice(a_opts), b=rng.choice(b_opts), h=rng.choice(h_opts))
            status = rng.choice(["Tier 1, unrestricted.", "Tier 2.", "Tier 2, pharmacist verification required.",
                                 "Non-formulary; available by request."])
            out.setdefault(section, []).append((code, name, cls, status, ind, dose, rng.choice(renal),
                                                rng.choice(inter), rng.choice(mon), rng.choice(notes)))
    return out


def render_formulary() -> str:
    lines = ["Halverd Valley Medical Center (HVMC)", "Adult Inpatient Formulary, 2026 edition", DISCLAIMER, "",
             "How to use this formulary",
             "Each entry lists the HVMC formulary code (HV-RX-####), drug class, formulary status, indications, "
             "usual adult dosing, renal dose adjustment, key interactions, monitoring and pharmacist notes. "
             "Renal adjustments use creatinine clearance (CrCl) calculated with the Cockcroft-Gault equation and "
             "actual body weight. This formulary covers adults only; consult the separate pediatric handbook "
             "for patients under 18. Non-formulary requests are submitted to pharmacy and reviewed within "
             "24 hours.", ""]
    extra = distractor_drugs()
    for section in dict.fromkeys([*FORMULARY, *extra]):  # hand-written sections first, then new ones
        # Sort by code inside each section so distractors interleave with the hand-written entries
        drugs = sorted(FORMULARY.get(section, []) + extra.get(section, []), key=lambda d: d[0])
        lines += [f"Section: {section}", ""]
        for code, name, cls, status, ind, dose, renal, inter, mon, notes in drugs:
            lines += [f"{code} {name} ({cls})",
                      f"Formulary status: {status}",
                      f"Indications: {ind}",
                      f"Adult dosing: {dose}",
                      f"Renal adjustment: {renal}",
                      f"Interactions: {inter}",
                      f"Monitoring: {mon}",
                      f"Pharmacist notes: {notes}", ""]
    return "\n".join(lines)


# ---------- 2. Laboratory test catalog ----------

# (code, test, specimen, reference range, critical values or None, turnaround, notes)
LABS = {
    "Chemistry": [
        ("LC-0410", "Sodium, serum", "lithium heparin (mint green top) tube, minimum 3 mL", "135 to 145 mmol/L",
         "below 120 or above 160 mmol/L", "STAT 45 minutes; routine 4 hours",
         "Correct chronic hyponatremia by no more than 8 mmol/L in 24 hours."),
        ("LC-0412", "Potassium, serum", "lithium heparin (mint green top) tube, minimum 3 mL",
         "3.5 to 5.1 mmol/L", "below 2.8 or above 6.2 mmol/L", "STAT 45 minutes; routine 4 hours",
         "Hemolysis falsely raises potassium. A hemolyzed sample is reported with a comment and should be "
         "redrawn before treating."),
        ("LC-0421", "Chloride, serum", "lithium heparin (mint green top) tube, minimum 3 mL", "98 to 107 mmol/L",
         None, "routine 4 hours", "Reported as part of the basic metabolic panel."),
        ("LC-0430", "Bicarbonate (total CO2), serum", "lithium heparin (mint green top) tube, minimum 3 mL",
         "22 to 29 mmol/L", "below 10 or above 40 mmol/L", "STAT 45 minutes; routine 4 hours",
         "Underfilled tubes give falsely low results."),
        ("LC-0440", "Blood urea nitrogen (BUN)", "lithium heparin (mint green top) tube, minimum 3 mL",
         "7 to 20 mg/dL", None, "routine 4 hours", "Rises with gastrointestinal bleeding and dehydration."),
        ("LC-0445", "Creatinine, serum", "lithium heparin (mint green top) tube, minimum 3 mL",
         "0.6 to 1.2 mg/dL", None, "STAT 45 minutes; routine 4 hours",
         "eGFR is reported automatically using the CKD-EPI 2021 equation. Formulary renal dosing uses CrCl, "
         "not eGFR, unless stated."),
        ("LC-0450", "Glucose, plasma", "sodium fluoride (gray top) tube, minimum 2 mL",
         "70 to 99 mg/dL fasting", "below 50 or above 450 mg/dL", "STAT 30 minutes; routine 4 hours",
         "Point-of-care glucose readings below 40 or above 500 mg/dL must be confirmed with this test."),
        ("LC-0460", "Calcium, total", "lithium heparin (mint green top) tube, minimum 3 mL", "8.6 to 10.2 mg/dL",
         "below 6.5 or above 13.0 mg/dL", "routine 4 hours",
         "Interpret with albumin (LC-0530), or order ionized calcium."),
        ("LC-0465", "Magnesium, serum", "lithium heparin (mint green top) tube, minimum 3 mL", "1.7 to 2.2 mg/dL",
         "below 1.0 or above 4.9 mg/dL", "routine 4 hours", "Replace with Magnora per formulary."),
        ("LC-0470", "Phosphate, serum", "lithium heparin (mint green top) tube, minimum 3 mL",
         "2.5 to 4.5 mg/dL", "below 1.0 mg/dL", "routine 4 hours",
         "Watch for refeeding syndrome in malnourished patients."),
        ("LC-0510", "Alanine aminotransferase (ALT)", "lithium heparin (mint green top) tube, minimum 3 mL",
         "7 to 56 U/L", None, "routine 6 hours", "Part of the hepatic function panel."),
        ("LC-0512", "Aspartate aminotransferase (AST)", "lithium heparin (mint green top) tube, minimum 3 mL",
         "10 to 40 U/L", None, "routine 6 hours", "Also released by muscle injury and hemolysis."),
        ("LC-0520", "Bilirubin, total", "lithium heparin (mint green top) tube, minimum 3 mL; protect from light",
         "0.1 to 1.2 mg/dL", None, "routine 6 hours", "Light exposure falsely lowers results."),
        ("LC-0530", "Albumin", "lithium heparin (mint green top) tube, minimum 3 mL", "3.5 to 5.0 g/dL", None,
         "routine 6 hours", "Low albumin lowers total calcium without changing ionized calcium."),
        ("LC-0540", "Lipase", "lithium heparin (mint green top) tube, minimum 3 mL", "13 to 60 U/L", None,
         "STAT 60 minutes; routine 6 hours",
         "A value above 3 times the upper limit supports acute pancreatitis."),
    ],
    "Hematology and coagulation": [
        ("LC-0610", "Hemoglobin", "EDTA (lavender top) tube, minimum 2 mL",
         "13.5 to 17.5 g/dL in males; 12.0 to 15.5 g/dL in females", "below 7.0 g/dL",
         "STAT 30 minutes; routine 2 hours", "Transfusion threshold for most stable inpatients is 7.0 g/dL."),
        ("LC-0612", "White blood cell count (WBC)", "EDTA (lavender top) tube, minimum 2 mL",
         "4.5 to 11.0 x10^9/L", "below 1.0 or above 50.0 x10^9/L", "STAT 30 minutes; routine 2 hours",
         "Differential added automatically when the WBC is flagged."),
        ("LC-0615", "Platelet count", "EDTA (lavender top) tube, minimum 2 mL", "150 to 400 x10^9/L",
         "below 20 x10^9/L", "STAT 30 minutes; routine 2 hours",
         "Clumping can cause a falsely low count; the lab redraws in a citrate tube."),
        ("LC-0620", "Prothrombin time with INR", "sodium citrate (light blue top) tube, filled to the line",
         "INR 0.9 to 1.1", "INR above 5.0", "STAT 45 minutes; routine 4 hours",
         "The tube must be filled to the fill line to keep the 9:1 blood-to-citrate ratio; underfilled tubes "
         "are rejected."),
        ("LC-0625", "Activated partial thromboplastin time (aPTT)",
         "sodium citrate (light blue top) tube, filled to the line", "25 to 35 seconds", "above 100 seconds",
         "STAT 45 minutes; routine 4 hours", "Do not draw from a line used for heparin infusion."),
        ("LC-0630", "Fibrinogen", "sodium citrate (light blue top) tube, filled to the line", "200 to 400 mg/dL",
         "below 100 mg/dL", "STAT 60 minutes", "Used in massive transfusion and suspected DIC."),
        ("LC-0640", "D-dimer", "sodium citrate (light blue top) tube, filled to the line",
         "below 0.50 mcg/mL FEU", None, "STAT 60 minutes",
         "For patients over 50, use the age-adjusted cutoff: age multiplied by 0.01 mcg/mL FEU. Only useful to "
         "exclude clots in patients with low pretest probability."),
    ],
    "Sepsis and infection": [
        ("LC-0710", "Lactate, venous", "sodium fluoride (gray top) tube, minimum 2 mL", "0.5 to 2.0 mmol/L",
         "4.0 mmol/L or above", "STAT 20 minutes",
         "Transport on ice and analyze within 15 minutes of collection. Avoid prolonged tourniquet use. If the "
         "first lactate is above 2.0 mmol/L, repeat within 2 to 4 hours (CP-07)."),
        ("LC-0720", "Procalcitonin", "lithium heparin (mint green top) tube, minimum 3 mL", "below 0.10 ng/mL",
         None, "routine 6 hours",
         "A value above 0.5 ng/mL suggests bacterial infection. Used to support stopping antibiotics, not to "
         "delay starting them."),
        ("LC-0730", "Blood culture set", "one aerobic and one anaerobic bottle, 8 to 10 mL of blood per bottle",
         "no growth", "any positive culture", "preliminary report at 24 hours; final report at 5 days",
         "Collect two sets from two separate venipuncture sites before antibiotics are given. Do not draw "
         "through an existing IV catheter unless a line infection is suspected."),
        ("LC-0740", "C-reactive protein (CRP)", "lithium heparin (mint green top) tube, minimum 3 mL",
         "below 5 mg/L", None, "routine 4 hours", "Non-specific marker of inflammation."),
    ],
    "Cardiac and endocrine": [
        ("LC-0815", "High-sensitivity troponin (HVMC hs-TnX assay)",
         "lithium heparin (mint green top) tube, minimum 3 mL", "99th percentile upper limit 14 ng/L",
         "52 ng/L or above", "STAT 45 minutes",
         "0/1-hour algorithm: rule-out if the presentation value is below 5 ng/L and symptoms began more than 3 "
         "hours earlier; rule-in if the value is 52 ng/L or above, or rises by 5 ng/L or more at 1 hour. All "
         "other patients enter the observation zone and are retested at 3 hours."),
        ("LC-0851", "B-type natriuretic peptide (BNP)", "EDTA (lavender top) tube, minimum 2 mL",
         "below 100 pg/mL", None, "STAT 60 minutes",
         "Values are lower in obesity; above 400 pg/mL makes heart failure likely."),
        ("LC-0860", "Hemoglobin A1c", "EDTA (lavender top) tube, minimum 2 mL", "4.0 to 5.6 percent", None,
         "routine 24 hours", "Ordered once per admission for patients with diabetes if none in 90 days."),
        ("LC-0870", "Thyroid-stimulating hormone (TSH)", "lithium heparin (mint green top) tube, minimum 3 mL",
         "0.4 to 4.0 mIU/L", None, "routine 24 hours", "Not recommended during acute critical illness."),
    ],
    "Drug levels and special handling": [
        ("LC-0910", "Vancorin levels for AUC dosing", "lithium heparin (mint green top) tube, minimum 3 mL",
         "target AUC 400 to 600", None, "STAT 2 hours",
         "Draw two levels around the same dose: the first 1 hour after the end of the infusion and the second "
         "within 30 minutes before the next dose. Label each tube with the exact draw time."),
        ("LC-0920", "Anti-Xa level (Heparivex)", "sodium citrate (light blue top) tube, filled to the line",
         "treatment range 0.5 to 1.0 IU/mL; prophylaxis range 0.2 to 0.5 IU/mL", None, "routine 4 hours",
         "Draw 4 hours after the third or fourth dose."),
        ("LC-0930", "Ammonia, plasma", "EDTA (lavender top) tube, minimum 2 mL", "15 to 45 mcg/dL", None,
         "STAT 30 minutes", "Transport on ice and analyze within 30 minutes; do not use a tourniquet."),
        ("LC-0940", "Arterial blood gas", "pre-heparinized arterial syringe, minimum 1 mL",
         "pH 7.35 to 7.45; PaCO2 35 to 45 mmHg; PaO2 80 to 100 mmHg", "pH below 7.20 or above 7.60",
         "STAT 15 minutes",
         "Expel all air bubbles and cap the syringe. Place on ice if analysis is delayed more than 30 minutes."),
    ],
}


GREEN = "lithium heparin (mint green top) tube, minimum 3 mL"
LAVENDER = "EDTA (lavender top) tube, minimum 2 mL"
BLUE = "sodium citrate (light blue top) tube, filled to the line"
GOLD = "serum separator (gold top) tube, minimum 4 mL"

# Distractor tests, including near-miss codes (LC-0414 next to LC-0412) and tests that sound alike
LABS_EXTRA = {
    "Chemistry": [
        ("LC-0414", "Potassium, whole blood (blood gas analyzer)", "pre-heparinized arterial or venous syringe, minimum 1 mL",
         "3.4 to 4.9 mmol/L", "below 2.5 or above 6.5 mmol/L", "STAT 15 minutes",
         "Faster than LC-0412 but not interchangeable with it; use the same method when trending."),
        ("LC-0416", "Sodium, whole blood (blood gas analyzer)", "pre-heparinized arterial or venous syringe, minimum 1 mL",
         "136 to 146 mmol/L", "below 118 or above 162 mmol/L", "STAT 15 minutes",
         "Results can differ from LC-0410 by up to 4 mmol/L."),
        ("LC-0432", "Anion gap (calculated)", "calculated from LC-0410, LC-0421 and LC-0430", "4 to 12 mmol/L", None,
         "reported with the basic metabolic panel", "A gap above 12 mmol/L suggests ketoacidosis, lactic acidosis or toxins."),
        ("LC-0447", "Cystatin C", GREEN, "0.6 to 1.0 mg/L", None, "routine 24 hours",
         "Used with creatinine when muscle mass is unusually high or low."),
        ("LC-0452", "Beta-hydroxybutyrate (ketones), serum", GREEN, "below 0.4 mmol/L", "3.0 mmol/L or above",
         "STAT 60 minutes", "Preferred ketone test for monitoring diabetic ketoacidosis (CP-27)."),
        ("LC-0455", "Osmolality, serum", GOLD, "275 to 295 mOsm/kg", "below 250 or above 320 mOsm/kg", "STAT 60 minutes",
         "An osmolal gap above 10 suggests toxic alcohol ingestion."),
        ("LC-0462", "Calcium, ionized", "pre-heparinized syringe on ice, minimum 1 mL", "1.15 to 1.33 mmol/L",
         "below 0.80 or above 1.60 mmol/L", "STAT 30 minutes", "Unaffected by albumin; preferred during massive transfusion."),
        ("LC-0475", "Uric acid", GREEN, "3.5 to 7.2 mg/dL", None, "routine 6 hours", "Monitored during Allopra titration."),
        ("LC-0515", "Alkaline phosphatase", GREEN, "44 to 147 U/L", None, "routine 6 hours", "Also raised in bone disease."),
        ("LC-0517", "Gamma-glutamyl transferase (GGT)", GREEN, "9 to 48 U/L", None, "routine 6 hours",
         "Helps confirm a liver source of raised alkaline phosphatase."),
        ("LC-0522", "Bilirubin, direct", GREEN + "; protect from light", "0.0 to 0.3 mg/dL", None, "routine 6 hours",
         "Ordered when total bilirubin is raised."),
        ("LC-0545", "Amylase", GREEN, "30 to 110 U/L", None, "routine 6 hours", "Lipase (LC-0540) is preferred for pancreatitis."),
        ("LC-0550", "Creatine kinase (CK)", GREEN, "39 to 308 U/L", "above 5,000 U/L", "routine 4 hours",
         "Very high values suggest rhabdomyolysis; check urine and creatinine."),
        ("LC-0560", "Triglycerides", GREEN, "below 150 mg/dL", None, "routine 24 hours", "Monitor in patients on lipid infusions."),
        ("LC-0565", "Lipid panel", GREEN, "LDL below 100 mg/dL; HDL above 40 mg/dL", None, "routine 24 hours",
         "Fasting not required."),
    ],
    "Hematology and coagulation": [
        ("LC-0611", "Hematocrit", LAVENDER, "41 to 53 percent in males; 36 to 46 percent in females", "below 20 percent",
         "STAT 30 minutes; routine 2 hours", "Reported with the complete blood count."),
        ("LC-0613", "Absolute neutrophil count", LAVENDER, "1.5 to 8.0 x10^9/L", "below 0.5 x10^9/L",
         "routine 2 hours", "Below 0.5 x10^9/L requires neutropenic precautions."),
        ("LC-0616", "Reticulocyte count", LAVENDER, "0.5 to 2.5 percent", None, "routine 6 hours",
         "Distinguishes reduced red cell production from blood loss or destruction."),
        ("LC-0618", "Peripheral blood smear", LAVENDER, "normal morphology", "schistocytes or blasts", "routine 24 hours",
         "Reviewed by a pathologist when flagged."),
        ("LC-0622", "Thrombin time", BLUE, "14 to 19 seconds", None, "routine 4 hours",
         "Very sensitive to Corvetrol; a normal result excludes significant drug effect."),
        ("LC-0627", "Heparin-induced thrombocytopenia (HIT) antibody", GOLD, "negative", None, "routine 24 hours",
         "Order only when the 4Ts score is intermediate or high."),
        ("LC-0632", "Factor Xa inhibitor level (Corvatrel-calibrated)", BLUE, "reported in ng/mL; no reference range", None,
         "STAT 2 hours", "Useful before urgent surgery in patients taking Corvatrel."),
        ("LC-0650", "Type and screen", "EDTA (pink top) tube, minimum 6 mL, hand-labeled at the bedside", "not applicable",
         None, "STAT 60 minutes", "Valid for 72 hours. Required before any transfusion."),
        ("LC-0652", "Crossmatch", "EDTA (pink top) tube, minimum 6 mL", "compatible", None, "STAT 45 minutes after type and screen",
         "Electronic crossmatch is used when two blood groups are on file."),
        ("LC-0655", "Direct antiglobulin test (DAT)", LAVENDER, "negative", None, "routine 4 hours",
         "Part of the transfusion reaction investigation."),
    ],
    "Sepsis and infection": [
        ("LC-0712", "Lactate, arterial", "pre-heparinized arterial syringe, minimum 1 mL", "0.5 to 1.6 mmol/L",
         "4.0 mmol/L or above", "STAT 15 minutes", "Run on the blood gas analyzer; arterial values run slightly lower than venous."),
        ("LC-0732", "Urine culture", "sterile urine cup or catheter sample port, minimum 10 mL", "no growth or below 10,000 CFU/mL",
         None, "preliminary report at 24 hours; final report at 48 hours",
         "Never collect from the drainage bag. Refrigerate if not sent within 30 minutes."),
        ("LC-0734", "Sputum culture", "sterile sputum cup", "normal respiratory flora", None, "final report at 72 hours",
         "Rejected if more than 10 squamous cells per field."),
        ("LC-0736", "MRSA nasal screen (PCR)", "nasal swab, both nostrils", "not detected", None, "routine 4 hours",
         "A negative result supports stopping Vancorin in pneumonia."),
        ("LC-0738", "Clostridioides difficile toxin PCR", "liquid stool sample", "not detected", None, "routine 6 hours",
         "Only unformed stool is tested; do not test after laxatives within 48 hours."),
        ("LC-0742", "Erythrocyte sedimentation rate (ESR)", LAVENDER, "0 to 20 mm/h", None, "routine 4 hours",
         "Rises slowly and falls slowly; less useful than CRP in acute illness."),
        ("LC-0744", "Respiratory viral panel (PCR)", "nasopharyngeal swab in viral transport medium", "not detected", None,
         "routine 6 hours", "Includes influenza A and B, RSV and SARS-CoV-2."),
    ],
    "Cardiac and endocrine": [
        ("LC-0818", "Troponin, point-of-care (bedside)", "lithium heparin whole blood, minimum 1 mL",
         "upper reference limit 30 ng/L", None, "15 minutes",
         "Less sensitive than LC-0815; never used alone to rule out myocardial infarction."),
        ("LC-0853", "NT-proBNP", GREEN, "below 125 pg/mL under age 75; below 450 pg/mL age 75 and over", None,
         "STAT 60 minutes", "Not interchangeable with BNP (LC-0851)."),
        ("LC-0872", "Free T4", GREEN, "0.8 to 1.8 ng/dL", None, "routine 24 hours", "Ordered when TSH is abnormal."),
        ("LC-0875", "Cortisol, random", GOLD, "morning value 6 to 23 mcg/dL", None, "routine 6 hours",
         "Draw before the first dose of Prednavex when adrenal insufficiency is suspected."),
    ],
    "Drug levels and special handling": [
        ("LC-0912", "Aminoglycoside level", GREEN, "interpreted by pharmacy using a dosing nomogram", None, "STAT 2 hours",
         "Draw 6 to 14 hours after the first once-daily dose."),
        ("LC-0915", "Valproic acid level", GOLD, "50 to 100 mcg/mL", "above 150 mcg/mL", "routine 6 hours",
         "Falls sharply with Meronex; see formulary."),
        ("LC-0918", "Digoxin-class glycoside level", GOLD, "0.5 to 0.9 ng/mL", "above 2.0 ng/mL", "routine 6 hours",
         "Draw at least 6 hours after the last dose."),
        ("LC-0922", "Anti-Xa level (unfractionated heparin)", BLUE, "0.3 to 0.7 IU/mL", None, "STAT 60 minutes",
         "Draw 6 hours after any rate change. Not the same calibration as LC-0920."),
        ("LC-0935", "Carboxyhemoglobin", "pre-heparinized syringe, minimum 1 mL", "below 3 percent in non-smokers",
         "above 20 percent", "STAT 15 minutes", "Pulse oximetry falsely reads normal in carbon monoxide poisoning."),
        ("LC-0945", "Venous blood gas", "pre-heparinized syringe, minimum 1 mL", "pH 7.31 to 7.41; PCO2 41 to 51 mmHg",
         "pH below 7.20", "STAT 15 minutes", "Acceptable instead of an arterial gas for pH and bicarbonate trends."),
    ],
}

for _section, _tests in LABS_EXTRA.items():
    LABS[_section] = sorted(LABS[_section] + _tests, key=lambda t: t[0])  # interleave by code, like the real catalog


def render_labs() -> str:
    lines = ["Halverd Valley Medical Center (HVMC)", "Laboratory Test Catalog and Reference Ranges, 2026", DISCLAIMER,
             "",
             "Critical value policy",
             "A critical value is a result that may indicate an immediately life-threatening condition. The "
             "laboratory phones every critical value to the responsible licensed caregiver within 30 minutes of "
             "result verification, and the receiver must read the value back. If the caregiver cannot be "
             "reached within 30 minutes, the laboratory escalates to the unit charge nurse and then to the "
             "attending physician. Reference ranges below apply to adults.", "",
             "Specimen rejection",
             "Specimens are rejected if unlabeled, labeled with fewer than two patient identifiers, collected in "
             "the wrong tube, clotted when anticoagulated, or underfilled. The lab notifies the unit to redraw.",
             ""]
    for section, tests in LABS.items():
        lines += [f"Section: {section}", ""]
        for code, test, specimen, ref, critical, tat, notes in tests:
            lines += [f"{code} {test}",
                      f"Specimen: {specimen}.",
                      f"Reference range: {ref}.",
                      f"Critical values: {critical}." if critical else "Critical values: none defined.",
                      f"Turnaround time: {tat}.",
                      f"Notes: {notes}", ""]
    return "\n".join(lines)


# ---------- 3. Infusion pump manual ----------

PUMP_INTRO = """Kestrel IV-220 Volumetric Infusion Pump
Operator and Troubleshooting Manual, HVMC edition, revision 7
{disclaimer}

1. Overview
The Kestrel IV-220 is a single-channel volumetric infusion pump used on all HVMC adult inpatient units. It delivers fluids, medications and blood products through the dedicated Kestrel administration set (catalog KS-110) or the blood set (catalog KS-115). Only these sets are validated: third-party tubing causes inaccurate flow and repeated alarms. Each pump is identified by a five-digit asset tag on the back of the housing.

2. Specifications
Flow rate range: 0.1 to 999 mL/h, in 0.1 mL/h steps below 100 mL/h and 1 mL/h steps above.
Volume to be infused (VTBI): 0.1 to 9,999 mL.
Accuracy: plus or minus 5 percent at flow rates of 1 mL/h or higher with a KS-110 set.
Keep vein open (KVO) rate after the VTBI is complete: 1 mL/h by default, or the programmed rate if lower.
Downstream occlusion pressure settings: low 300 mmHg, medium 500 mmHg (factory default), high 750 mmHg.
Air-in-line detection: single bubbles of 50 microliters or larger, or 1 mL of accumulated air within 15 minutes.
Battery: internal lithium-ion battery that runs the pump for up to 8 hours at 25 mL/h when fully charged. A full recharge takes 6 hours while plugged in, and the pump continues to infuse while charging.

3. Drug library and dose error reduction
All pumps carry the HVMC drug library, version DL-24.3, pushed wirelessly by biomedical engineering. Every infusion should be programmed through the drug library rather than in basic mL/h mode, because the library applies soft limits (which can be overridden with a confirmation) and hard limits (which cannot be overridden). High-alert medications such as insulin, opioid infusions, potassium chloride and vasopressors have hard limits. Basic mode is permitted only for maintenance IV fluids. If a hard limit stops a dose that the order requires, verify the order with pharmacy; do not reprogram the drug as a different concentration to get around the limit.

4. Loading the set and priming
Spike the bag and fill the drip chamber halfway. Open the roller clamp and prime the line by gravity until no air remains, then close the roller clamp. Open the pump door, place the set cassette into the channel with the flow arrow pointing toward the patient, and close the door until it clicks. The pump performs a 4-second self-test of the air sensor and pressure sensors every time the door is closed. Open the roller clamp before starting the infusion.

5. Alarm priorities
High-priority alarms show a flashing red light and a repeating three-tone sound; the infusion stops. Examples are occlusions, air in line and door open. Medium-priority alarms show a yellow light and a two-tone sound; the infusion may continue. Examples are low battery and VTBI near end. Low-priority alarms show a steady yellow light and a single tone, such as the reminder that the pump has been left on hold for 2 minutes. Never silence an alarm without identifying its cause. The alarm silence button mutes the sound for 2 minutes only.

6. Cleaning between patients
Wipe all external surfaces with a hospital-approved 70 percent isopropyl alcohol or quaternary ammonium wipe and allow 3 minutes of wet contact time. For patients on contact precautions for Clostridioides difficile, use a sporicidal wipe containing no more than 0.5 percent sodium hypochlorite. Never spray liquid into the door mechanism, and never immerse the pump. Place a green "clean" tag on the pump before returning it to the equipment room.

7. Battery care
Keep pumps plugged in whenever they are not being used for transport. A pump that cannot hold a charge for at least 2 hours must be tagged for biomedical engineering. Do not store pumps with a fully depleted battery for more than 30 days.

8. Fault codes
Fault codes appear on the display as E- followed by two digits. Clinical faults can be cleared at the bedside by following the action listed. Technical faults mean the pump has failed a self-check: stop using the pump, move the infusion to a replacement pump and follow the out-of-service procedure in section 9.
"""

# (code, title, meaning, action, kind)
FAULTS = [
    ("E-01", "Low battery", "Less than 30 minutes of battery remain.",
     "Plug the pump into mains power. The infusion continues.", "clinical, medium priority"),
    ("E-02", "Battery depleted", "Battery exhausted; the infusion has stopped.",
     "Plug in immediately and restart the infusion. If E-02 appears while the pump is plugged in, treat it as "
     "technical fault E-31.", "clinical, high priority"),
    ("E-03", "Door open during infusion", "The pump door was opened while an infusion was running.",
     "Check that the set is correctly loaded, close the door until it clicks, and press RESUME.",
     "clinical, high priority"),
    ("E-04", "Set not recognized", "The cassette is missing, upside down, or is not a Kestrel KS-110 or KS-115 set.",
     "Reload the set with the flow arrow pointing toward the patient. Replace third-party tubing with a "
     "Kestrel set.", "clinical, high priority"),
    ("E-05", "VTBI complete", "The programmed volume has been delivered; the pump is running at the KVO rate.",
     "Hang the next bag or stop the infusion as ordered.", "clinical, medium priority"),
    ("E-06", "Infusion on hold too long", "The pump has been on hold for 2 minutes without restarting.",
     "Restart the infusion or power the pump down.", "clinical, low priority"),
    ("E-07", "Drug library limit", "A hard limit was reached during programming.",
     "Check the order and the units entered. A hard limit cannot be overridden; contact pharmacy if the order "
     "is correct.", "clinical, high priority"),
    ("E-08", "Rate change not confirmed", "A new rate was entered but not confirmed within 30 seconds.",
     "Re-enter the rate and press CONFIRM.", "clinical, low priority"),
    ("E-09", "Secondary infusion complete", "The piggyback infusion has finished and the pump returned to the "
     "primary rate.", "Close the secondary clamp and document the dose.", "clinical, medium priority"),
    ("E-10", "Flow sensor out of range", "The drip chamber flow sensor reports flow that does not match the "
     "programmed rate.", "Check that the drip chamber is half full and vertical. Reprime if needed.",
     "clinical, high priority"),
    ("E-11", "Free-flow risk detected", "The set was removed with the roller clamp open.",
     "Close the roller clamp immediately and check the patient for unintended bolus.", "clinical, high priority"),
    ("E-12", "Downstream occlusion",
     "Pressure between the pump and the patient exceeded the selected occlusion limit.",
     "Check for a closed slide clamp, kinked tubing or an infiltrated IV site. Clear the cause and press RESUME. "
     "If E-12 recurs three times within one hour, assess and replace the IV site.", "clinical, high priority"),
    ("E-13", "Pressure rising", "Downstream pressure has reached 75 percent of the occlusion limit.",
     "Inspect the line and IV site before a full occlusion alarm occurs.", "clinical, medium priority"),
    ("E-14", "Back-pressure after bolus", "Pressure remained high after a bolus.",
     "Pause and allow pressure to fall; check IV site patency.", "clinical, medium priority"),
    ("E-15", "Bolus limit reached", "The requested bolus exceeds the drug library bolus limit.",
     "Confirm the ordered bolus dose; re-enter within limits.", "clinical, medium priority"),
    ("E-16", "Programming timeout", "Programming was left incomplete for 2 minutes.",
     "Restart programming from the drug library screen.", "clinical, low priority"),
    ("E-17", "Air in line", "A bubble of 50 microliters or larger, or accumulated air, passed the air sensor.",
     "Clamp the line, open the door, and remove the air by priming it out through a distal port or by "
     "disconnecting and repriming the set. Never push the air toward the patient. Close the door and press "
     "RESUME.", "clinical, high priority"),
    ("E-18", "Empty container", "Air detected continuously at the top of the set; the bag is likely empty.",
     "Hang a new bag and reprime the line.", "clinical, high priority"),
    ("E-19", "Wrong set for blood", "A blood product was selected but a KS-110 set is loaded.",
     "Replace with a KS-115 blood set.", "clinical, high priority"),
    ("E-20", "Temperature warning", "The pump is operating outside 10 to 40 degrees Celsius.",
     "Move the pump away from heaters or direct sunlight.", "clinical, medium priority"),
    ("E-21", "Upstream occlusion",
     "No fluid is reaching the pump from the bag side.",
     "Check that the bag is not empty, the roller clamp above the pump is open, and the spike is fully inserted "
     "and vented if using a glass bottle. Press RESUME.", "clinical, high priority"),
    ("E-22", "Drug library out of date", "The pump has not received a drug library update in 30 days.",
     "Connect the pump to the network or tag it for biomedical engineering.", "clinical, low priority"),
    ("E-23", "Network connection lost", "The pump cannot report to the infusion server.",
     "The infusion continues. Report persistent loss to the help desk.", "clinical, low priority"),
    ("E-30", "Door latch sensor failure", "The door latch sensor reports an impossible state.",
     "Technical fault. Remove the pump from service.", "technical"),
    ("E-31", "Power supply failure", "The pump is plugged in but not charging.",
     "Technical fault. Remove the pump from service.", "technical"),
    ("E-32", "Motor stall", "The pumping mechanism did not move as commanded.",
     "Technical fault. Remove the pump from service.", "technical"),
    ("E-33", "Pressure sensor self-test failure", "The pressure sensor failed the door-close self-test.",
     "Technical fault. Remove the pump from service.", "technical"),
    ("E-34", "Memory error", "The stored drug library or event log is corrupted.",
     "Technical fault. Remove the pump from service.", "technical"),
    ("E-35", "Keypad stuck", "A key has been reported pressed for more than 20 seconds.",
     "Technical fault. Remove the pump from service.", "technical"),
    ("E-36", "Watchdog reset", "The pump restarted unexpectedly during an infusion.",
     "Technical fault. Remove the pump from service and check the patient for missed or extra volume.",
     "technical"),
    ("E-71", "Air sensor self-test failure", "The ultrasonic air sensor failed its self-test, so air in the line "
     "can no longer be detected.",
     "Technical fault. Stop the infusion and remove the pump from service immediately; do not keep using it "
     "even if fluid appears to flow normally.", "technical"),
]

PUMP_OUTRO = """
9. Out-of-service procedure
Move the infusion to a replacement pump from the equipment room. Leave the administration set in the faulty pump only if a patient was harmed or nearly harmed, so the set can be examined. Attach a red "DO NOT USE" tag, write the fault code, asset tag and a short description on the tag, and call biomedical engineering at extension 4417, available 24 hours a day. Any event that reached the patient must also be reported in the HVMC safety event system within the same shift.

10. Preventive maintenance
Biomedical engineering inspects every pump every 12 months, including an air sensor calibration and a battery capacity test. A pump whose inspection sticker is out of date must not be used.
"""


def render_faults(faults) -> list[str]:
    lines = []
    for code, title, meaning, action, kind in faults:
        lines += [f"{code} {title}", f"Type: {kind}.", f"Meaning: {meaning}", f"Action: {action}", ""]
    return lines


def render_pump() -> str:
    return "\n".join([PUMP_INTRO.format(disclaimer=DISCLAIMER), *render_faults(FAULTS), PUMP_OUTRO])


# Two sibling devices from the same (fictional) maker. They share code NUMBERS with the IV-220 but not meanings:
# "E-17" on a syringe pump is not "E-17" on the IV-220. Answering correctly needs the right manual.
SP40_INTRO = """Kestrel SP-40 Syringe Infusion Pump
Operator and Troubleshooting Manual, HVMC edition, revision 3
{disclaimer}

1. Overview
The Kestrel SP-40 is a syringe pump for small-volume, high-precision infusions. At HVMC it is used in critical care, the neonatal-capable recovery area and for vasoactive infusions such as Norvessa and Vasoplex. It accepts syringes of 5, 10, 20, 30 and 50 mL from the brands listed in the syringe table on the pump; the brand must be confirmed on screen every time a syringe is loaded, because the plunger force calculation depends on it.

2. Specifications
Flow rate range: 0.01 to 200 mL/h with a 50 mL syringe. Bolus rate: up to 1,200 mL/h. Syringe occlusion pressure: adjustable in nine levels from 50 to 900 mmHg, factory default level 5. Near-end-of-infusion warning: 15 minutes before the syringe is empty. Battery: up to 10 hours at 5 mL/h when fully charged.

3. Loading a syringe
Prime the extension line before loading. Lift the plunger clamp, seat the syringe flange in the slot, lower the barrel clamp and close the plunger clamp over the plunger. Confirm the syringe brand and size on screen. Purge the line using the PURGE key, never by pushing the syringe by hand. Mount the pump no more than 30 cm above the patient's heart; raising a syringe pump higher can cause siphoning and an unintended bolus.

4. Changing syringes for vasoactive drugs
Prepare the new syringe before the current one runs out. Use the double-pump method for vasopressors: start the new syringe on a second SP-40 at the same rate, wait until the blood pressure is stable, then stop the old syringe. This avoids the drop in blood pressure that occurs when a vasopressor infusion pauses.

5. Fault codes
Codes on the SP-40 have their own meanings and must not be read from the IV-220 manual.
"""

PCA90_INTRO = """Kestrel PCA-90 Patient-Controlled Analgesia Pump
Operator and Troubleshooting Manual, HVMC edition, revision 5
{disclaimer}

1. Overview
The Kestrel PCA-90 lets a patient give themselves a small, preset dose of opioid by pressing a handset button, within limits set by the prescriber. At HVMC it is used with Hydrovane and Morphalin in prefilled 30 mL cassettes. Only the patient may press the button; family members and staff must never press it for them ("PCA by proxy"), because sedation is the safety signal that stops further doses.

2. Programming parameters
Each PCA order specifies a demand dose, a lockout interval, an optional continuous (basal) rate and a 1-hour or 4-hour dose limit. The lockout interval is the minimum time between doses; button presses during lockout are recorded but not delivered. Continuous basal infusions are not used for opioid-naive patients at HVMC. Two nurses must independently verify the programming against the order at initial setup, at every cassette change and at every handover.

3. The security key
The cassette door is locked with a security key held by the charge nurse. A missing key must be reported to pharmacy within the same shift. Wasting the remaining opioid in a cassette requires a second nurse as witness.

4. Monitoring
Assess the sedation score (POSS) and respiratory rate every hour for the first 12 hours, then every 2 hours. Continuous pulse oximetry is required. If the patient is difficult to rouse, stop the PCA and follow protocol CP-15.

5. Fault codes
Codes on the PCA-90 have their own meanings and must not be read from the IV-220 manual.
"""

SP40_FAULTS = [
    ("E-01", "Syringe not detected", "The barrel clamp does not sense a syringe.", "Reseat the syringe and lower the barrel clamp fully.", "clinical, high priority"),
    ("E-02", "Plunger not engaged", "The plunger clamp is open or the plunger is not captured.", "Close the plunger clamp over the plunger and confirm on screen.", "clinical, high priority"),
    ("E-04", "Syringe brand not confirmed", "A syringe was loaded but its brand was not confirmed within 60 seconds.", "Select the syringe brand and size from the list.", "clinical, medium priority"),
    ("E-05", "Near end of infusion", "About 15 minutes of drug remain in the syringe.", "Prepare the next syringe; use the double-pump method for vasopressors.", "clinical, medium priority"),
    ("E-06", "Syringe empty", "The plunger has reached the end of the syringe.", "Load the new syringe and restart.", "clinical, high priority"),
    ("E-09", "Bolus interrupted", "A bolus was stopped before the full volume was given.", "Check the volume delivered on the history screen before giving more.", "clinical, medium priority"),
    ("E-12", "Syringe line occlusion", "Plunger force exceeded the selected occlusion level.", "Check the extension line and three-way taps for closed positions, then restart. For vasopressors, restart quickly because pressure support has stopped.", "clinical, high priority"),
    ("E-14", "Siphoning risk", "The pump detects a height above the patient that could cause free flow.", "Lower the pump to no more than 30 cm above the heart.", "clinical, high priority"),
    ("E-17", "Plunger clamp opened during infusion", "The plunger clamp was lifted while the syringe was infusing.", "Close the clamp, check the patient for an unintended bolus and restart.", "clinical, high priority"),
    ("E-21", "Rate above drug library limit", "The entered rate exceeds the soft limit for the selected drug.", "Confirm the order and either correct the rate or override with a documented reason.", "clinical, medium priority"),
    ("E-23", "Standby time expired", "The pump has been in standby longer than the set time.", "Restart the infusion or extend standby.", "clinical, low priority"),
    ("E-40", "Drive mechanism fault", "The lead screw did not move as commanded.", "Technical fault. Remove from service and tag for biomedical engineering.", "technical"),
    ("E-41", "Force sensor calibration error", "The plunger force sensor is out of calibration.", "Technical fault. Remove from service.", "technical"),
    ("E-42", "Barrel clamp sensor fault", "The syringe size cannot be measured.", "Technical fault. Remove from service.", "technical"),
    ("E-44", "Battery fault", "The battery cannot be charged.", "Technical fault. Remove from service.", "technical"),
    ("E-71", "Syringe size sensor drift", "The barrel clamp reports a size that does not match the selected syringe.", "Technical fault. Remove from service; volumes delivered may be inaccurate.", "technical"),
]

PCA90_FAULTS = [
    ("E-01", "Cassette not locked", "The cassette door is unlocked.", "Lock the door with the security key held by the charge nurse.", "clinical, high priority"),
    ("E-03", "Handset disconnected", "The patient handset cable is unplugged.", "Reconnect the handset and check that the patient can reach it.", "clinical, medium priority"),
    ("E-05", "Cassette low", "Less than 5 mL remains in the cassette.", "Request a new cassette from pharmacy.", "clinical, medium priority"),
    ("E-06", "Cassette empty", "The cassette is empty; delivery has stopped.", "Change the cassette with a second nurse verifying the programming.", "clinical, high priority"),
    ("E-08", "Dose limit reached", "The 1-hour or 4-hour dose limit has been reached; demand doses are blocked.", "Assess pain and sedation; contact the prescriber if pain is uncontrolled. Do not raise the limit without an order.", "clinical, medium priority"),
    ("E-12", "Patient line occlusion", "The tubing to the patient is blocked.", "Check the clamp and the IV site, then restart.", "clinical, high priority"),
    ("E-17", "Excess demand presses", "More than 10 button presses occurred during one lockout interval.", "Assess pain control; frequent presses during lockout usually mean the regimen is inadequate. Notify the prescriber.", "clinical, low priority"),
    ("E-19", "Programming not verified", "The second-nurse verification step was skipped.", "Complete independent double verification before starting.", "clinical, high priority"),
    ("E-24", "Security key left in door", "The key has remained in the cassette lock for more than 2 minutes.", "Remove the key and return it to the charge nurse.", "clinical, low priority"),
    ("E-52", "Pump door forced", "The locked cassette door was opened without the key.", "Treat as a possible diversion event: secure the pump, notify the charge nurse and pharmacy, and report in the safety event system.", "clinical, high priority"),
    ("E-61", "Handset button stuck", "The handset reports a continuous press.", "Technical fault. Replace the handset; if the code persists, remove the pump from service.", "technical"),
    ("E-62", "Delivery mechanism fault", "The pumping mechanism did not move as commanded.", "Technical fault. Remove from service.", "technical"),
    ("E-71", "Event log full", "The pump history memory is full and older events cannot be overwritten.", "Technical fault. Remove from service so biomedical engineering can download the log for opioid accountability.", "technical"),
]

DEVICE_OUTRO = """
Out-of-service procedure
Attach a red "DO NOT USE" tag with the fault code and asset tag and call biomedical engineering at extension 4417. For any device holding a controlled substance, the remaining drug is wasted with a witness before the device leaves the unit.
"""


def render_sp40() -> str:
    return "\n".join([SP40_INTRO.format(disclaimer=DISCLAIMER), *render_faults(SP40_FAULTS), DEVICE_OUTRO])


def render_pca90() -> str:
    return "\n".join([PCA90_INTRO.format(disclaimer=DISCLAIMER), *render_faults(PCA90_FAULTS), DEVICE_OUTRO])


# ---------- 4. Clinical protocols (prose) ----------

PROTOCOLS = """Halverd Valley Medical Center (HVMC)
Nursing and Medical Clinical Protocols, 2026
{disclaimer}

These protocols apply to adult inpatients on all HVMC units unless a unit-specific protocol says otherwise. Drug names refer to the HVMC Adult Inpatient Formulary.

CP-02 Rapid response team activation
Any staff member, patient or family member may call the rapid response team (RRT) by dialing 5555. No physician order is needed and nobody is criticized for calling. Call the RRT for acute changes including a respiratory rate below 8 or above 28 breaths per minute, oxygen saturation below 90 percent despite supplemental oxygen, heart rate below 40 or above 130 beats per minute, systolic blood pressure below 90 mmHg, new confusion or a sudden drop in level of consciousness, seizure, or simply staff worry that the patient is deteriorating. The team, made up of a critical care nurse, a respiratory therapist and an ICU resident, aims to arrive within 10 minutes. The primary nurse stays at the bedside, gives a handover in SBAR format and brings the latest vital signs and medication record. The attending physician is notified of every RRT call within one hour.

CP-03 Hypoglycemia management
Hypoglycemia is a blood glucose below 70 mg/dL. Treat it immediately; do not wait for a provider order. If the patient is alert and can swallow, give 15 g of fast-acting carbohydrate such as 120 mL of juice or 4 glucose tablets. If the patient cannot safely swallow and has IV access, give Dextriva 25 g (50 mL) IV push. If there is no IV access, give Glucarin 3 mg intranasally. Recheck the glucose 15 minutes after any treatment. If it is still below 70 mg/dL, repeat the treatment and recheck again in 15 minutes. If glucose remains below 70 mg/dL after two treatments, call the rapid response team. Once glucose is 70 mg/dL or higher, give a snack containing protein and complex carbohydrate if the next meal is more than one hour away. Notify the provider of every hypoglycemic event, review the insulin orders, and document the event in the glucose record. A glucose below 40 mg/dL is a severe event and requires a medication safety report.

CP-05 Venous thromboembolism prophylaxis
Every adult admitted to HVMC has a VTE risk assessment using the Padua score within 24 hours of admission and again at any transfer of care. Medical patients with a Padua score of 4 or higher are high risk. Unless contraindicated, high-risk patients receive pharmacological prophylaxis with Heparivex 40 mg subcutaneously once daily (30 mg if CrCl is below 30 mL/min). Contraindications include active bleeding, platelets below 50 x10^9/L, and neuraxial anesthesia catheter placement or removal within the previous 12 hours. Patients with contraindications receive intermittent pneumatic compression devices on both legs, worn at least 18 hours per day. Graduated compression stockings alone are not adequate prophylaxis. Patients already on therapeutic anticoagulation such as Corvatrel or Corvetrol do not need additional prophylaxis. Early walking is encouraged for all patients able to do so.

CP-07 Sepsis recognition and the one-hour bundle
Screen every patient for sepsis on admission and every 12 hours. A positive screen is suspected infection plus two or more of: temperature above 38.3 or below 36.0 degrees Celsius, heart rate above 90 beats per minute, respiratory rate above 20 breaths per minute, new confusion, systolic blood pressure below 100 mmHg, or WBC above 12.0 or below 4.0 x10^9/L. A positive screen starts the clock, and the following one-hour bundle must be completed within 60 minutes of recognition. First, measure a lactate (LC-0710). Second, draw two sets of blood cultures (LC-0730) before antibiotics, as long as this does not delay antibiotics by more than 45 minutes. Third, give broad-spectrum antibiotics: the default empiric regimen is Zephacillin plus Vancorin, with Meronex for patients with prior ESBL organisms. Fourth, for hypotension or a lactate of 4.0 mmol/L or above, give 30 mL/kg of balanced crystalloid within the first 3 hours. Fifth, if mean arterial pressure stays below 65 mmHg during or after fluid resuscitation, start Norvessa, which is the first-line vasopressor. Repeat the lactate within 2 to 4 hours if the first value was above 2.0 mmol/L. Septic shock is defined as needing a vasopressor to keep the mean arterial pressure at 65 mmHg or higher with a lactate above 2.0 mmol/L despite adequate fluids. Every case where the bundle is missed is reviewed by the sepsis committee.

CP-09 Suspected transfusion reaction
Signs of a transfusion reaction include fever of 1 degree Celsius or more above baseline, chills, hives, itching, shortness of breath, low blood pressure, back or flank pain, and dark urine. If any sign appears, stop the transfusion immediately and keep the IV line open with normal saline using new tubing; do not flush the blood remaining in the old tubing into the patient. Check vital signs every 15 minutes, notify the provider and the blood bank, and recheck the patient identification and the blood product label at the bedside. Send the blood bag, the attached tubing and a fresh blood sample from the patient to the blood bank for investigation, together with the first voided urine sample. Do not restart the same unit. Mild hives alone may allow the transfusion to resume only after a provider assesses the patient and gives an order. Every transfusion reaction is reported to the blood bank within the same shift, even if symptoms resolve.

CP-11 Fall prevention
Assess fall risk with the Morse Fall Scale on admission, every shift, after any fall, and after any change in condition. A Morse score of 45 or higher is high risk. Universal precautions for all patients include the bed in the lowest position with brakes locked, the call light and personal items within reach, non-slip footwear, and adequate lighting. High-risk patients also receive hourly purposeful rounding that covers pain, toileting, position and possessions, a bed or chair exit alarm, a yellow wristband and door sign, and scheduled toileting every 2 hours while awake. Review medications that increase fall risk, including opioids, benzodiazepines, Pregavel and blood pressure drugs. After a fall, do not move the patient until assessed for injury, check neurological status if the head was struck or the fall was unwitnessed, and notify the provider. Patients on anticoagulants who strike their head need a CT scan of the head ordered.

CP-13 Pressure injury prevention
Assess every patient with the Braden Scale on admission and every shift. A Braden score of 18 or below indicates risk. At-risk patients are repositioned at least every 2 hours in bed and shifted every 15 minutes when sitting in a chair, and placed on a pressure-redistribution mattress. Keep the head of the bed at 30 degrees or lower unless contraindicated, to reduce shear. Float the heels off the bed with a pillow under the calves or with heel offloading boots. Inspect the skin under medical devices such as oxygen tubing and splints at least twice each shift. Manage moisture with barrier cream and breathable pads. Any new pressure injury of stage 2 or above is reported to the wound care nurse within 24 hours.

CP-15 Opioid-induced respiratory depression
Patients receiving opioids are monitored with the Pasero Opioid-Induced Sedation Scale (POSS) and respiratory rate. Hold the opioid if the POSS score is 3 (frequently drowsy, drifts off to sleep during conversation) and notify the provider. If the respiratory rate falls below 8 breaths per minute or the POSS score reaches 4 (minimal or no response to physical stimulation), stop all opioids and sedatives, stimulate the patient, apply oxygen, call the rapid response team and give Naloxar 0.04 mg IV every 2 minutes until the respiratory rate is at least 10. Continuous pulse oximetry and capnography are required for 24 hours after any naloxone rescue, and for patients with obstructive sleep apnea who receive opioid infusions.

CP-17 Alcohol withdrawal
Ask every patient about alcohol use on admission with the AUDIT-C questions. Patients who drink daily or report previous withdrawal are assessed with the CIWA-Ar scale every 4 hours, increasing to every hour once the score is 10 or above. Calmazine is given on a symptom-triggered basis when the CIWA-Ar score is 10 or more. A score above 20, a seizure, or hallucinations require a rapid response team call and ICU evaluation. Give thiamine 100 mg IV before any glucose-containing fluid to prevent Wernicke encephalopathy, and continue it daily for at least 3 days.

CP-19 Discharge medication reconciliation
Before discharge, the pharmacist or discharging nurse compares the home medication list with the inpatient orders and the discharge prescriptions. Every home medication is marked as continue, change or stop, with a reason for each change. Anticoagulants, insulin and opioids receive pharmacist counselling before the patient leaves. The reconciled list is given to the patient in plain language and sent to the primary care provider within 24 hours of discharge.
"""


PROTOCOLS_EXTRA = """
CP-21 Stroke alert
Suspect a stroke when a patient develops sudden loss of balance, blurred or lost vision, facial droop, arm or leg weakness, or slurred or absent speech (the BE-FAST signs). Dial 5555 and request a "stroke alert", and record the exact time the patient was last known to be well, because every treatment decision depends on it. Check a point-of-care glucose immediately, since low glucose can mimic a stroke. The stroke team aims for a CT scan of the head within 25 minutes of the alert. Thrombolysis can be considered up to 4.5 hours after the patient was last known well, and blood pressure must be below 185/110 mmHg before it is given. Keep the patient nil by mouth until a swallow screen (CP-45) is passed, including oral medications.

CP-23 Acute chest pain
Obtain a 12-lead ECG within 10 minutes of any report of new chest pain and show it to a physician immediately. If the ECG shows ST elevation, page the cardiac catheterization team on pager 7700; the goal from first medical contact to balloon inflation is 90 minutes. Unless the patient is allergic or actively bleeding, give aspirin 324 mg chewed. Send high-sensitivity troponin (LC-0815) at presentation and again at 1 hour using the 0/1-hour algorithm. Place the patient on continuous cardiac monitoring, keep them in bed, and give oxygen only if the oxygen saturation is below 90 percent.

CP-25 Anaphylaxis
Anaphylaxis is a severe allergic reaction affecting the airway, breathing or circulation, often with hives or swelling. Stop the suspected trigger immediately, whether a drug infusion, blood product or food. Give epinephrine 0.5 mg (0.5 mL of the 1 mg/mL concentration) intramuscularly into the outer thigh, and repeat every 5 to 15 minutes if there is no improvement. Call the rapid response team. Lay the patient flat with the legs raised, or sitting up if breathing is difficult, and give a 1 to 2 litre bolus of crystalloid for low blood pressure. Antihistamines and steroids are add-on treatments and never replace epinephrine. Observe the patient for at least 6 hours after symptoms resolve because a second, delayed reaction can occur. Record the allergy in the health record before the end of the shift.

CP-27 Diabetic ketoacidosis
Diabetic ketoacidosis (DKA) is diagnosed when glucose is above 250 mg/dL, venous pH is below 7.30 or bicarbonate is below 18 mmol/L, and serum ketones (LC-0452) are raised. Start with 1 litre of balanced crystalloid in the first hour. Check potassium before insulin: if it is below 3.3 mmol/L, replace potassium first and delay insulin. Then start an intravenous insulin infusion at 0.1 units/kg/h. Check glucose every hour and electrolytes every 2 to 4 hours. When glucose falls below 250 mg/dL, add dextrose to the fluids rather than stopping insulin, because insulin is what clears the ketones. DKA has resolved when the anion gap (LC-0432) is 12 mmol/L or below. Give subcutaneous basal insulin 2 hours before stopping the infusion so the patient is never without insulin.

CP-29 Hyperkalemia
Treat potassium of 6.0 mmol/L or above, or any potassium rise with ECG changes, as an emergency. Repeat the test if the sample was hemolyzed and the ECG is normal. If the ECG shows peaked T waves, widened QRS or arrhythmia, give 30 mL of 10 percent calcium gluconate IV over 10 minutes to protect the heart. Shift potassium into the cells with 10 units of regular insulin IV together with 25 g of dextrose, then check glucose every hour for 4 hours because of the risk of late hypoglycemia. Stop all potassium sources, including Kaliveen, Spirolan and potassium in IV fluids. Recheck potassium 1 to 2 hours after treatment. Persistent hyperkalemia in kidney failure needs a nephrology consult for dialysis.

CP-31 Delirium prevention and management
Screen every patient over 65, and every ICU patient, for delirium each shift with the Confusion Assessment Method (CAM). Delirium is an acute change in attention and thinking that fluctuates during the day. Prevention relies on non-drug measures: reorientation with clocks and calendars, glasses and hearing aids in place, a quiet dark room at night, avoiding sleep interruptions between midnight and 5 a.m., early mobilization, and family presence. Look for causes such as infection, urinary retention, constipation, pain, dehydration and new medications, especially benzodiazepines and anticholinergic drugs. Avoid physical restraints. Halopran is reserved for agitation that endangers the patient or staff after non-drug measures have failed.

CP-33 Central venous catheter care
Every central line is reviewed daily on rounds, and the line is removed as soon as it is no longer essential. Scrub the hub with chlorhexidine for 15 seconds and let it dry before every access. Transparent dressings are changed every 7 days, and gauze dressings every 48 hours, or sooner if damp, loose or soiled. Administration sets for continuous fluids are changed every 96 hours, but sets used for blood or lipid infusions are changed every 24 hours. Bathe patients with central lines daily with chlorhexidine cloths. A new fever in a patient with a central line triggers blood cultures from the line and from a peripheral vein.

CP-35 Indwelling urinary catheter management
Insert a urinary catheter only for an approved reason: acute urinary retention, accurate output monitoring in critical illness, selected surgery, comfort at the end of life, or healing of a sacral wound in an incontinent patient. Remove catheters placed for surgery within 24 hours after the operation unless the surgeon documents a reason. Before re-inserting a catheter, scan the bladder: straight catheterize if the volume is above 400 mL. Keep the drainage bag below the level of the bladder and off the floor, and keep the tubing free of kinks. Never send a urine culture from the drainage bag.

CP-37 Isolation precautions
Contact precautions (gown and gloves) are used for MRSA, VRE and other multidrug-resistant organisms. Contact-plus precautions are used for Clostridioides difficile: staff must wash hands with soap and water because alcohol rub does not kill the spores, and the room is cleaned with a sporicidal agent. Droplet precautions require a surgical mask within 2 metres of the patient, and are used for influenza and pertussis. Airborne precautions require a fit-tested N95 respirator and a negative-pressure room with the door closed, and are used for tuberculosis, measles and chickenpox. Patients on any isolation leave the room only for essential tests, wearing a mask if on droplet or airborne precautions.

CP-39 Restraint and seclusion
Restraints are a last resort after less restrictive measures have failed. A physician order is required within 1 hour of applying any restraint. Orders for non-violent restraint, such as preventing a patient from pulling out a breathing tube, are renewed every 24 hours. Orders for violent or self-destructive behavior are renewed every 4 hours for adults, and a physician must see the patient face to face within 1 hour. Patients in violent restraints are checked every 15 minutes; patients in non-violent restraints every 2 hours, including skin, circulation, toileting, fluids and readiness for release. Release must happen as soon as the behavior that required it has stopped.

CP-41 Pain assessment
Ask every patient who can communicate to rate pain on the 0 to 10 numeric rating scale at least every 4 hours and before and after every pain intervention. For patients with advanced dementia, use the PAINAD scale; for intubated or sedated ICU patients, use the Critical-Care Pain Observation Tool (CPOT). Reassess 30 minutes after an IV analgesic dose and 60 minutes after an oral dose. A pain goal is agreed with each patient and documented. Use a multimodal approach starting with Acetavol or oral acetaminophen and adding opioids only when needed.

CP-43 Preoperative fasting and medication management
Patients may eat a light meal up to 6 hours before anesthesia, and fatty or fried food up to 8 hours before. Clear fluids such as water, apple juice and black coffee are encouraged until 2 hours before. Chewing gum does not delay surgery. Hold Corvatrel for 48 hours before procedures with high bleeding risk. Hold Corvetrol for 2 to 4 days depending on kidney function. Continue beta blockers and statins on the morning of surgery. Give 80 percent of the usual basal insulin dose the evening before surgery and hold mealtime insulin while fasting.

CP-45 Dysphagia screening
Every patient admitted with a stroke, and any patient with new coughing or choking at meals, must pass a swallow screen before taking anything by mouth, including tablets. The nurse performs the 3-ounce water swallow test: the patient drinks 90 mL of water from a cup without stopping. Coughing, choking, a wet voice or stopping is a failure. A patient who fails stays nil by mouth and is referred to speech-language pathology, who assess within 24 hours. Patients with dysphagia sit fully upright for all meals and for 30 minutes afterwards.

CP-47 Massive transfusion protocol
Activate the massive transfusion protocol (MTP) by dialing 5555 and stating "massive transfusion" with the patient's location. Expect uncontrolled bleeding requiring more than 4 units of red cells in 1 hour. The blood bank releases coolers containing 6 units of red cells, 6 units of plasma and 1 pool of platelets, aiming for a 1:1:1 ratio. Warm all products through a rapid infuser. Send fibrinogen (LC-0630), ionized calcium (LC-0462) and a full blood count every 30 to 60 minutes. Give cryoprecipitate if fibrinogen is below 150 mg/dL, and calcium if ionized calcium is below 1.1 mmol/L. The MTP is stopped by the team leader, who must call the blood bank to return unused products.

CP-49 Point-of-care glucose testing
Only staff with current competency may use the bedside glucose meters. Run the meter's quality control check once every 24 hours and whenever results seem inconsistent with the patient's condition. Clean the side of the fingertip with alcohol, let it dry, and wipe away the first drop if the skin may be contaminated with sugar. Do not test fingers on the side of an IV infusion. Results below 40 or above 500 mg/dL must be confirmed with a laboratory plasma glucose. Meters are cleaned with a disinfectant wipe after every patient.
"""


# (unit, location, extension, specialty)
UNITS = [
    ("2 North", "Main Building, floor 2", "3201", "general medicine"),
    ("2 South", "Main Building, floor 2", "3205", "general medicine and geriatrics"),
    ("3 North", "Main Building, floor 3", "3301", "cardiology and telemetry"),
    ("3 South", "Main Building, floor 3", "3305", "nephrology and dialysis"),
    ("4 North", "Main Building, floor 4", "3401", "general surgery"),
    ("4 South", "Main Building, floor 4", "3405", "orthopedics"),
    ("5 North", "Main Building, floor 5", "3501", "oncology and hematology"),
    ("5 South", "Main Building, floor 5", "3505", "neurology and stroke unit"),
    ("Medical ICU", "Tower A, floor 3", "3610", "critical care"),
    ("Surgical ICU", "Tower A, floor 4", "3620", "critical care"),
    ("Cardiac ICU", "Tower A, floor 5", "3630", "critical care"),
    ("Step-down unit", "Tower A, floor 2", "3640", "intermediate care"),
    ("Emergency department", "Ground floor, east entrance", "3000", "emergency medicine"),
    ("Post-anesthesia care unit", "Tower B, floor 2", "3710", "recovery"),
    ("Operating rooms", "Tower B, floor 2", "3700", "surgery"),
    ("Labor and delivery", "Tower B, floor 3", "3800", "obstetrics"),
    ("Behavioral health unit", "West Wing, floor 1", "3900", "psychiatry"),
    ("Rehabilitation unit", "West Wing, floor 2", "3910", "physical rehabilitation"),
    ("Endoscopy suite", "Main Building, floor 1", "3110", "gastroenterology"),
    ("Cardiac catheterization lab", "Tower A, floor 1", "3120", "interventional cardiology"),
    ("Infusion center", "Outpatient Pavilion, floor 1", "3130", "outpatient infusions"),
    ("Dialysis unit", "Main Building, floor 3", "3310", "hemodialysis"),
]

# (service, contact)
SERVICES = [
    ("Rapid response team, stroke alert and massive transfusion activation", "dial 5555"),
    ("Cardiac catheterization team (STEMI)", "pager 7700"),
    ("Biomedical engineering, 24 hours", "extension 4417"),
    ("Central pharmacy", "extension 4200"),
    ("Clinical pharmacist on call (evenings and nights)", "pager 7210"),
    ("Blood bank", "extension 4310"),
    ("Core laboratory", "extension 4350"),
    ("Microbiology", "extension 4360"),
    ("Radiology reading room", "extension 4500"),
    ("Infection prevention", "extension 4620"),
    ("Wound care nurse", "pager 7340"),
    ("Speech-language pathology", "extension 4710"),
    ("Respiratory therapy", "pager 7400"),
    ("Security", "extension 4911"),
    ("Interpreter services", "extension 4720"),
    ("Patient safety event line", "extension 4800"),
    ("Information technology help desk", "extension 4100"),
    ("Environmental services (spills and terminal cleans)", "extension 4150"),
]

ON_CALL = [
    ("Infectious diseases", "pager 7150", "ID approval for restricted antimicrobials such as Meronex is requested through this pager."),
    ("Hematology", "pager 7160", "Approves Revexa and advises on bleeding in anticoagulated patients."),
    ("Nephrology", "pager 7170", "Urgent dialysis decisions, including refractory hyperkalemia."),
    ("Neurology (stroke)", "pager 7180", "Joins every stroke alert."),
    ("General surgery", "pager 7190", "Surgical consults and acute abdomen."),
    ("Psychiatry", "pager 7200", "Capacity assessments and behavioral emergencies."),
    ("Anesthesia (pain service)", "pager 7220", "Epidurals, nerve block catheters and PCA problems."),
    ("Endocrinology", "pager 7230", "Insulin infusions outside the ICU and difficult glucose control."),
]


def render_directory() -> str:
    lines = ["Halverd Valley Medical Center (HVMC)", "Clinical Unit and Service Directory, 2026", DISCLAIMER, "",
             "Internal extensions are four digits; dial 9 for an outside line. Pagers are reached by dialing 7 "
             "followed by the pager number's last three digits, then entering a callback extension.", "",
             "Section: Inpatient and procedural units", ""]
    for unit, location, ext, specialty in UNITS:
        lines += [f"{unit}: {specialty}. Location: {location}. Charge nurse: extension {ext}.", ""]
    lines += ["Section: Hospital services", ""]
    for service, contact in SERVICES:
        lines += [f"{service}: {contact}.", ""]
    lines += ["Section: Physician on-call services", ""]
    for service, pager, note in ON_CALL:
        lines += [f"{service} on call: {pager}. {note}", ""]
    return "\n".join(lines)


# ---------- IV compatibility chart and antibiogram (generated tables) ----------

IV_DRUGS = ["Zephacillin", "Cefalyn", "Meronex", "Vancorin", "Azithrel", "Tarnaflox", "Nephrazole", "Lumidex",
            "Ondelix", "Pantorex", "Morphalin", "Hydrovane", "Acetavol", "Ketrolin", "Sedoflux", "Halopran",
            "Calmazine", "Furovex", "Norvessa", "Vasoplex", "Kaliveen", "Magnora", "Prednavex", "Heparivex",
            "Dextriva", "Fosferra"]
COMPAT_NOTES = {
    "compatible": ["Compatible at standard concentrations for at least 4 hours.",
                   "Compatible; flush the line with 10 mL saline between doses when running intermittently.",
                   "Compatible at the HVMC standard concentrations in the drug library."],
    "incompatible": ["Incompatible: forms a precipitate. Use a separate lumen or a separate line.",
                     "Incompatible: visible haze within 1 hour. Never run through the same Y-site.",
                     "Incompatible: loss of potency of the first drug. Use a separate lumen."],
    "variable": ["Variable: compatible only below the standard concentration; check with pharmacy.",
                 "No data: treat as incompatible until pharmacy confirms.",
                 "Variable: depends on the diluent; compatible in saline but not in dextrose."],
}


def render_compatibility() -> str:
    rng = random.Random(7)
    lines = ["Halverd Valley Medical Center (HVMC)", "Y-Site IV Compatibility Chart, 2026", DISCLAIMER, "",
             "Y-site compatibility means two drugs can meet in the same IV line for the short time they mix at the "
             "connection. It does not mean they can be mixed in the same bag. When a pair is not listed, call "
             "central pharmacy at extension 4200 before running them together. Blood products are compatible only "
             "with normal saline.", ""]
    for i, first in enumerate(IV_DRUGS):
        lines += [f"Section: {first} Y-site compatibility", ""]
        for second in IV_DRUGS[i + 1:]:
            status = rng.choices(["compatible", "incompatible", "variable"], weights=[6, 2, 2])[0]
            lines.append(f"{first} with {second}: {rng.choice(COMPAT_NOTES[status])}")
        lines.append("")
    return "\n".join(lines)


ORGANISMS = ["Escherichia coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa", "Enterobacter cloacae",
             "Proteus mirabilis", "Staphylococcus aureus (MSSA)", "Staphylococcus aureus (MRSA)",
             "Enterococcus faecalis", "Streptococcus pneumoniae", "Acinetobacter baumannii"]
ANTIBIOTICS = ["Zephacillin", "Cefalyn", "Meronex", "Vancorin", "Tarnaflox", "Azithrel"]


def render_antibiogram() -> str:
    rng = random.Random(11)
    lines = ["Halverd Valley Medical Center (HVMC)", "Cumulative Antibiogram, January to December 2025", DISCLAIMER, "",
             "The antibiogram reports the percentage of first isolates from each organism that tested susceptible "
             "to each antibiotic, one isolate per patient per year. Results are shown only for organisms with at "
             "least 30 isolates. Use it to choose empiric therapy before culture results are back; once results "
             "are available, narrow therapy to the individual susceptibility report. An agent below 80 percent "
             "susceptibility is not recommended for empiric use against that organism.", ""]
    for org in ORGANISMS:
        n = rng.randint(34, 612)
        lines += [f"Organism: {org} ({n} isolates)"]
        for abx in ANTIBIOTICS:
            if abx == "Vancorin" and not any(k in org for k in ("Staphylococcus", "Enterococcus", "Streptococcus")):
                continue  # glycopeptides have no gram-negative activity
            pct = rng.randint(55, 100) if "Acinetobacter" not in org else rng.randint(20, 70)
            lines.append(f"{org} susceptibility to {abx}: {pct} percent.")
        lines.append("")
    lines += ["Trends", "ESBL-producing Escherichia coli rose from 11 to 14 percent of urinary isolates. Meronex "
              "remains reliable against ESBL organisms and is restricted to protect this. MRSA made up 31 percent of "
              "Staphylococcus aureus bloodstream isolates. The antimicrobial stewardship committee reviews these "
              "figures every spring and updates the sepsis order set (CP-07) if empiric choices need to change."]
    return "\n".join(lines)


# ---------- Patient monitor manual ----------

MONITOR_INTRO = """Talon VM-8 Bedside Patient Monitor
Quick Reference and Alarm Guide, HVMC edition
{disclaimer}

1. Overview
The Talon VM-8 displays ECG, heart rate, non-invasive blood pressure, oxygen saturation (SpO2), respiratory rate, temperature and, when the module is fitted, end-tidal carbon dioxide (EtCO2). Every inpatient bed on the telemetry, step-down and ICU units has a VM-8 that reports to the central station at the nurses' desk. Monitors on general wards are portable and do not report centrally unless docked.

2. Default alarm limits for adults
Heart rate: low 50, high 120 beats per minute. SpO2: low 90 percent. Respiratory rate: low 8, high 30 breaths per minute. Systolic blood pressure: low 90, high 180 mmHg. EtCO2: low 25, high 50 mmHg. Limits may be changed only with a provider order or under a unit protocol, and changes must be documented. Alarm limits reset to the defaults when a new patient is admitted to the monitor.

3. Reducing nuisance alarms
Change ECG electrodes every 24 hours after cleaning the skin with soap and water. Place the SpO2 sensor on a warm finger without nail polish. Use the correct cuff size: the bladder should cover 80 percent of the arm circumference. Never turn alarm volume below the unit minimum or disable an alarm category.

4. Alarm messages
"""

# (message, meaning, action, priority)
MONITOR_ALARMS = [
    ("ASYSTOLE", "No QRS complexes detected for 4 seconds.", "Check the patient immediately; start resuscitation if unresponsive and pulseless.", "red, highest priority"),
    ("VFIB/VTACH", "Ventricular fibrillation or sustained ventricular tachycardia detected.", "Check the patient immediately; call a code if pulseless.", "red, highest priority"),
    ("BRADY", "Heart rate below the low limit.", "Assess the patient and look for drug causes such as Metavol or Sedoflux.", "yellow, medium priority"),
    ("TACHY", "Heart rate above the high limit.", "Assess for pain, fever, bleeding or arrhythmia.", "yellow, medium priority"),
    ("SPO2 LOW", "Oxygen saturation below the low limit for 10 seconds.", "Check the probe and the patient's breathing; give oxygen per order.", "yellow, medium priority"),
    ("SPO2 PROBE OFF", "The oxygen sensor is not on the patient.", "Reattach the sensor.", "cyan, technical"),
    ("RR LOW", "Respiratory rate below the low limit.", "Assess sedation and breathing; consider opioid-induced respiratory depression (CP-15).", "yellow, medium priority"),
    ("APNEA", "No breath detected for 20 seconds.", "Stimulate the patient and assess airway and breathing immediately.", "red, highest priority"),
    ("ETCO2 HIGH", "End-tidal carbon dioxide above the high limit.", "Assess for hypoventilation, especially in patients on opioids.", "yellow, medium priority"),
    ("NBP CUFF LEAK", "The blood pressure cuff did not inflate.", "Check the cuff and hose connections.", "cyan, technical"),
    ("LEADS OFF", "One or more ECG electrodes are disconnected.", "Replace electrodes; check skin preparation.", "cyan, technical"),
    ("ARTIFACT", "Signal noise prevents analysis.", "Check electrodes and patient movement.", "cyan, technical"),
    ("BATTERY LOW", "Less than 20 minutes of battery remain on a portable monitor.", "Dock or plug in the monitor.", "cyan, technical"),
    ("NO CENTRAL", "The monitor has lost its connection to the central station.", "Check the network cable or wireless status; inform the charge nurse because alarms are not reaching the desk.", "yellow, technical"),
]


def render_monitor() -> str:
    lines = [MONITOR_INTRO.format(disclaimer=DISCLAIMER)]
    for msg, meaning, action, priority in MONITOR_ALARMS:
        lines += [f"Alarm {msg}", f"Priority: {priority}.", f"Meaning: {meaning}", f"Action: {action}", ""]
    lines.append("5. Service\nFaulty monitors are tagged and reported to biomedical engineering at extension 4417.")
    return "\n".join(lines)


# ---------- Patient leaflets (plain language) ----------
# Deliberately written in everyday words, close to how the plain-language golden questions are phrased.
# They make dense retrieval's job harder: a leaflet "sounds like" the question but the protocol holds the answer.

LEAFLETS = """Halverd Valley Medical Center (HVMC)
Patient and Family Information Leaflets
{disclaimer}

Leaflet: Low blood sugar in hospital
If you have diabetes, your blood sugar can sometimes drop too low while you are in hospital, especially if you are eating less than usual or your medicines have changed. Low sugar can make you feel shaky, sweaty, hungry, dizzy or confused. Tell your nurse straight away if you feel like this. Your nurse will check your sugar with a small finger-prick test and give you something sugary to eat or drink, or medicine through your drip if you cannot eat. They will check again a little later to make sure your sugar has come back up. Please do not skip meals without telling the nurse, because your insulin doses may need to change.

Leaflet: Preventing falls during your stay
Being in hospital can make you unsteady. Illness, new medicines, being in bed for a long time and an unfamiliar room all make falls more likely. Please call for help before getting up if you have been told you are at risk, wear the non-slip socks we give you, and keep your glasses and walking aid close by. Staff will visit you regularly to ask whether you need the toilet, a drink or help changing position. Some beds have an alarm that tells staff when you get up. This is not a punishment; it is there to keep you safe.

Leaflet: Your pain control pump (PCA)
A PCA pump lets you give yourself a small dose of pain medicine by pressing a button. The pump is set so you cannot give yourself too much. Only you should press the button, never your visitors or family, even if you are asleep and seem uncomfortable. If you feel very sleepy, find it hard to breathe, feel sick or itchy, tell your nurse. Your nurse will check your breathing and how awake you are regularly.

Leaflet: Blood clots and how we prevent them
Lying in bed in hospital makes blood clots in the legs more likely. A clot can travel to the lungs, which is serious. Most patients receive a small daily injection under the skin to thin the blood a little. If you cannot have the injection, we may give you leg sleeves that gently squeeze your calves. Walking, moving your feet and ankles and drinking enough fluids also help. Tell us if one leg becomes swollen, painful or red, or if you suddenly feel short of breath.

Leaflet: Sepsis, what families should know
Sepsis is when the body's response to an infection starts to harm its own organs. It can start with a chest, urine, skin or tummy infection. Warning signs include a high or very low temperature, shivering, fast breathing, a racing heart, feeling very unwell, or new confusion. If you are worried that your relative is getting worse, tell the nurse and ask "could this be sepsis?" Staff will act quickly with tests, fluids and antibiotics. You can also call the hospital's emergency team yourself by dialing 5555 from any bedside phone.

Leaflet: Having a blood transfusion
A blood transfusion replaces blood you have lost or cannot make. Before it starts, two staff members check your name and date of birth against the blood bag. Most transfusions take 2 to 4 hours per bag. Tell your nurse at once if you feel hot, shivery, itchy, short of breath, or have pain in your back during the transfusion. The nurse will stop it and check you. Reactions are rare and usually mild.

Leaflet: Your drip (IV) and the pump
A drip is a thin plastic tube in a vein, usually in your hand or arm, that gives you fluids or medicines. A pump controls how fast the fluid goes in. If the pump beeps, press the call button rather than touching the pump yourself. Tell the nurse if the area around the drip becomes swollen, painful, red, cold or leaks. The pump can run on its battery for a while if you need to walk to the bathroom, but it should be plugged back in when you return.

Leaflet: Going home with new medicines
Before you leave, a pharmacist or nurse will go through every medicine with you: which to keep taking, which have changed and which to stop. Ask questions until you are sure. Bring your list to your family doctor. Blood thinners, insulin and strong pain medicines need extra care, so make sure you understand how and when to take them and what side effects to look out for.
"""


# ---------- Policy manual ----------

POLICIES = """Halverd Valley Medical Center (HVMC)
Selected Hospital-Wide Clinical Policies
{disclaimer}

Policy HP-04 Patient identification
Use two identifiers, the patient's full name and date of birth, before giving any medication, blood product or treatment, collecting any specimen, or performing any procedure. The room or bed number is never an identifier. Ask the patient to state their name and date of birth rather than reading it to them. If the patient cannot respond, check the wristband against the order. A patient without a wristband receives no treatment except in an emergency until one is applied.

Policy HP-07 Verbal and telephone orders
Verbal orders are accepted only in emergencies or when the prescriber is physically unable to write or enter the order, such as during a sterile procedure. Telephone orders are accepted when the prescriber is off site. The receiving nurse writes the order down and reads it back in full, including drug name, dose, route and frequency, and the prescriber confirms. Numbers between 13 and 19 are spelled out digit by digit ("one-five") to avoid confusion with 30 to 90. Verbal and telephone orders must be signed by the prescriber within 24 hours. Verbal orders are never accepted for chemotherapy.

Policy HP-11 High-alert medications
High-alert medications carry a heightened risk of significant harm when used in error. At HVMC they include all insulins, opioids, IV potassium chloride, anticoagulants, vasopressors, neuromuscular blockers and chemotherapy. These drugs are stored separately with warning labels. Insulin, opioid infusions, PCA, IV potassium and heparin infusions require an independent double check by a second nurse, which means the second nurse reviews the order, the drug, the dose calculation and the pump programming without being prompted by the first nurse.

Policy HP-15 Hand hygiene
Clean hands at the five moments: before touching a patient, before a clean or aseptic procedure, after body fluid exposure risk, after touching a patient, and after touching the patient's surroundings. Alcohol-based hand rub is the default, rubbed for 20 seconds until dry. Wash with soap and water when hands are visibly soiled, after caring for a patient with Clostridioides difficile or norovirus, and before eating. Artificial nails and nail extensions are not permitted for staff who provide direct patient care.

Policy HP-19 Sharps safety
Use safety-engineered needles and never recap a used needle. Dispose of sharps at the point of use in a sharps container, which is replaced when three-quarters full. A needlestick or splash injury is first-aided immediately by washing with soap and water, then reported to employee health at extension 4650 within 1 hour, because post-exposure treatment for HIV works best when started within 2 hours.

Policy HP-23 Handover between shifts
Nurses hand over at the bedside using the SBAR structure: situation, background, assessment and recommendation. The outgoing and incoming nurse check together the patient's identity, infusions and pump settings, drains, and any high-alert medications. Patients and families are invited to take part. Handover is not complete until the incoming nurse has had the chance to ask questions.

Policy HP-27 Do-not-resuscitate orders
A do-not-resuscitate (DNR) order is written by the attending physician after a discussion with the patient or their legal decision maker, and the discussion is documented in the health record. DNR applies only to cardiopulmonary resuscitation; it does not mean other treatments are withheld. A purple wristband identifies patients with a DNR order. DNR orders are reviewed whenever the patient's condition changes significantly and are suspended during procedures under anesthesia only after a documented discussion.
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = {
        "hvmc_formulary": render_formulary(),
        "hvmc_lab_catalog": render_labs(),
        "kestrel_iv220_manual": render_pump(),
        "kestrel_sp40_manual": render_sp40(),
        "kestrel_pca90_manual": render_pca90(),
        "hvmc_clinical_protocols": PROTOCOLS.format(disclaimer=DISCLAIMER) + PROTOCOLS_EXTRA,
        "hvmc_directory": render_directory(),
        "hvmc_iv_compatibility": render_compatibility(),
        "hvmc_antibiogram": render_antibiogram(),
        "talon_vm8_monitor_manual": render_monitor(),
        "hvmc_patient_leaflets": LEAFLETS.format(disclaimer=DISCLAIMER),
        "hvmc_policies": POLICIES.format(disclaimer=DISCLAIMER),
    }
    for doc_id, text in docs.items():
        path = OUT_DIR / f"{doc_id}.txt"
        path.write_text(text.strip() + "\n")
        print(f"wrote {path.relative_to(OUT_DIR.parents[1])}: {len(text.split()):,} words")


if __name__ == "__main__":
    main()
