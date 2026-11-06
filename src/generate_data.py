"""Generate a synthetic, privacy-safe insurance instruction dataset (JSONL).

Usage: python src/generate_data.py --n 3000 --out data/train.jsonl --seed 42
"""
import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

CITIES = ["Winnipeg", "Brandon", "Toronto", "Calgary", "Regina", "Vancouver", "Ottawa", "Halifax"]
STREETS = ["Elm St", "Main St", "Portage Ave", "Pembina Hwy", "Oak Dr", "King St", "Lakeshore Rd"]

SCENARIOS = {
    "property": [
        ("water_damage", "a pipe burst in the {room} overnight and water came through the ceiling"),
        ("fire", "a fire started in the {room} from a faulty appliance"),
        ("theft", "the home was broken into and electronics were stolen from the {room}"),
        ("wind", "a storm tore shingles off the roof and damaged the {room}"),
        ("flood", "the basement flooded after heavy rain and overland water entered"),
    ],
    "auto": [
        ("collision", "was rear-ended at a red light on {street}"),
        ("collision", "slid on ice and hit a pole on {street}"),
        ("theft", "found the vehicle stolen from the driveway"),
        ("glass", "had the windshield cracked by a rock on the highway"),
        ("hit_and_run", "was struck by another vehicle on {street} that fled the scene"),
    ],
}
ROOMS = ["kitchen", "basement", "living room", "bathroom", "garage"]

CLAUSES = {
    "water_damage": ("Section I.3: Sudden and accidental escape of water from plumbing is covered.", "covered"),
    "fire": ("Section I.1: Fire and lightning are covered perils.", "covered"),
    "theft": ("Section I.5: Theft following forcible entry is covered.", "covered"),
    "wind": ("Section I.2: Windstorm and hail are covered perils.", "covered"),
    "flood": ("Exclusion E.4: Overland water and flood are excluded unless the flood endorsement is purchased.", "not_covered"),
    "collision": ("Part C: Collision coverage applies to direct impact with another object.", "covered"),
    "glass": ("Part D: Comprehensive covers glass breakage, subject to deductible.", "covered"),
    "hit_and_run": ("Part E: Uninsured motorist coverage applies when the at-fault party is unidentified; report to police within 24 hours.", "needs_review"),
}


def rand_date(rng):
    return date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))


def make_claim(rng):
    lob = rng.choice(list(SCENARIOS))
    loss_type, tmpl = rng.choice(SCENARIOS[lob])
    city, street = rng.choice(CITIES), rng.choice(STREETS)
    d = rand_date(rng)
    injuries = lob == "auto" and rng.random() < 0.35
    third = loss_type in ("collision", "hit_and_run") and rng.random() < 0.7
    severity = rng.choice(["low", "medium", "high"]) if not injuries else "high"
    desc = tmpl.format(room=rng.choice(ROOMS), street=street)
    who = "Insured" if lob == "property" else "Driver"
    note = (
        f"{who} called on {(d + timedelta(days=rng.randint(0, 3))).isoformat()}. Reports that on {d.isoformat()} {desc}. "
        f"Location: {rng.randint(1, 999)} {street}, {city}. "
        + ("Insured reports neck and back pain, attended ER. " if injuries else "No injuries reported. ")
        + f"Estimated damage is {severity}."
    )
    return lob, loss_type, d, city, street, injuries, third, severity, note


def extraction_example(rng):
    lob, lt, d, city, street, inj, third, sev, note = make_claim(rng)
    loc = note.split("Location: ")[1].split(". ")[0]
    out = dict(line_of_business=lob, loss_type=lt, loss_date=d.isoformat(), location=loc,
               injuries=inj, severity=sev, third_party_involved=third,
               notes="Bodily injury reported." if inj else "Property damage only.")
    return {"task": "extract", "input": note, "output": json.dumps(out)}


def coverage_example(rng):
    lob, lt, *_rest, note = make_claim(rng)
    clause, decision = CLAUSES[lt]
    if lt == "flood" and rng.random() < 0.4:
        clause += " Policy schedule: Flood endorsement PURCHASED."
        decision = "covered"
    text = f"Policy clause: {clause}\nLoss scenario: {note}"
    expl = {
        "covered": f"The loss ({lt.replace('_', ' ')}) matches a covered peril in the cited clause.",
        "not_covered": f"The loss ({lt.replace('_', ' ')}) falls under an exclusion in the cited clause.",
        "needs_review": "Coverage depends on conditions (e.g. police report timing) that must be verified by an adjuster.",
    }[decision]
    return {"task": "coverage", "input": text,
            "output": json.dumps(dict(decision=decision, cited_clause=clause.split(":")[0], explanation=expl))}


def fraud_example(rng):
    *_rest, d, city, street, inj, third, sev, note = make_claim(rng)
    flags = []
    days_since_inception = rng.choice([5, 12, 25, 200, 400, 900])
    report_delay = rng.choice([0, 1, 2, 30, 45])
    prior_claims = rng.choice([0, 0, 1, 3, 4])
    cov_increase = rng.random() < 0.2
    if days_since_inception < 30:
        flags.append(f"Loss occurred {days_since_inception} days after policy inception")
    if report_delay > 14:
        flags.append(f"Loss reported {report_delay} days after occurrence")
    if prior_claims >= 3:
        flags.append(f"{prior_claims} prior claims in 3 years")
    if cov_increase:
        flags.append("Coverage limit increased within 60 days before loss")
    risk = "low" if not flags else ("medium" if len(flags) == 1 else "high")
    meta = (f"Policy metadata: days_since_inception={days_since_inception}, report_delay_days={report_delay}, "
            f"prior_claims_3y={prior_claims}, recent_coverage_increase={cov_increase}")
    just = "No material indicators found." if not flags else f"{len(flags)} indicator(s) present; refer to SIU if high."
    return {"task": "fraud", "input": f"{note}\n{meta}",
            "output": json.dumps(dict(risk=risk, red_flags=flags, justification=just))}


GENERATORS = [extraction_example, coverage_example, fraud_example]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.n):
            f.write(json.dumps(GENERATORS[i % 3](rng)) + "\n")
    print(f"Wrote {a.n} examples to {a.out}")


if __name__ == "__main__":
    main()
