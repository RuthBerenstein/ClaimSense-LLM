import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from generate_data import GENERATORS  # noqa: E402


def test_outputs_are_valid_json():
    rng = random.Random(0)
    for i in range(60):
        ex = GENERATORS[i % 3](rng)
        assert ex["task"] in {"extract", "coverage", "fraud"}
        json.loads(ex["output"])


def test_flood_without_endorsement_not_covered():
    rng = random.Random(1)
    for _ in range(300):
        ex = GENERATORS[1](rng)
        if "Exclusion E.4" in ex["input"] and "PURCHASED" not in ex["input"]:
            assert json.loads(ex["output"])["decision"] == "not_covered"
