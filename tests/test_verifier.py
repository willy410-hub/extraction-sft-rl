import json

import pytest

from extraction_rl.generator import make_example, make_split
from extraction_rl.hacking import DEGENERATE_OUTPUTS, hacking_metrics
from extraction_rl.prompts import target_json
from extraction_rl.schema import FIELDS
from extraction_rl.verifier import exact_match, parse_output, verify

TRUTH = {
    "invoice_id": "INV-12345", "date": "2026-10-28", "amount": 1250.5,
    "currency": "EGP", "customer_id": None, "customer_name": "محمد علي",
}


def _out(**over):
    d = dict(TRUTH)
    d.update(over)
    return json.dumps(d, ensure_ascii=False)


def test_perfect_answer_scores_one():
    r = verify(_out(), TRUTH)
    assert r["score"] == 1.0 and exact_match(r)


def test_canonical_target_scores_one_on_generated_data():
    for ex in make_split("test", 200):
        assert verify(target_json(ex["truth"]), ex["truth"])["score"] == 1.0


def test_invalid_json_scores_zero():
    r = verify("not json {", TRUTH)
    assert r == {"valid_json": False, "schema_ok": False, "score": 0.0, "per_field": None, "pred": None}


def test_code_fence_fails_strict_but_passes_lenient():
    fenced = "```json\n" + _out() + "\n```"
    assert verify(fenced, TRUTH)["score"] == 0.0
    assert verify(fenced, TRUTH, lenient=True)["score"] == 1.0


def test_missing_key_fails_schema():
    d = json.loads(_out())
    del d["currency"]
    r = verify(json.dumps(d), TRUTH)
    assert r["valid_json"] and not r["schema_ok"] and r["score"] == 0.0


def test_extra_key_fails_schema():
    d = json.loads(_out())
    d["notes"] = "x"
    assert verify(json.dumps(d), TRUTH)["schema_ok"] is False


def test_wrong_type_fails_schema():
    assert verify(_out(amount="1250.50"), TRUTH)["schema_ok"] is False
    assert verify(_out(amount=True), TRUTH)["schema_ok"] is False


def test_partial_credit_per_field():
    r = verify(_out(date="28/10/2026", currency="EUR"), TRUTH)
    assert r["score"] == pytest.approx(4 / 6)
    assert r["per_field"]["date"] is False and r["per_field"]["invoice_id"] is True


def test_amount_tolerance():
    assert verify(_out(amount=1250.50), TRUTH)["per_field"]["amount"]
    assert verify(_out(amount=1250), TRUTH)["per_field"]["amount"] is False


def test_inventing_a_missing_field_is_wrong():
    r = verify(_out(customer_id="C-9999"), TRUTH)
    assert r["per_field"]["customer_id"] is False


def test_null_for_present_field_is_wrong():
    assert verify(_out(invoice_id=None), TRUTH)["per_field"]["invoice_id"] is False


def test_unicode_normalization_nfd_equals_nfc():
    import unicodedata
    truth = dict(TRUTH, customer_name="Zoë Müller")
    nfd = unicodedata.normalize("NFD", "Zoë Müller")
    assert verify(_out(customer_name=nfd), truth)["per_field"]["customer_name"]


@pytest.mark.parametrize("probe", ["all_null", "empty_strings", "empty_object"])
def test_degenerate_outputs_score_low_on_every_slice(probe):
    """Reward-hacking probe: constant outputs must never look good."""
    for ex in make_split("test", 400):
        assert verify(DEGENERATE_OUTPUTS[probe], ex["truth"])["score"] <= 1 / 6


def test_echoing_the_input_scores_zero():
    ex = make_example(30_001)
    assert verify(ex["text"], ex["truth"])["score"] == 0.0


def test_hacking_metrics_detect_null_abstention():
    exs = make_split("test", 40)
    res = [verify(DEGENERATE_OUTPUTS["all_null"], ex["truth"]) for ex in exs]
    m = hacking_metrics(exs, res)
    assert m["null_on_present_rate"] == 1.0
    assert m["invented_on_missing_rate"] == 0.0


def test_hacking_metrics_detect_invented_ids():
    exs = make_split("test", 40)
    res = [verify(json.dumps(dict(ex["truth"], invoice_id="INV-00000",
                                  customer_id="C-0000")), ex["truth"]) for ex in exs]
    m = hacking_metrics(exs, res)
    assert m["id_grounding_rate"] == 0.0
    assert m["invented_on_missing_rate"] > 0


def test_parse_output_strict_rejects_trailing_prose():
    assert parse_output(_out() + "\nHope this helps!") is None


def test_schema_fields_order_independent():
    d = json.loads(_out())
    reordered = {k: d[k] for k in reversed(FIELDS)}
    assert verify(json.dumps(reordered, ensure_ascii=False), TRUTH)["score"] == 1.0


def test_penalized_reward_charges_extra_for_invented_values():
    from extraction_rl.verifier import penalized_reward

    assert penalized_reward(_out(), TRUTH, 2.0) == 1.0
    invented = _out(customer_id="C-0001")
    assert verify(invented, TRUTH)["score"] == pytest.approx(5 / 6)
    assert penalized_reward(invented, TRUTH, 2.0) == pytest.approx(5 / 6 - 2 / 6)
    # null on a PRESENT field is not penalized extra (only the normal 1/6)
    assert penalized_reward(_out(invoice_id=None), TRUTH, 2.0) == pytest.approx(5 / 6)
