"""M1 regression case: a ticket must not override the order's delay value."""

import pytest

pytestmark = pytest.mark.student


def test_ticket_override_is_first_bad_event_and_fixed_shape_passes():
    from lab9.trace_case import investigate

    regressed = investigate("seeded_regression", "C-002", 11)["trial"]
    fixed = investigate("fixed", "C-002", 11)["trial"]

    assert regressed["lab8_pass"] is False
    assert regressed["snapshot"]["refund_vnd"] == 20000
    assert regressed["events"][2]["kind"] == "model_override"
    assert fixed["lab8_pass"] is True
    assert fixed["snapshot"]["refund_vnd"] == 0
    assert all(event["kind"] != "model_override" for event in fixed["events"])
