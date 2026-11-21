"""The experiment wizard: time-lapse is a step of its own with a plain
description of what the numbers amount to, every step has a title, and the
layout file still parses with the step strip in it."""

from __future__ import annotations

import os
import re

import experiment_wizard as ew

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_time_lapse_is_a_step_of_its_own_between_z_stack_and_save():
    keys = [s[0] for s in ew.WIZARD_STEPS]
    assert keys[-3:] == ['acquire', 'timelapse', 'save']
    titles = {s[0]: s[1] for s in ew.WIZARD_STEPS}
    assert titles['timelapse'].startswith('Time-lapse')
    assert 'lapse' not in titles['acquire'].lower()
    for key, title, text in ew.WIZARD_STEPS:
        assert title and title[0].isupper() and len(text) > 40, key


def test_the_time_lapse_is_described_in_words():
    text = ew.ExperimentWizardMixin.timelapse_text
    assert 'imaged once' in text(False, 12, 5.0)
    assert 'imaged once' in text(True, 1, 5.0)
    said = text(True, 13, 10.0)
    assert '13 timepoints' in said and 'every 10 min' in said
    assert '2 h 0 min after the first' in said
    assert '55 min after the first' in text(True, 12, 5.0)
    assert 'no waiting' in text(True, 3, 0.0)


def test_the_layout_parses_and_has_a_button_for_every_step():
    from kivy.lang import Parser
    with open(os.path.join(HERE, 'lumaviewpro.kv'), encoding='utf-8') as f:
        kv = f.read()
    Parser(content=kv)                       # raises on a syntax error
    jumps = sorted(int(n) for n in re.findall(r'app\.wiz_jump\((\d+)\)', kv))
    assert jumps == list(range(len(ew.WIZARD_STEPS)))
    for key, _, _ in ew.WIZARD_STEPS:
        assert f"app.wiz_key == '{key}'" in kv, key
    opened = set(re.findall(r"app\.wiz_open_tool\('(\w+)'\)", kv))
    assert opened <= {s[0] for s in ew.WIZARD_STEPS}
    assert 'timelapse' in opened
