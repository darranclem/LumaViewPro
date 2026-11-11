"""Custom labware: validation, naming, rename and persistence (labware_defs)."""

from __future__ import annotations

import json

import pytest

import labware_defs as lwd


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setattr(lwd, 'CUSTOM_LABWARE_PATH',
                        str(tmp_path / 'custom_labware.json'))
    return lwd.LabwareLibrary()


def _plate(**kw):
    base = dict(name='Mine', rows=8, cols=12, spacing_x_mm=9.0,
                spacing_y_mm=9.0, well_size_x_mm=6.4, well_size_y_mm=6.4)
    base.update(kw)
    return lwd.LabwareDef(**base)


def test_every_builtin_is_valid():
    for d in lwd.builtin_labware().values():
        assert lwd.validate_labware(d) == ([], []), d.name


@pytest.mark.parametrize('kw,needle', [
    (dict(rows=0), 'at least 1'),
    (dict(well_size_x_mm=0), 'Well size X'),
    (dict(spacing_x_mm=0), 'Spacing'),
    (dict(cols=14), 'past the plate edge'),
    (dict(a1_offset_x_mm=2.0), 'past the plate edge'),
    (dict(name='  '), 'Name is empty'),
])
def test_invalid_designs_are_refused(kw, needle):
    errors, _ = lwd.validate_labware(_plate(**kw))
    assert any(needle in e for e in errors), errors


def test_overlapping_wells_are_only_a_warning():
    errors, warnings = lwd.validate_labware(_plate(well_size_x_mm=9.5,
                                                   well_size_y_mm=9.5))
    assert not errors and warnings


def test_a_single_well_needs_no_spacing():
    d = _plate(rows=1, cols=1, spacing_x_mm=0, spacing_y_mm=0,
               well_size_x_mm=22, well_size_y_mm=60,
               a1_offset_x_mm=63.88, a1_offset_y_mm=42.74)
    assert lwd.validate_labware(d)[0] == []


def test_centering_puts_the_grid_in_the_middle():
    ax, ay = lwd.centered_a1_offsets(8, 12, 9.0, 9.0, 127.76, 85.48)
    assert ax == pytest.approx(14.38) and ay == pytest.approx(11.24)


def test_names_cannot_shadow_builtins_or_other_customs(lib):
    assert 'built-in' in lib.check_name('96-well plate')
    lib.save(_plate(name='A'))
    assert 'already exists' in lib.check_name('A')
    assert lib.check_name('A', replacing='A') is None
    with pytest.raises(ValueError):
        lib.save(_plate(name='96-well plate'))


def test_renaming_a_custom_replaces_it(lib):
    lib.save(_plate(name='Old'))
    lib.save(_plate(name='New', rows=4), replacing='Old')
    assert lib.get('Old') is None and lib.get('New').rows == 4
    assert [d['name'] for d in json.load(open(lwd.CUSTOM_LABWARE_PATH))] \
        == ['New']


def test_customs_persist_and_a_bad_entry_does_not_hide_the_rest(lib):
    lib.save(_plate(name='Good'))
    raw = json.load(open(lwd.CUSTOM_LABWARE_PATH))
    raw.append({'name': 'Broken', 'rows': 'x'})          # missing fields
    raw.append(dict(raw[0], name='Future', new_field=1))  # unknown key
    json.dump(raw, open(lwd.CUSTOM_LABWARE_PATH, 'w'))
    again = lwd.LabwareLibrary()
    assert again.get('Good') is not None and again.get('Future') is not None
    assert again.get('Broken') is None
    assert again.names()[:len(again.builtins)] == list(again.builtins)


def test_a_definition_round_trips_through_a_dict():
    d = _plate(shape='square')
    back = lwd.labware_from_dict(lwd.labware_to_dict(d))
    assert back == d and back.builtin is False


def test_unique_name(lib):
    assert lib.unique_name('My labware') == 'My labware'
    lib.save(_plate(name='My labware'))
    assert lib.unique_name('My labware') == 'My labware 2'
