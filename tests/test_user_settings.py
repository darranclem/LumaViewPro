"""Per-account settings (acquisition.save_settings / load_settings)."""

from __future__ import annotations

import json
import os

import pytest

import acquisition as acq


@pytest.fixture
def paths(tmp_path, monkeypatch):
    shared = tmp_path / 'program' / 'data' / 'lvp_custom_settings.json'
    user = tmp_path / 'profile' / 'LumaViewPro' / 'settings.json'
    shared.parent.mkdir(parents=True)
    monkeypatch.setattr(acq, 'SHARED_SETTINGS_PATH', str(shared))
    monkeypatch.setattr(acq, 'SETTINGS_PATH', str(user))
    return shared, user


def test_settings_are_written_to_the_users_own_file(paths):
    shared, user = paths
    acq.save_settings({'protocol_dir': 'D:\\mine', 'output_dir': 'Y:\\data'})
    assert not shared.exists()
    assert json.loads(user.read_text())['protocol_dir'] == 'D:\\mine'
    assert acq.load_settings() == {'protocol_dir': 'D:\\mine',
                                   'output_dir': 'Y:\\data'}
    assert not os.path.exists(str(user) + '.tmp')


def test_a_users_file_wins_over_the_one_with_the_program(paths):
    shared, user = paths
    shared.write_text(json.dumps({'protocol_dir': 'C:\\program\\protocols'}))
    acq.save_settings({'protocol_dir': 'D:\\mine'})
    assert acq.load_settings()['protocol_dir'] == 'D:\\mine'


def test_the_owner_of_the_program_folder_keeps_the_old_settings(paths):
    shared, _ = paths
    shared.write_text(json.dumps({'objective_name': '10x Oly',
                                  'output_dir': 'Y:\\data',
                                  'protocol_dir': 'C:\\p'}))
    assert acq.load_settings() == {'objective_name': '10x Oly',
                                   'output_dir': 'Y:\\data',
                                   'protocol_dir': 'C:\\p'}


def test_another_account_starts_from_the_instrument_settings_without_folders(
        paths, monkeypatch):
    shared, _ = paths
    shared.write_text(json.dumps({'objective_name': '10x Oly',
                                  'output_dir': 'Y:\\Imaging\\Darran',
                                  'protocol_dir': 'C:\\p'}))
    monkeypatch.setattr(acq, 'dir_writable', lambda p: False)
    assert acq.load_settings() == {'objective_name': '10x Oly'}


def test_a_broken_user_file_falls_back_instead_of_failing(paths):
    shared, user = paths
    shared.write_text(json.dumps({'objective_name': '4x Oly'}))
    user.parent.mkdir(parents=True)
    user.write_text('{ not json')
    assert acq.load_settings() == {'objective_name': '4x Oly'}


def test_no_settings_anywhere_gives_defaults(paths):
    assert acq.load_settings() == {}


def test_dir_writable(tmp_path):
    assert acq.dir_writable(str(tmp_path / 'new' / 'folder'))
    assert os.listdir(tmp_path / 'new' / 'folder') == []
    blocked = tmp_path / 'file.txt'
    blocked.write_text('x')
    assert not acq.dir_writable(str(blocked / 'under_a_file'))
