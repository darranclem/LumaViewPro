"""
plate_navigator.py - Modular Plate Navigator & Registration Studio Extension for LumaViewPro.
"""

from __future__ import annotations
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import math
import re
import json
from typing import Dict, Optional, List, Tuple

import image_utils
import labware

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button

_plate_navigator_kv_loaded = False
_kv_file = os.path.join(script_dir, 'plate_navigator.kv')
if os.path.exists(_kv_file) and _kv_file not in Builder.files:
    try:
        Builder.load_file(_kv_file)
        _plate_navigator_kv_loaded = True
    except Exception as e:
        print(f"[plate_navigator.py] KV Load note: {e}")


def attach_navigator_to_lumaviewpro(app_instance):
    if not hasattr(app_instance, 'well_plate_format'):
        app_instance.well_plate_format = labware.WellPlateFormat(num_wells=96)
    if not hasattr(app_instance, 'plate_calibration'):
        app_instance.plate_calibration = labware.NavigatorPlateCalibration(app_instance.well_plate_format)
    if not hasattr(app_instance, 'focus_map'):
        app_instance.focus_map = labware.FocusMap()
    if not hasattr(app_instance, 'fov_manager'):
        app_instance.fov_manager = labware.PlateFOVManager(
            app_instance.well_plate_format,
            app_instance.plate_calibration,
            app_instance.focus_map
        )

    if hasattr(app_instance.root_layout, 'ids') and 'scaled_canvas_tab1' in app_instance.root_layout.ids:
        canvas_widget = app_instance.root_layout.ids.scaled_canvas_tab1
        canvas_widget.set_references(
            app_instance.well_plate_format,
            app_instance.plate_calibration,
            app_instance.focus_map,
            app_instance.fov_manager
        )
    return None
