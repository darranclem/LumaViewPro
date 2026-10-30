from __future__ import annotations

import os
import sys
import enum

# Polyfill StrEnum for Python 3.10 compatibility
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        """Polyfill for Python 3.10 compatibility where enum.StrEnum is missing."""
        def __str__(self):
            return str(self.value)
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()
    enum.StrEnum = StrEnum

# Standard default paths on Windows microscope installation
STOCK_INSTALL_DIRS = [
    r"C:\Program Files (x86)\LumaViewPro-stock",
    r"C:\Program Files (x86)\Etaluma\LumaViewPro",
    r"C:\Program Files\Etaluma\LumaViewPro",
    r"C:\Program Files (x86)\LumaViewPro-main",
    r"C:\Users\dkc25.BLUE\AppData\Local\Programs\Python\Python310\lib\site-packages"
]

JAVA_HOME_DEFAULT = r"C:\Program Files\Zulu\zulu-8"
MAVEN_BIN_DEFAULT = r"C:\Maven\apache-maven-3.9.8\bin"


def run_etaluma_stock_initialization():
    """Initializes environment, driver paths, Java/Maven links, and SciJava bridge."""
    print("=== LumaViewPro Stock Environment Initialization ===")

    # 1. Inject stock driver paths into sys.path
    for d in STOCK_INSTALL_DIRS:
        if os.path.exists(d) and d not in sys.path:
            sys.path.insert(0, d)
            print(f"--> Added driver path to sys.path: {d}")

    # 2. Configure JAVA_HOME if not assigned
    if not os.environ.get("JAVA_HOME"):
        if os.path.exists(JAVA_HOME_DEFAULT):
            os.environ["JAVA_HOME"] = JAVA_HOME_DEFAULT
            print(f"--> Set JAVA_HOME to: {JAVA_HOME_DEFAULT}")
        else:
            print("--> WARNING: JAVA_HOME is not set and default Zulu 8 path was not found.")
    else:
        print(f"--> JAVA_HOME = {os.environ.get('JAVA_HOME')}")

    # 3. Add Maven bin to System Path if present
    if os.path.exists(MAVEN_BIN_DEFAULT):
        if MAVEN_BIN_DEFAULT not in os.environ.get("PATH", ""):
            os.environ["PATH"] = MAVEN_BIN_DEFAULT + os.pathsep + os.environ.get("PATH", "")
            print(f"--> Added Maven to PATH: {MAVEN_BIN_DEFAULT}")

    # 4. Check SciJava / PyImageJ / jgo dependencies
    for mod_name in ["jgo", "scyjava", "imglyb", "pyimagej"]:
        try:
            __import__(mod_name)
            print(f"--> Dependency '{mod_name}': OK")
        except ImportError:
            print(f"--> Dependency '{mod_name}': Not loaded (using native Python drivers)")

    print("=== Initialization Complete. Launching LumaViewPro App... ===")


# Run on module import so sys.path and StrEnum polyfill are ready immediately
run_etaluma_stock_initialization()
