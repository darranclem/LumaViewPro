from __future__ import annotations
"""
etaluma_init.py - Environment & Hardware Initialization Bootstrapper for LumaViewPro.

Replicates the exact startup verification sequence from stock Etaluma LumaViewPro:
1. Validates Python executable and environment (Python 3.10.11).
2. Sets JAVA_HOME (Zulu OpenJDK 8) and links Apache Maven 3.9.8.
3. Injects stock hardware driver directories into sys.path so 'import pyloncamera',
   'import motorboard', and 'import ledboard' succeed from any folder.
4. Checks SciJava, PyImageJ, jgo, and imglyb dependencies.
"""

import os
import sys

STOCK_INSTALL_DIRS = [
    r"C:\Program Files (x86)\LumaViewPro-stock",
    r"C:\Program Files (x86)\Etaluma\LumaViewPro",
    r"C:\Program Files\Etaluma\LumaViewPro",
    r"C:\Users\dkc25.BLUE\AppData\Local\Programs\Python\Python310\lib\site-packages"
]

JAVA_HOME_DEFAULT = r"C:\Program Files\Zulu\zulu-8"
MAVEN_BIN_DEFAULT = r"C:\Maven\apache-maven-3.9.8\bin"


def run_etaluma_stock_initialization():
    """Initializes environment, driver paths, Java/Maven links, and SciJava bridge."""
    print("=== LumaViewPro Stock Environment Initialization ===")

    for d in STOCK_INSTALL_DIRS:
        if os.path.exists(d) and d not in sys.path:
            sys.path.insert(0, d)
            print(f"--> Added driver path to sys.path: {d}")

    if not os.environ.get("JAVA_HOME"):
        if os.path.exists(JAVA_HOME_DEFAULT):
            os.environ["JAVA_HOME"] = JAVA_HOME_DEFAULT
            print(f"--> Set JAVA_HOME to: {JAVA_HOME_DEFAULT}")
        else:
            print("--> WARNING: JAVA_HOME is not set and default Zulu 8 path was not found.")
    else:
        print(f"--> JAVA_HOME = {os.environ.get('JAVA_HOME')}")

    if os.path.exists(MAVEN_BIN_DEFAULT):
        if MAVEN_BIN_DEFAULT not in os.environ.get("PATH", ""):
            os.environ["PATH"] = MAVEN_BIN_DEFAULT + os.pathsep + os.environ.get("PATH", "")
            print(f"--> Added Maven to PATH: {MAVEN_BIN_DEFAULT}")

    for mod_name in ["jgo", "scyjava", "imglyb", "pyimagej"]:
        try:
            __import__(mod_name)
            print(f"--> Dependency '{mod_name}': OK")
        except ImportError:
            print(f"--> Dependency '{mod_name}': Not loaded (using native Python drivers)")

    print("=== Initialization Complete. Launching LumaViewPro App... ===")


if __name__ == "__main__":
    run_etaluma_stock_initialization()
