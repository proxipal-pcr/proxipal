# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 18:00:24 2025

@author: smith.j
"""

#!/usr/bin/env python3
"""
ProxiPal Setup Script
- Checks imports from ProxiPal.py
- Verifies non-standard dependencies
- Creates the expected directory structure (safe)
"""

import os
import sys
import importlib
import ast
from pathlib import Path

# ------------------------------
# Utility: parse imports
# ------------------------------
def get_imports_from_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports

def detect_external_imports(file_path: Path):
    stdlib = sys.stdlib_module_names if hasattr(sys, "stdlib_module_names") else set()
    imports = get_imports_from_file(file_path)
    external = sorted([i for i in imports if i not in stdlib])
    return external

# ------------------------------
# Dependency Checker
# ------------------------------
def check_dependencies(packages):
    print("\n🔍 Checking dependencies...\n")
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"✅ {pkg} is installed")
        except ImportError:
            print(f"❌ {pkg} is missing")
            missing.append(pkg)
    if missing:
        print("\n⚠️ Missing dependencies detected:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        print("\n💡 Please install these before running ProxiPal.\n")
    else:
        print("\n🎉 All dependencies are installed!\n")

# ------------------------------
# Directory Structure Creator
# ------------------------------
EXPECTED_DIRS = ["data", "exports", "python", "quality", "samples", "templates"]

def create_structure(base_path: Path):
    print(f"\n🧱 Ensuring ProxiPal directory structure at: {base_path.resolve()}")
    created, existing = [], []
    for folder in EXPECTED_DIRS:
        folder_path = base_path / folder
        if folder_path.exists():
            existing.append(folder)
        else:
            folder_path.mkdir(parents=True, exist_ok=True)
            created.append(folder)
    if created:
        print(f"✅ Created: {', '.join(created)}")
    if existing:
        print(f"ℹ️ Already present: {', '.join(existing)}")
    print("\n✨ Directory structure is ready.")

# ------------------------------
# Main
# ------------------------------
def main():
    print("🚀 ProxiPal Setup\n")

    proxipal_path = Path("python") / "ProxiPal.py"
    if not proxipal_path.exists():
        print("⚠️ Could not find python/ProxiPal.py — using default dependency list.")
        required_packages = ["pandas", "numpy", "scipy", "scikit-learn",
                             "pymannkendall", "outliers", "plotly", "matplotlib", "seaborn", "rdmlpython"]
    else:
        print(f"📄 Inspecting imports from {proxipal_path}")
        required_packages = detect_external_imports(proxipal_path)
        print(f"🧩 Detected external imports: {', '.join(required_packages)}")

    check_dependencies(required_packages)

    cwd = Path.cwd()
    answer = input(f"\n📁 Set up directory structure here? ({cwd}) [Y/n]: ").strip().lower()
    base_path = cwd if answer in ("", "y", "yes") else Path(input("Enter target path: ").strip())
    create_structure(base_path)

if __name__ == "__main__":
    main()
