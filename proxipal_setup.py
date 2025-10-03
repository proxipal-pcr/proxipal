# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 18:00:24 2025

@author: smith.j
"""

#!/usr/bin/env python3
"""
ProxiPal Setup Script
- Prepares app/ workspace directory structure
- Copies developer/src/ProxiPal.py into app/python/ (if missing)
- Detects dependencies from the copied script
- Advises user on missing packages
"""

import os
import sys
import importlib
import ast
from pathlib import Path
import shutil

# ------------------------------
# Directory Structure
# ------------------------------
EXPECTED_DIRS = [
    "data",
    "exports",
    "python",
    "quality",
    "samples",
    "templates"
]

def create_structure(base_path: Path) -> Path:
    """
    Creates the required workspace directory structure
    inside 'app/' folder without overwriting or deleting existing data.
    Returns path to workspace ProxiPal.py
    """
    app_path = base_path / "app"
    print(f"\n🧱 Ensuring ProxiPal workspace at: {app_path.resolve()}")
    created, existing = [], []
    for folder in EXPECTED_DIRS:
        folder_path = app_path / folder
        if folder_path.exists():
            existing.append(folder)
        else:
            folder_path.mkdir(parents=True, exist_ok=True)
            created.append(folder)
    if created:
        print(f"✅ Created: {', '.join(created)}")
    if existing:
        print(f"ℹ️ Already present: {', '.join(existing)}")
    print("✨ Workspace is ready.")
    return app_path / "python" / "ProxiPal.py"

# ------------------------------
# Utilities: Import Detection
# ------------------------------
def get_imports_from_file(file_path: Path):
    """
    Parse a Python file and extract top-level imports.
    """
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
    """
    Filters imports to only include non-standard library modules.
    """
    stdlib = sys.stdlib_module_names if hasattr(sys, "stdlib_module_names") else set()
    imports = get_imports_from_file(file_path)
    external = sorted([i for i in imports if i not in stdlib])
    return external

# ------------------------------
# Dependency Checker
# ------------------------------
def check_dependencies(packages):
    """
    Checks whether required packages are installed.
    Advises user on how to install missing ones.
    """
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
# Main
# ------------------------------
def main():
    print("🚀 ProxiPal Setup\n")

    # Step 1: Ask user where to create workspace
    cwd = Path.cwd()
    answer = input(f"📁 Create or verify workspace here? ({cwd}) [Y/n]: ").strip().lower()
    base_path = cwd if answer in ("", "y", "yes") else Path(input("Enter target path: ").strip()).expanduser().resolve()

    # Step 2: Create directory structure first (inside 'app/')
    proxipal_file = create_structure(base_path)

    # Step 3: Copy ProxiPal.py from developer/src/ if needed
    repo_src = Path(__file__).parent / "developer" / "src" / "ProxiPal.py"
    if not proxipal_file.exists():
        if repo_src.exists():
            print(f"\n📄 Copying {repo_src.name} from {repo_src.parent} to {proxipal_file.parent}")
            shutil.copy2(repo_src, proxipal_file)
            print("✅ ProxiPal.py copied to workspace.")
        else:
            print(f"\n⚠️ No source file found at {repo_src}")
    else:
        print(f"\nℹ️ {proxipal_file.name} already exists in workspace; leaving it unchanged.")

    # Step 4: Detect dependencies
    if proxipal_file.exists():
        print(f"\n📄 Inspecting imports from {proxipal_file}")
        required_packages = detect_external_imports(proxipal_file)
        print(f"🧩 Detected external imports: {', '.join(required_packages)}")
    else:
        print(f"\n⚠️ Still no {proxipal_file.name}; using default dependency list.")
        required_packages = [
            "pandas",
            "numpy",
            "scipy",
            "scikit-learn",
            "pymannkendall",
            "outliers",
            "plotly",
            "matplotlib",
            "seaborn",
            "rdmlpython"
        ]

    # Step 5: Check package availability
    check_dependencies(required_packages)

    # Step 6: Print next steps
    print("✅ Setup complete.")
    print("➡️ Next steps:")
    print(f"   1. Navigate to: {proxipal_file.parent}")
    print(f"   2. Run with:    python {proxipal_file.name}")

# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    main()
