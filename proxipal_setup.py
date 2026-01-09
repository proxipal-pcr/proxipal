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
- Copies developer/templates/* into app/templates/ (updates existing)
- Detects dependencies from the copied script
- Advises user on missing packages

Added:
- Prompts for research/dev when interactive (TTY)
- Defaults to research when non-interactive
- Research mode disables git push in this clone (pushurl + pre-push hook)
"""

import os
import sys
import importlib
import ast
from pathlib import Path
import shutil
import subprocess

# ------------------------------
# Git Environment Mode
# ------------------------------
PRE_PUSH_HOOK = """#!/bin/sh
echo "ERROR: Push disabled in this clone (research environment)."
exit 1
"""
HOOK_SENTINEL = "Push disabled in this clone (research environment)."


def _run(cmd, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), check=check, text=True, capture_output=True)


def _interactive_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _find_repo_root(start: Path) -> Path | None:
    """
    Walk upwards from start to find a .git directory or git worktree.
    Returns repo root path or None.
    """
    cur = start.resolve()
    for _ in range(50):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _is_git_repo(repo_dir: Path) -> bool:
    p = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_dir, check=False)
    return p.returncode == 0 and p.stdout.strip() == "true"


def _get_git_dir(repo_dir: Path) -> Path:
    """
    Supports normal repos and worktrees: `git rev-parse --git-dir` may be relative.
    """
    p = _run(["git", "rev-parse", "--git-dir"], cwd=repo_dir, check=True)
    git_dir = p.stdout.strip()
    gd = Path(git_dir)
    return (repo_dir / gd).resolve() if not gd.is_absolute() else gd


def _prompt_env() -> str:
    print(
        "\nEnvironment selection:\n"
        "  research    - Git push disabled. Safe for analysis, shared machines, and user data.\n"
        "  development - Git push enabled. Use only if you intend to modify and publish code.\n"
    )
    while True:
        val = input("Select environment [research/dev]: ").strip().lower()
        if val in ("research", "dev"):
            return val
        print("Invalid value. Type: research or dev.")


def _set_research_mode(repo_root: Path):
    # Blocks `git push origin ...`
    _run(["git", "config", "--local", "remote.origin.pushurl", "DISABLED"], cwd=repo_root, check=True)

    # Blocks `git push https://...` too
    git_dir = _get_git_dir(repo_root)
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "pre-push"
    hook_path.write_text(PRE_PUSH_HOOK, encoding="utf-8")
    hook_path.chmod(0o755)


def _set_dev_mode(repo_root: Path):
    # Re-enable push by removing override
    _run(["git", "config", "--local", "--unset-all", "remote.origin.pushurl"], cwd=repo_root, check=False)

    # Remove proxipal pre-push hook only if it matches sentinel
    git_dir = _get_git_dir(repo_root)
    hook_path = git_dir / "hooks" / "pre-push"
    if hook_path.exists():
        try:
            txt = hook_path.read_text(encoding="utf-8", errors="ignore")
            if HOOK_SENTINEL in txt:
                hook_path.unlink()
        except OSError:
            pass


# def configure_git_mode():
#     """
#     If this script is run inside a git clone:
#       - interactive: prompt research/dev
#       - non-interactive: default to research
#     If not in a git repo: no-op.
#     """
#     repo_root = _find_repo_root(Path(__file__).parent)
#     if not repo_root:
#         return

#     if not _is_git_repo(repo_root):
#         return

#     env = _prompt_env() if _interactive_tty() else "research"

#     if env == "research":
#         _set_research_mode(repo_root)
#         print("Configured: research mode (git push disabled in this clone).")
#     else:
#         _set_dev_mode(repo_root)
#         print("Configured: development mode (git push enabled in this clone).")

def configure_git_mode() -> str:
    """
    If this script is run inside a git clone:
      - interactive: prompt research/dev
      - non-interactive: default to research
    If not in a git repo: returns "research" and does nothing else.
    Returns the selected environment: "research" or "dev".
    """
    repo_root = _find_repo_root(Path(__file__).parent)
    if not repo_root:
        return "research"

    if not _is_git_repo(repo_root):
        return "research"

    env = _prompt_env() if _interactive_tty() else "research"

    if env == "research":
        _set_research_mode(repo_root)
        print("Configured: research mode (git push disabled in this clone).")
    else:
        _set_dev_mode(repo_root)
        print("Configured: development mode (git push enabled in this clone).")

    return env



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
# Utilities
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


def copy_templates(src_dir: Path, dest_dir: Path):
    """
    Copies all files from developer/templates/ into app/templates/.
    Existing files are overwritten with latest versions.
    """
    if not src_dir.exists():
        print(f"⚠️ No template source found at {src_dir}")
        return
    print(f"\n📂 Copying templates from {src_dir} to {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dest_dir / item.name)
            print(f"   📄 {item.name}")
    print("✅ Templates copied successfully.")


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


# # ------------------------------
# # Main
# # ------------------------------
# def main():
#     print("🚀 ProxiPal Setup\n")

#     # Step 0: Configure git mode (research/dev)
#     # Interactive: prompt. Non-interactive: defaults to research.
#     configure_git_mode()

#     # Step 1: Ask user where to create workspace
#     cwd = Path.cwd()
#     answer = input(f"📁 Create or verify workspace here? ({cwd}) [Y/n]: ").strip().lower()
#     base_path = cwd if answer in ("", "y", "yes") else Path(input("Enter target path: ").strip()).expanduser().resolve()

#     # Step 2: Create directory structure (inside 'app/')
#     proxipal_file = create_structure(base_path)
#     app_path = base_path / "app"

#     # Step 3: Copy ProxiPal.py from developer/src/ if needed
#     repo_src = Path(__file__).parent / "developer" / "src" / "ProxiPal.py"
#     if not proxipal_file.exists():
#         if repo_src.exists():
#             print(f"\n📄 Copying {repo_src.name} from {repo_src.parent} to {proxipal_file.parent}")
#             shutil.copy2(repo_src, proxipal_file)
#             print("✅ ProxiPal.py copied to workspace.")
#         else:
#             print(f"\n⚠️ No source file found at {repo_src}")
#     else:
#         print(f"\nℹ️ {proxipal_file.name} already exists in workspace; leaving it unchanged.")

#     # Step 4: Copy templates
#     templates_src = Path(__file__).parent / "developer" / "templates"
#     templates_dest = app_path / "templates"
#     copy_templates(templates_src, templates_dest)

#     # Step 5: Detect dependencies
#     if proxipal_file.exists():
#         print(f"\n📄 Inspecting imports from {proxipal_file}")
#         required_packages = detect_external_imports(proxipal_file)
#         print(f"🧩 Detected external imports: {', '.join(required_packages)}")
#     else:
#         print(f"\n⚠️ Still no {proxipal_file.name}; using default dependency list.")
#         required_packages = [
#             "pandas",
#             "numpy",
#             "scipy",
#             "scikit-learn",
#             "pymannkendall",
#             "outliers",
#             "plotly",
#             "matplotlib",
#             "seaborn",
#             "rdmlpython"
#         ]

#     # Step 6: Check package availability
#     check_dependencies(required_packages)

#     # Step 7: Print next steps
#     print("✅ Setup complete.")
#     print("➡️ Next steps:")
#     print(f"   1. Navigate to: {proxipal_file.parent}")
#     print(f"   2. Run with:    python {proxipal_file.name}")

# ------------------------------
# Main
# ------------------------------
def main():
    print("🚀 ProxiPal Setup\n")

    # Step 0: Configure git mode (research/dev)
    # Interactive: prompt. Non-interactive: defaults to research.
    env = configure_git_mode()

    # Step 1: Ask user where to create workspace
    # Research default: keep workspace out of the repo by default.
    # Dev default: current working directory.
    default_base = (Path.home() / "ProxiPal_Workspace") if env == "research" else Path.cwd()
    answer = input(f"📁 Create or verify workspace here? ({default_base}) [Y/n]: ").strip().lower()
    base_path = default_base if answer in ("", "y", "yes") else Path(input("Enter target path: ").strip()).expanduser().resolve()

    # Step 2: Create directory structure (inside 'app/')
    proxipal_file = create_structure(base_path)
    app_path = base_path / "app"

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

    # Step 4: Copy templates
    templates_src = Path(__file__).parent / "developer" / "templates"
    templates_dest = app_path / "templates"
    copy_templates(templates_src, templates_dest)

    # Step 5: Detect dependencies
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

    # Step 6: Check package availability
    check_dependencies(required_packages)

    # Step 7: Print next steps
    print("✅ Setup complete.")
    print("➡️ Next steps:")
    print(f"   1. Navigate to: {proxipal_file.parent}")
    print(f"   2. Run with:    python {proxipal_file.name}")



# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    main()
