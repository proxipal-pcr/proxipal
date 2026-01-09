# 🚀 Getting Started with ProxiPal

ProxiPal separates **research execution** from **code development**.  
This guide explains how to set up your environment safely and how to choose the correct operating mode.

---

## 📦 1. Prerequisites

Before you begin, make sure you have:

- **Python 3.9+**
- **pip**
- Recommended: a **virtual environment** (e.g. `venv` or `conda`)

Verify your installation:

```bash
python --version
pip --version
```

---

## 🧰 2. Clone the Repository

Clone the repository to any location:

```bash
git clone https://github.com/proxipal-pcr/proxipal.git
cd proxipal
```

After cloning, the repository contains:

```
proxipal/
├── developer/          # Versioned source code and templates (Git-tracked)
├── proxipal_setup.py   # Setup script
├── README.md
└── LICENSE
```

The presence of `developer/` is normal in **all clones**.

---

## ⚙️ 3. Run the Setup Script (Required)

Run the setup script:

```bash
python proxipal_setup.py
```

### Environment selection

When run interactively, the script will prompt you to select an environment:

```
research    - Git push disabled. Safe for analysis and real data.
development - Git push enabled. Intended for publishing code or notebooks.
```

If the script is run **non-interactively**, it defaults to **research mode**.

---

## 🔐 4. Research vs Development Mode (Important)

ProxiPal is designed to support collaboration between **coding and non-coding users** by operating on a **shared, predictable directory structure**.  
Any directory created by the setup script may contain arbitrary file types; ProxiPal will only read and act on files relevant to its functionality.

This structure has two important implications:

- Administrators can apply **OS-level permissions** (by user or group) to sensitive subfolders (for example, samples or restricted data).
- Users can run `git clone` and re-run the setup script against **existing, populated directories** to stay up to date with the code without disturbing their data.

Because the repository includes **demo data and sample folders** (e.g. under `data/` and `samples/`), cloning the repository may reset folder permissions.  
If permissions are important in your environment, they should be reinstated after cloning.

The **mode you select during setup** determines how safe the clone is with respect to publishing content to GitHub and is therefore critical.

---

### Research mode (default, recommended)

Use **research mode** when:

- Working with real or populated data directories
- Running analyses, notebooks, or pipelines
- Operating on shared machines or production data

Behavior:

- `git push` is **disabled** in this clone
- Push is blocked even if an explicit GitHub URL is used
- Workspace defaults **outside the repository** (e.g. `~/ProxiPal_Workspace`)
- Prevents accidental publication of data, notebooks, or analysis artifacts

This mode is intended for day-to-day analytical work and is safe by default.

---

### Development mode

Use **development mode** only when you intend to:

- Modify ProxiPal source code
- Add curated notebooks or code intended for the repository
- Commit and push changes upstream

Behavior:

- `git push` is enabled
- Normal Git workflows apply

⚠️ **Important warning**

Do **not** switch a populated research clone into development mode.

If a notebook or code derived from real data should be published:

**Recommended workflow**
1. Clone the repository into a **new, clean location**
2. Run the setup script and select **development**
3. Add only the intended files
4. Commit and push from that clean clone

This ensures that only approved code and artifacts enter GitHub, while real data remains local and protected.

---

## 📁 5. Workspace Layout

The setup script creates a runtime workspace under `app/`:

```
app/
├── data/        # User data (never pushed)
├── exports/     # Generated outputs
├── python/      # Runtime ProxiPal.py
├── quality/
├── samples/
└── templates/
```

Notes:

- `app/python/ProxiPal.py` is a **runtime snapshot**
- It is copied once from `developer/src/ProxiPal.py` if missing
- It is **not kept in sync** with the developer source
- Existing runtime files are never overwritten automatically

Templates in `app/templates/` are updated **by filename only**; all other files are preserved.

---

## 📦 6. Dependencies

The setup script inspects imports and reports missing packages.

Install missing dependencies as needed, for example:

```bash
pip install pandas numpy scipy scikit-learn pymannkendall outliers plotly matplotlib seaborn rdmlpython
```

---

## ▶️ 7. Running ProxiPal

Typical usage is via Jupyter Notebook:

```bash
jupyter notebook
```

Then, from a notebook:

```python
from ProxiPal import *
```

Ensure your input data is placed under `app/data/` before execution.

---

## 🔄 8. Updating ProxiPal

To update your local copy of the code and templates:

```bash
git pull origin main
python proxipal_setup.py
```

Your data and runtime files are preserved.

---

## 🧭 9. Operational Rules (Summary)

- Research mode is for **analysis and real data**
- Development mode is for **publishing code**
- Never enable push in a clone containing real data
- Prefer clean clones for publication
- `developer/` is always present; `app/` is always local

---

© 2025 ProxiPal Project
