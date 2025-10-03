````markdown
# 🚀 Getting Started with ProxiPal

Welcome to **ProxiPal**!  
This guide will help you set up your local workspace and run the application for the first time.

---

## 📦 1. Prerequisites

Before you begin, make sure you have:

- **Python 3.9+** installed  
- **pip** available on your system  
- Recommended: a **virtual environment** (e.g. `venv` or `conda`)

To check:
```bash
python --version
pip --version
````

---

## 🧰 2. Clone the Repository

Open your terminal (Anaconda Prompt, PowerShell, or CMD) and run:

```bash
git clone https://github.com/proxipal-pcr/proxipal.git
cd proxipal
```

This folder contains:

```
proxipal/
├── developer/         # Source code, templates, documentation
├── proxipal_setup.py  # Setup script
├── README.md
└── LICENSE
```

---

## ⚙️ 3. Run the Setup Script

Run the setup script to prepare your working environment:

```bash
python proxipal_setup.py
```

The script will:

* ✅ Create an **app/** workspace with folders:

  ```
  app/
  ├── data/
  ├── exports/
  ├── python/
  ├── quality/
  ├── samples/
  └── templates/
  ```
* 📄 Copy the main code file from `developer/src/ProxiPal.py` into `app/python/`
* 📋 Copy templates from `developer/templates/` into `app/templates/`
* 🔍 Inspect the code for required Python packages
* ⚠️ Advise you of any missing dependencies

If any packages are missing, install them with:

```bash
pip install package_name
```

or all at once:

```bash
pip install pandas numpy scipy scikit-learn pymannkendall outliers plotly matplotlib seaborn rdmlpython
```

---

## ▶️ 4. Run ProxiPal

Once setup is complete:

```bash
cd app/python
python ProxiPal.py
```

ProxiPal expects the following folder structure under `app/`:

```
data/       # Your raw input data
samples/    # Sample metadata
templates/  # Excel or CSV templates
exports/    # Generated outputs
```

Place your files accordingly before running.

---

## 🧪 5. Updating

To update your local copy:

```bash
cd proxipal
git pull origin main
python proxipal_setup.py
```

This ensures you have the latest code and templates.

---

## 💡 6. Notes

* All user data and work should happen inside `app/`
* `developer/` is for source code and should not be modified by non-developers
* You can rerun the setup script anytime — it’s safe and will not overwrite existing data

---

## 🆘 7. Need Help?

For questions or issues, please open an [Issue](https://github.com/proxipal-pcr/proxipal/issues) on GitHub.

---

© 2025 ProxiPal Project

````
