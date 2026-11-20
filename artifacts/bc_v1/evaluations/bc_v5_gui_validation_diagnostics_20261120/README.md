# GUI diagnostics BC-v5 VALIDATION (20/11/2026)

Contenuti: `raw/good_datas_GUI.zip` (originale), `audit/episode_audit.csv`, `audit/manifest.json`, `audit/report.md`.

Per installazione locale senza modificare files di codice:

```bash
cd ~/MSSR_thesis
mkdir -p artifacts/bc_v1/gui_diagnostics
unzip -n "$HOME/Downloads/BCv5_GUI_diagnostics_VALIDATION_20261120.zip" -d artifacts/bc_v1/gui_diagnostics
python3 -m json.tool artifacts/bc_v1/gui_diagnostics/bc_v5_gui_validation_diagnostics_20261120/audit/manifest.json >/dev/null
git status --short
```

Nota: in `.gitignore` la directory `artifacts/bc_v1` e tutti gli archivi `.zip` sono ignorati. La copia locale non è una git commit.

Preflight di sola lettura per il prossimo TEST, dopo aver copiato il bundle dentro la repo:

```bash
cd ~/MSSR_thesis
python3 artifacts/bc_v1/gui_diagnostics/bc_v5_gui_validation_diagnostics_20261120/audit/test_gate_readiness.py
```

Non esegue il TEST e non apre i suoi campioni; controlla solo split, identità del modello e policy F2.
