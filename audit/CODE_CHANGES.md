# Notebook source changes (publication copies)

All original numeric cells are preserved except the setup cell in each of the large-cat and experimental-kitten notebooks. Notebook outputs and execution counts were cleared. An explanatory Markdown cell was prepended. The small-cat notebook has no Python source edits.

## 02_large_cat_comparison.ipynb — original cell 1

```diff
--- before
+++ after
@@ -1,13 +1,11 @@
-try:
-    from google.colab import drive
-    drive.mount("/content/drive")
-except Exception as e:
-    print("Drive mount skipped:", e)
-
-MYDRIVE = Path("/content/drive/MyDrive")
-
-ROOT_CANDIDATES = list(MYDRIVE.rglob("multimode_cat_plots"))
-if not ROOT_CANDIDATES:
-    raise FileNotFoundError("multimode_cat_plots directory not found")
-
-ROOT = ROOT_CANDIDATES[0]
+# Publication-copy path adaptation only (no estimator changes).
+# Start Jupyter from the repository root, or from notebooks/.
+REPO_ROOT = Path.cwd().resolve()
+if not (REPO_ROOT / "notebooks").exists():
+    REPO_ROOT = REPO_ROOT.parent
+ROOT = REPO_ROOT / "data" / "multimode_cat_plots"
+if not (ROOT / "cat_state_data.dat").is_file():
+    raise FileNotFoundError(
+        "Missing data/multimode_cat_plots/cat_state_data.dat. "
+        "See docs/DATA.md for local-only preparation."
+    )
```

## 03_experimental_kitten.ipynb — original cell 1

```diff
--- before
+++ after
@@ -1,11 +1,7 @@
-try:
-    from google.colab import drive
-    drive.mount("/content/drive")
-except Exception as e:
-    print("Drive mount skipped:", e)
-
-ROOT = Path("/content/drive/MyDrive/KQSE_a1_R100_newapprox_floor003_clip_signfix")
-if not ROOT.exists():
-    ROOT = Path("/content/drive/MyDrive")
-
-OUTDIR = ROOT / "final_method_comparison_noise_correction_nofloor"
+# Publication-copy path adaptation only (no estimator changes).
+# Start Jupyter from the repository root, or from notebooks/.
+REPO_ROOT = Path.cwd().resolve()
+if not (REPO_ROOT / "notebooks").exists():
+    REPO_ROOT = REPO_ROOT.parent
+ROOT = REPO_ROOT / "data"
+OUTDIR = REPO_ROOT / "outputs" / "experimental_kitten"
```
