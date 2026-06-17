# Notebooks

Tutorial-style, **self-contained** walkthroughs (they run on Google Colab — no local
package needed). They live in [`tutorials/`](tutorials/) and are versioned: a notebook
is iterated as `_v1`, `_v2`, … so the history of each tutorial is preserved.

## `tutorials/`

- **`padel_ar_showcase_v1.ipynb`** — ⭐ the flagship: raw clip → RF-DETR detection →
  court-region filter → ByteTrack identities (+ EMA stabilizer) → perspective AR ground
  rings → court homography (corner picker) → FIFA-style zonal heatmap → segmentation
  matte (players in front) → full-clip "final cut" with timed transitions.
- `01_object_detection.ipynb` — the original Stage-1 walkthrough: detecting players,
  exploring confidence thresholds and per-frame counts.
- `court_corners.txt` — saved court calibration used by the flagship notebook (set it
  with the in-notebook picker or `padel-vision court adjust`).

> Tips
> - Heavy outputs bloat git — clear notebook outputs before committing, or keep the
>   embedded preview small (the `.ipynb_checkpoints` rule is already in `.gitignore`).
> - Need a GPU runtime on Colab: **Runtime → Change runtime type → GPU**.
