# DecipherTCR — Custom Visualization Functions

This folder contains notebooks for analyzing T cell dynamics in GVHD using the DecipherTCR framework. Three custom plotting functions are defined in the main analysis notebook to visualize TCR clone transitions between anatomical compartments (Lamina Propria → Intraepithelial).

---

## Setup

The functions depend on the following libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import ConnectionPatch
```

All functions accept an `AnnData` object (`adata`) and operate on `.obs` metadata and `.obsm` embeddings (typically `decipher_v`).

---

## Functions

### 1. `plot_cell_transitions_stream`

Visualizes TCR clone transitions as a **streamplot** (velocity field) overlaid on the embedding. The stream field is computed by interpolating clone-level transition vectors (mean LP position → mean IE position) onto a grid using nearest-neighbor smoothing.

**Signature:**
```python
ax = plot_cell_transitions_stream(
    adata,
    category_key,       # obs column separating the two compartments (e.g. 'LPorIE')
    category_1,         # source compartment label (e.g. 'Lamina Propria Cells')
    category_2,         # target compartment label (e.g. 'Intraepithelial Cells')
    cell_type_key,      # obs column for clone identity (e.g. 'cc_aa_identity_Grade')
    color_cat='T_cell_subsets_refined_resi',  # obs column used to color background cells
    basis='umap',                      # embedding key in adata.obsm
    n_grid=50,                         # grid resolution for streamplot
    smooth=0.5,                        # kernel smoothing bandwidth
    min_mass=1,                        # minimum density to show stream
    density=2,                         # streamline density
    arrow_size=1,
    linewidth=1,
    density_threshold=0.5,             # grid points below this density are masked
    figsize=(20, 20),
    scale_arrows_by_clone_size=True,   # vary arrow width by clone size
    min_arrow_width=0.5,
    max_arrow_width=5,
    show_only_small_clones=True,       # restrict to clones below size threshold
    small_clone_threshold=20,          # max clone size when filtering
)
```

**Returns:** `matplotlib.axes.Axes`

**Example (from notebook):**
```python
ax = plot_cell_transitions_stream(
    adata_largest_clones,
    category_key='LPorIE',
    category_1='Lamina Propria Cells',
    category_2='Intraepithelial Cells',
    cell_type_key='cc_aa_identity_Grade',
    color_cat='T_cell_subsets_refined_resi',
    basis='decipher_v',
    n_grid=50,
    smooth=0.8,
    density=1.8,
    arrow_size=2.5,
    linewidth=2.5,
    density_threshold=0.1,
    figsize=(15, 6),
    scale_arrows_by_clone_size=True,
    min_arrow_width=1.0,
    max_arrow_width=7.0,
    show_only_small_clones=False,
    small_clone_threshold=20,
)
plt.tight_layout()
plt.savefig(FIGURES_TRA_DIR / 'transition_stream.png', dpi=300, transparent=True)
plt.show()
```

**Key behavior:**
- Only clones present in **both** compartments contribute to the velocity field.
- When `scale_arrows_by_clone_size=True`, arrow width is linearly scaled between `min_arrow_width` and `max_arrow_width` based on clone size.
- Set `show_only_small_clones=False` to include all clones regardless of size.
- Axis labels ("Decipher 1", "Decipher 2") and coordinate arrows are drawn automatically.

---

### 2. `plot_cell_transitions_scatter`

Visualizes the embedding as a **scatter plot** with cells colored by a chosen category. Optionally overlays clone transition arrows between compartments.

**Signature:**
```python
ax = plot_cell_transitions_scatter(
    adata,
    category_key,               # obs column separating compartments
    cell_type_key,              # obs column for clone identity
    color_cat='leiden_new_cluster_2',  # obs column used to color cells
    basis='umap',
    figsize=(20, 20),
    show_only_small_clones=True,
    small_clone_threshold=20,
    point_size=6,
    point_alpha=1,
    show_clone_transitions=False,  # overlay transition arrows
    arrow_color='black',
    arrow_alpha=0.5,
    arrow_width=1,
    category_1=None,           # source compartment (required if show_clone_transitions=True)
    category_2=None,           # target compartment (required if show_clone_transitions=True)
)
```

**Returns:** `matplotlib.axes.Axes`

**Example — color by T cell subset:**
```python
ax = plot_cell_transitions_scatter(
    Tcell_sc_3_7000deg,
    category_key='LPorIE',
    cell_type_key='cc_aa_identity_Grade',
    color_cat='T_cell_subsets_refined_resi',
    basis='decipher_v',
    figsize=(15, 6),
    show_only_small_clones=False,
    point_size=6,
)
plt.tight_layout()
plt.savefig(FIGURES_TRA_DIR / 'scatter_subsets.png', dpi=300, transparent=True)
plt.show()
```

**Example — color by LP/IE compartment:**
```python
ax = plot_cell_transitions_scatter(
    Tcell_sc_3_7000deg,
    category_key='LPorIE',
    cell_type_key='cc_aa_identity_Grade',
    color_cat='LPorIE',
    basis='decipher_v',
    figsize=(15, 6),
    show_only_small_clones=False,
    point_size=6,
)
plt.savefig(FIGURES_TRA_DIR / 'scatter_LP_IE.png', dpi=300, transparent=True)
plt.show()
```

**Key behavior:**
- Colors are pulled from `adata.uns[f'{color_cat}_colors']` if available, otherwise a `husl` palette is used.
- Transition arrows are drawn from mean LP position to mean IE position per clone, and are only shown when `show_clone_transitions=True` with `category_1` and `category_2` provided.

---

### 3. `validate_clone_transitions`

Produces a **grid of per-clone subplots** showing LP and IE cells individually, with an arrow indicating the mean transition direction. Useful for manually validating that the stream direction is biologically sensible.

**Signature:**
```python
fig, axes = validate_clone_transitions(
    adata,
    clone_key,                          # obs column for clone identity
    location_key='LPorIE',
    category_1='Lamina Propria Cells',
    category_2='Intraepithelial Cells',
    basis='umap',
    min_shared_cells=3,    # min cells per compartment for a clone to be shown
    n_cols=4,              # columns in the subplot grid
    figsize=None,          # auto-calculated if None
    arrow_scale=1.0,
    cluster_key='T_cell_subsets_refined_resi',  # obs column for cell coloring
)
```

**Returns:** `(matplotlib.figure.Figure, numpy.ndarray of Axes)`

**Example (from notebook):**
```python
fig, axes = validate_clone_transitions(
    adata=adata_largest_clones,
    clone_key='cc_aa_identity_Grade',
    location_key='LPorIE',
    basis='decipher_v',
    min_shared_cells=3,
    n_cols=4,
    figsize=(15, 8),
    arrow_scale=1.0,
    cluster_key='T_cell_subsets_refined_resi',
)
fig.savefig(FIGURES_TRA_DIR / 'validate_clone_transitions.png', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig(FIGURES_TRA_DIR / 'validate_clone_transitions.pdf', dpi=300, transparent=True, bbox_inches='tight')
plt.show()
```

**Key behavior:**
- Clones are sorted by transition magnitude (largest displacement first).
- Each subplot shows LP cells (blue star = mean) and IE cells (red star = mean) with a black arrow from LP mean to IE mean.
- Clones with fewer than `min_shared_cells` in either compartment are excluded.
- A shared legend is placed outside the grid using `bbox_to_anchor`.

---

## Common Parameters

| Parameter | Description |
|---|---|
| `basis` | Key in `adata.obsm` for the 2D embedding (e.g. `'decipher_v'`, `'umap'`) |
| `category_key` | `adata.obs` column that labels compartments (e.g. `'LPorIE'`) |
| `category_1` / `category_2` | String labels for source and target compartments |
| `cell_type_key` / `clone_key` | `adata.obs` column identifying individual TCR clones |
| `color_cat` / `cluster_key` | `adata.obs` column used to color cells |

---

## Figure Output Pattern

All figures are saved in three formats (PNG, SVG, PDF) for publication flexibility:

```python
plt.savefig(FIGURES_TRA_DIR / 'filename.png', dpi=300, transparent=True, bbox_inches='tight')
plt.savefig(FIGURES_TRA_DIR / 'filename.svg', dpi=300, transparent=True, bbox_inches='tight')
plt.savefig(FIGURES_TRA_DIR / 'filename.pdf', dpi=300, transparent=True, bbox_inches='tight')
```

`FIGURES_TRA_DIR` is a `pathlib.Path` defined at the top of the notebook pointing to the figures output directory.
