# ROI Selector

Native GUI (Rust, [egui](https://github.com/emilk/egui)) to select regions of
interest on a neutron-imaging data set and export them as a **mask file**:
selected pixels are **1**, everything else is **0**.

Companion tool for the VENUS marimo notebooks (e.g. the TOF normalization
notebook's "Match sample background with OB" step), and usable standalone.
Interaction model based on
[rust_hyperspectral_masker](https://github.com/ornlneutronimaging/rust_hyperspectral_masker).

## Build

```bash
cargo build --release
# binary: target/release/roi_selector
```

## Usage

```bash
# Standalone: open data from within the app (Open Files… / Open Folder…)
roi_selector

# Display the integrated image of a folder of TIFF images
roi_selector /SNS/VENUS/IPTS-XXXX/.../Run_YYYY

# Display a pre-integrated 2-D .npy image and write the mask where the
# caller expects it — the workflow used by the marimo notebooks
roi_selector integrated_sample.npy --output mask.tif --called-from-python
```

- **INPUT** — TIFF file(s) (multi-page supported), a folder of TIFF/`.npy`
  images, or a 2-D/3-D `.npy` array. Multiple frames are combined into one
  displayed image (Sum / Mean / Max, selectable in the toolbar).
- **`-o, --output <PATH>`** — enables the **✅ Save mask & quit** button, which
  writes the mask to `PATH` and closes the app. `.tif`/`.tiff` → 8-bit
  grayscale TIFF; `.npy` → `uint8` NumPy array of shape `(height, width)`.
- **`--called-from-python`** — the app is driven by another application that
  is blocked waiting for the mask: the button reads **⏎ Return to main
  application** instead, which writes the mask to `--output` (required with
  this flag) and closes the window so the caller resumes.
- Without `--output`, use **💾 Save mask as…** to pick the destination.

## Controls

| Action | How |
| --- | --- |
| Draw a region | Pick Rectangle / Ellipse / Circle, then click-drag on the image |
| Move a region | Drag inside it |
| Resize a region | Drag its white handles (select it first) |
| Edit exact pixel values | Select the region and edit the fields in the right panel |
| Carve a hole | Check **Subtract**, then draw over an existing region |
| Delete | Select, then `Delete`/`Backspace`, or the Delete button in the panel |
| Contrast / colormap / zoom | Toolbar |

The status bar shows the live count of selected pixels and the pixel value
under the cursor.

## Calling from Python / marimo

```python
import subprocess, tempfile
from pathlib import Path
import numpy as np
import tifffile

with tempfile.TemporaryDirectory() as tmp:
    image_path = Path(tmp) / "integrated_sample.npy"
    mask_path = Path(tmp) / "mask.tif"
    np.save(image_path, integrated_sample)

    subprocess.run(
        ["roi_selector", str(image_path),
         "--output", str(mask_path), "--called-from-python"],
        check=True,
    )  # blocks until the user clicks "⏎ Return to main application"

    mask = tifffile.imread(mask_path)  # uint8, 1 inside the ROIs, 0 outside
```

The window opens on the machine running Python, so this requires a working
display (e.g. a ThinLinc session); it will not appear on a remote browser
viewing a headless notebook server.

## Tests

```bash
cargo test
```
