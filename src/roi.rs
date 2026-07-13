//! ROI geometry and mask export.
//!
//! ROIs (rectangle, circle, ellipse) are kept as an ordered list; each one
//! either adds to the selection or carves out of it. The exported mask is a
//! `(height, width)` grid of `u8` where selected pixels are 1 and everything
//! else is 0, written as a grayscale TIFF or a NumPy `.npy` array depending on
//! the output file extension.

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use std::path::Path;

/// Which drawing tool is currently active in the UI.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Rectangle,
    Ellipse,
    Circle,
}

impl Tool {
    pub fn label(self) -> &'static str {
        match self {
            Tool::Rectangle => "Rectangle",
            Tool::Ellipse => "Ellipse",
            Tool::Circle => "Circle",
        }
    }
}

/// An analytic ROI in image-pixel space (x = column, y = row).
#[derive(Clone, Copy)]
pub enum Geometry {
    Rect { x0: f32, y0: f32, x1: f32, y1: f32 },
    Ellipse { cx: f32, cy: f32, rx: f32, ry: f32 },
    Circle { cx: f32, cy: f32, r: f32 },
}

impl Geometry {
    /// Build a geometry of the given `tool` from a click-drag between two
    /// image-space corners.
    pub fn from_drag(tool: Tool, a: (f32, f32), b: (f32, f32)) -> Geometry {
        let (ax, ay) = a;
        let (bx, by) = b;
        match tool {
            Tool::Rectangle => Geometry::Rect {
                x0: ax,
                y0: ay,
                x1: bx,
                y1: by,
            },
            Tool::Ellipse => Geometry::Ellipse {
                cx: (ax + bx) * 0.5,
                cy: (ay + by) * 0.5,
                rx: (bx - ax).abs() * 0.5,
                ry: (by - ay).abs() * 0.5,
            },
            Tool::Circle => {
                // Centered on the drag start; radius follows the cursor.
                let r = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
                Geometry::Circle { cx: ax, cy: ay, r }
            }
        }
    }

    /// Inclusive integer bounding box `(x0, y0, x1, y1)` clamped to the image.
    fn bbox(&self, w: usize, h: usize) -> (usize, usize, usize, usize) {
        let (minx, miny, maxx, maxy) = match *self {
            Geometry::Rect { x0, y0, x1, y1 } => (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)),
            Geometry::Ellipse { cx, cy, rx, ry } => (cx - rx, cy - ry, cx + rx, cy + ry),
            Geometry::Circle { cx, cy, r } => (cx - r, cy - r, cx + r, cy + r),
        };
        let cx = |v: f32| v.clamp(0.0, w.saturating_sub(1) as f32);
        let cy = |v: f32| v.clamp(0.0, h.saturating_sub(1) as f32);
        (
            cx(minx.floor()) as usize,
            cy(miny.floor()) as usize,
            cx(maxx.ceil()) as usize,
            cy(maxy.ceil()) as usize,
        )
    }

    /// Translate the shape by `(dx, dy)` image pixels.
    pub fn translate(&mut self, dx: f32, dy: f32) {
        match self {
            Geometry::Rect { x0, y0, x1, y1 } => {
                *x0 += dx;
                *x1 += dx;
                *y0 += dy;
                *y1 += dy;
            }
            Geometry::Ellipse { cx, cy, .. } | Geometry::Circle { cx, cy, .. } => {
                *cx += dx;
                *cy += dy;
            }
        }
    }

    /// Center of the shape in image pixel coordinates.
    pub fn center(&self) -> (f32, f32) {
        match *self {
            Geometry::Rect { x0, y0, x1, y1 } => ((x0 + x1) * 0.5, (y0 + y1) * 0.5),
            Geometry::Ellipse { cx, cy, .. } | Geometry::Circle { cx, cy, .. } => (cx, cy),
        }
    }

    /// Does pixel center `(px, py)` fall inside the shape?
    pub fn contains(&self, px: f32, py: f32) -> bool {
        match *self {
            Geometry::Rect { x0, y0, x1, y1 } => {
                px >= x0.min(x1) && px <= x0.max(x1) && py >= y0.min(y1) && py <= y0.max(y1)
            }
            Geometry::Ellipse { cx, cy, rx, ry } => {
                if rx <= 0.0 || ry <= 0.0 {
                    return false;
                }
                let dx = (px - cx) / rx;
                let dy = (py - cy) / ry;
                dx * dx + dy * dy <= 1.0
            }
            Geometry::Circle { cx, cy, r } => {
                if r <= 0.0 {
                    return false;
                }
                (px - cx).powi(2) + (py - cy).powi(2) <= r * r
            }
        }
    }

    /// Stamp this shape into `mask`, setting covered pixels to `value`.
    pub fn stamp(&self, mask: &mut Array2<bool>, value: bool) {
        let (h, w) = (mask.shape()[0], mask.shape()[1]);
        if h == 0 || w == 0 {
            return;
        }
        let (x0, y0, x1, y1) = self.bbox(w, h);
        for y in y0..=y1 {
            for x in x0..=x1 {
                if self.contains(x as f32 + 0.5, y as f32 + 0.5) {
                    mask[(y, x)] = value;
                }
            }
        }
    }
}

/// Write the mask to `path` with selected pixels = 1 and the rest = 0.
///
/// The format follows the extension: `.tif`/`.tiff` writes an 8-bit grayscale
/// TIFF, `.npy` a NumPy `uint8` array of shape `(height, width)`.
pub fn save_mask(path: &Path, mask: &Array2<bool>) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "npy" => {
            let m: Array2<u8> = mask.mapv(|b| b as u8);
            ndarray_npy::write_npy(path, &m)
                .with_context(|| format!("write mask to {}", path.display()))?;
        }
        "tif" | "tiff" => {
            use tiff::encoder::{colortype::Gray8, TiffEncoder};

            let (h, w) = (mask.shape()[0], mask.shape()[1]);
            let file = std::fs::File::create(path)
                .with_context(|| format!("create {}", path.display()))?;
            let mut enc = TiffEncoder::new(std::io::BufWriter::new(file))
                .with_context(|| format!("init TIFF encoder for {}", path.display()))?;
            let buf: Vec<u8> = mask.iter().map(|&b| b as u8).collect();
            enc.write_image::<Gray8>(w as u32, h as u32, &buf)
                .with_context(|| format!("write mask to {}", path.display()))?;
        }
        other => bail!("Unsupported mask extension '.{other}' — use .tif, .tiff or .npy"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("roi_selector_mask_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn rect_stamp_covers_expected_pixels() {
        let mut m = Array2::<bool>::default((10, 10));
        Geometry::Rect {
            x0: 2.0,
            y0: 3.0,
            x1: 5.0,
            y1: 6.0,
        }
        .stamp(&mut m, true);
        assert!(m[(3, 2)] && m[(5, 4)]);
        assert!(!m[(3, 6)] && !m[(7, 2)]);
        assert_eq!(m.iter().filter(|&&b| b).count(), 9, "3x3 pixel centers");
    }

    #[test]
    fn circle_and_ellipse_stamp_are_symmetric() {
        let mut c = Array2::<bool>::default((20, 20));
        Geometry::Circle {
            cx: 10.0,
            cy: 10.0,
            r: 5.0,
        }
        .stamp(&mut c, true);
        assert!(c[(10, 10)] && c[(10, 14)] && c[(14, 10)]);
        assert!(!c[(10, 16)]);

        let mut e = Array2::<bool>::default((20, 20));
        Geometry::Ellipse {
            cx: 10.0,
            cy: 10.0,
            rx: 6.0,
            ry: 3.0,
        }
        .stamp(&mut e, true);
        assert!(e[(10, 15)] && !e[(15, 10)], "wide but not tall");
    }

    #[test]
    fn subtract_carves_out_of_the_mask() {
        let mut m = Array2::<bool>::default((10, 10));
        Geometry::Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 9.0,
            y1: 9.0,
        }
        .stamp(&mut m, true);
        Geometry::Circle {
            cx: 5.0,
            cy: 5.0,
            r: 2.0,
        }
        .stamp(&mut m, false);
        assert!(!m[(5, 5)] && m[(0, 0)]);
    }

    #[test]
    fn save_mask_npy_roundtrips_as_u8_ones_and_zeros() {
        use ndarray_npy::ReadNpyExt;

        let dir = tmp_dir("npy");
        let path = dir.join("mask.npy");
        let mut m = Array2::<bool>::default((4, 5));
        m[(1, 2)] = true;
        save_mask(&path, &m).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let back = Array2::<u8>::read_npy(file).unwrap();
        assert_eq!(back.shape(), &[4, 5]);
        assert_eq!(back[(1, 2)], 1);
        assert_eq!(back.iter().map(|&v| v as usize).sum::<usize>(), 1);
    }

    #[test]
    fn save_mask_tiff_roundtrips_as_ones_and_zeros() {
        let dir = tmp_dir("tiff");
        let path = dir.join("mask.tif");
        let mut m = Array2::<bool>::default((4, 5));
        m[(3, 4)] = true;
        m[(0, 0)] = true;
        save_mask(&path, &m).unwrap();

        let stack = crate::loader::load_paths(&[path]).unwrap();
        assert_eq!((stack.height, stack.width), (4, 5));
        let img = &stack.frames[0];
        assert_eq!(img[(3, 4)], 1.0);
        assert_eq!(img[(0, 0)], 1.0);
        assert_eq!(img.iter().sum::<f32>(), 2.0);
    }

    #[test]
    fn save_mask_rejects_unknown_extension() {
        let dir = tmp_dir("ext");
        let m = Array2::<bool>::default((2, 2));
        assert!(save_mask(&dir.join("mask.png"), &m).is_err());
    }
}
