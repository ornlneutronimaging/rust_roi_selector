//! Loading a stack of images from TIFF and NumPy `.npy` files.
//!
//! A "stack" can come from:
//!   * several single-image files selected together,
//!   * a multi-page TIFF (each page is a frame),
//!   * a 2-D `.npy` array (one frame — e.g. an integrated image exported by a
//!     notebook) or a 3-D `.npy` array (one frame per plane along axis 0).
//!
//! Every frame is normalised to an `Array2<f32>` with shape `(height, width)`,
//! row-major, so the rest of the program never has to care about the on-disk
//! sample format. TIFF frames are re-oriented according to the detector that
//! wrote them (see [`detector_orientation`]: Timepix transposed, CCD flipped
//! vertically, QHY not decided yet / unknown as-is); `.npy` arrays come from
//! Python callers that already pass display-ready data and are loaded as-is.

use anyhow::{anyhow, bail, Context, Result};
pub use detector_orientation::{Detector, Orientation, Selection, Source};
use ndarray::{Array2, Array3};
use std::path::{Path, PathBuf};

/// An in-memory stack of equally-sized frames.
pub struct ImageStack {
    pub frames: Vec<Array2<f32>>,
    pub width: usize,
    pub height: usize,
    /// Source file for each frame (parallel to `frames`); useful for the UI.
    pub sources: Vec<PathBuf>,
    /// Detector the stack was recorded with (automatic guess + user override).
    pub detector: Selection,
    /// How the TIFF frames were re-oriented on load (see [`to_frame`]). A
    /// mask saved from this stack must be put back in the on-disk orientation
    /// with [`Orientation::undo`] so it aligns with the input files; `.npy`
    /// input is loaded as-is (`Identity`), so its masks are saved as-is.
    pub orientation: Orientation,
}

impl ImageStack {
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }
}

/// File extensions we know how to open.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["tif", "tiff", "npy"];

fn ext_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

/// Load and concatenate every frame contained in `paths`, in sorted order,
/// with the detector guessed from the first path (see [`Selection::from_path`]).
pub fn load_paths(paths: &[PathBuf]) -> Result<ImageStack> {
    let detector = paths.first().map(|p| Selection::from_path(p)).unwrap_or_default();
    load_paths_with_progress(paths, detector, |_, _| {})
}

/// [`load_paths`] with the detector forced.
pub fn load_paths_as(paths: &[PathBuf], detector: Detector) -> Result<ImageStack> {
    let mut sel = paths.first().map(|p| Selection::from_path(p)).unwrap_or_default();
    sel.manual = Some(detector);
    load_paths_with_progress(paths, sel, |_, _| {})
}

/// Like [`load_paths`], but with an explicit detector selection, and invokes
/// `on_progress(files_done, files_total)` after each input file is read, so a
/// caller can drive a progress bar.
pub fn load_paths_with_progress<F>(
    paths: &[PathBuf],
    detector: Selection,
    mut on_progress: F,
) -> Result<ImageStack>
where
    F: FnMut(usize, usize),
{
    if paths.is_empty() {
        bail!("No files selected");
    }

    let mut sorted = paths.to_vec();
    sorted.sort();
    let total = sorted.len();

    let mut frames: Vec<Array2<f32>> = Vec::new();
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut dims: Option<(usize, usize)> = None;
    // `.npy` input is already display-ready: only TIFF frames are re-oriented.
    let mut orientation = Orientation::Identity;

    for (idx, path) in sorted.iter().enumerate() {
        let loaded = match ext_of(path).as_str() {
            "tif" | "tiff" => {
                orientation = detector.orientation();
                load_tiff(path, orientation)?
            }
            "npy" => load_npy(path)?,
            other => bail!("Unsupported file type '.{other}': {}", path.display()),
        };

        for frame in loaded {
            let (h, w) = (frame.shape()[0], frame.shape()[1]);
            match dims {
                None => dims = Some((h, w)),
                Some((dh, dw)) if (dh, dw) != (h, w) => {
                    bail!(
                        "Frame size mismatch: {}x{} in {} does not match {}x{}",
                        w,
                        h,
                        path.display(),
                        dw,
                        dh
                    );
                }
                _ => {}
            }
            frames.push(frame);
            sources.push(path.clone());
        }
        on_progress(idx + 1, total);
    }

    let (height, width) = dims.ok_or_else(|| anyhow!("No frames were loaded"))?;
    Ok(ImageStack {
        frames,
        width,
        height,
        sources,
        detector,
        orientation,
    })
}

/// Collect every supported image file directly inside `dir`.
pub fn list_supported_in_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let path = entry?.path();
        if path.is_file() && SUPPORTED_EXTENSIONS.contains(&ext_of(&path).as_str()) {
            out.push(path);
        }
    }
    if out.is_empty() {
        bail!("No TIFF or .npy files found in {}", dir.display());
    }
    Ok(out)
}

/// Read every page of a (possibly multi-page) TIFF file.
fn load_tiff(path: &Path, orientation: Orientation) -> Result<Vec<Array2<f32>>> {
    use tiff::decoder::{Decoder, DecodingResult};

    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("decode TIFF {}", path.display()))?;

    let mut out = Vec::new();
    loop {
        let (w, h) = decoder.dimensions()?;
        let (w, h) = (w as usize, h as usize);

        let data = decoder.read_image()?;
        let values: Vec<f32> = match data {
            DecodingResult::U8(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::U16(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::U32(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::U64(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::I8(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::I16(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::I32(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::I64(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::F16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            DecodingResult::F32(v) => v,
            DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        };

        out.push(to_frame(values, w, h, orientation)?);

        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
    }

    Ok(out)
}

/// Read a `.npy` file: a 2-D array becomes one frame, a 3-D array one frame per
/// plane along axis 0. The dtype is probed among the common numeric types.
fn load_npy(path: &Path) -> Result<Vec<Array2<f32>>> {
    use ndarray_npy::ReadNpyExt;
    use std::io::Cursor;

    let bytes = std::fs::read(path).with_context(|| format!("open {}", path.display()))?;

    macro_rules! try_2d {
        ($t:ty) => {
            if let Ok(a) = Array2::<$t>::read_npy(Cursor::new(&bytes[..])) {
                return Ok(vec![a.mapv(|v| v as f32)]);
            }
        };
    }
    macro_rules! try_3d {
        ($t:ty) => {
            if let Ok(a) = Array3::<$t>::read_npy(Cursor::new(&bytes[..])) {
                return Ok(a
                    .outer_iter()
                    .map(|plane| plane.mapv(|v| v as f32))
                    .collect());
            }
        };
    }

    try_2d!(f32);
    try_2d!(f64);
    try_2d!(u8);
    try_2d!(u16);
    try_2d!(i16);
    try_2d!(u32);
    try_2d!(i32);
    try_2d!(u64);
    try_2d!(i64);
    if let Ok(a) = Array2::<bool>::read_npy(Cursor::new(&bytes[..])) {
        return Ok(vec![a.mapv(|v| if v { 1.0 } else { 0.0 })]);
    }
    try_3d!(f32);
    try_3d!(f64);
    try_3d!(u8);
    try_3d!(u16);
    try_3d!(i16);
    try_3d!(u32);
    try_3d!(i32);
    try_3d!(u64);
    try_3d!(i64);

    bail!(
        "Unsupported .npy dtype or shape (need a 2-D or 3-D numeric array): {}",
        path.display()
    )
}

/// Turn a flat, row-major buffer into a frame. If the buffer carries several
/// samples per pixel (e.g. RGB TIFF) only the first sample is kept.
///
/// TIFF pages are re-oriented on the way in according to the detector: a
/// Timepix page is transposed (the detector writes rows/columns swapped
/// relative to the sample orientation, same convention as rust_tiff_viewer),
/// a CCD page is flipped vertically, QHY (not decided yet) and unknown
/// detectors are kept as-is. The .npy path is never re-oriented — those
/// arrays come from Python callers that already pass display-ready data.
fn to_frame(values: Vec<f32>, w: usize, h: usize, orientation: Orientation) -> Result<Array2<f32>> {
    let expected = w * h;
    let frame = if values.len() == expected {
        Array2::from_shape_vec((h, w), values)?
    } else if expected > 0 && values.len() % expected == 0 {
        let spp = values.len() / expected;
        let first: Vec<f32> = (0..expected).map(|i| values[i * spp]).collect();
        Array2::from_shape_vec((h, w), first)?
    } else {
        bail!(
            "Pixel count {} is not compatible with {}x{}",
            values.len(),
            w,
            h
        )
    };
    Ok(orientation.apply(frame))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("roi_selector_loader_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn npy_2d_f64_loads_as_one_frame() {
        let dir = tmp_dir("npy2d");
        let path = dir.join("img.npy");
        let a = ndarray::Array2::<f64>::from_shape_fn((3, 4), |(y, x)| (y * 4 + x) as f64);
        ndarray_npy::write_npy(&path, &a).unwrap();

        let stack = load_paths(&[path]).unwrap();
        assert_eq!(stack.n_frames(), 1);
        assert_eq!((stack.height, stack.width), (3, 4));
        assert_eq!(stack.frames[0][(2, 3)], 11.0);
    }

    #[test]
    fn npy_3d_u16_loads_one_frame_per_plane() {
        let dir = tmp_dir("npy3d");
        let path = dir.join("cube.npy");
        let a = ndarray::Array3::<u16>::from_shape_fn((2, 3, 4), |(k, y, x)| {
            (100 * k + y * 4 + x) as u16
        });
        ndarray_npy::write_npy(&path, &a).unwrap();

        let stack = load_paths(&[path]).unwrap();
        assert_eq!(stack.n_frames(), 2);
        assert_eq!((stack.height, stack.width), (3, 4));
        assert_eq!(stack.frames[1][(0, 0)], 100.0);
        assert_eq!(stack.orientation, Orientation::Identity);
    }

    fn write_tiff_u16(path: &Path, w: usize, h: usize, values: &[u16]) {
        use tiff::encoder::{colortype::Gray16, TiffEncoder};
        let file = std::fs::File::create(path).unwrap();
        let mut enc = TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
        enc.write_image::<Gray16>(w as u32, h as u32, values).unwrap();
    }

    #[test]
    fn tiff_orientation_follows_detector() {
        let dir = tmp_dir("tiff_orientation");
        let path = dir.join("img.tif");
        // 3 wide × 2 tall on disk: [1 2 3; 4 5 6]
        write_tiff_u16(&path, 3, 2, &[1, 2, 3, 4, 5, 6]);

        // unknown detector (temp folder): as-is
        let stack = load_paths(&[path.clone()]).unwrap();
        assert_eq!(stack.orientation, Orientation::Identity);
        assert_eq!((stack.height, stack.width), (2, 3));
        assert_eq!(stack.frames[0][(1, 0)], 4.0);

        // Timepix: transposed, 2 wide × 3 tall
        let stack = load_paths_as(&[path.clone()], Detector::Timepix).unwrap();
        assert_eq!(stack.orientation, Orientation::Transpose);
        assert_eq!((stack.height, stack.width), (3, 2));
        assert_eq!(stack.frames[0][(1, 0)], 2.0);

        // CCD: flipped vertically, same size
        let stack = load_paths_as(&[path], Detector::Ccd).unwrap();
        assert_eq!(stack.orientation, Orientation::FlipVertical);
        assert_eq!((stack.height, stack.width), (2, 3));
        assert_eq!(stack.frames[0][(0, 0)], 4.0);
    }
}
