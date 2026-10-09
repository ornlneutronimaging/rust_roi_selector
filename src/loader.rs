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
//! vertically, QHY rotated 90° counterclockwise / unknown as-is). `.npy` arrays
//! come from Python callers and are loaded as-is unless the caller names the
//! detector explicitly (`--detector`, or the toolbar combobox): the folder
//! guess never applies to them, since they sit in temporary folders and are
//! usually already display-ready.

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
    load_paths_with_progress(paths, detector, |_| {})
}

/// [`load_paths`] with the detector forced.
pub fn load_paths_as(paths: &[PathBuf], detector: Detector) -> Result<ImageStack> {
    let mut sel = paths.first().map(|p| Selection::from_path(p)).unwrap_or_default();
    sel.manual = Some(detector);
    load_paths_with_progress(paths, sel, |_| {})
}

/// How far a load has come. Files are a poor unit of progress — a whole
/// stack can be one `.npy` — so the bytes read (known up front from the
/// file sizes) drive the bar and the frame count says what is being loaded.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadProgress {
    pub files_done: usize,
    pub files_total: usize,
    /// Frames the files hold, from the `.npy` headers (a TIFF counts as one
    /// until it is read, then as its real page count).
    pub frames_total: usize,
    pub frames_done: usize,
    pub bytes_done: u64,
    pub bytes_total: u64,
    /// File name being read right now.
    pub current: String,
}

impl LoadProgress {
    /// Fraction of the work done, from the bytes when they are known.
    pub fn fraction(&self) -> f32 {
        if self.bytes_total > 0 {
            (self.bytes_done as f64 / self.bytes_total as f64).clamp(0.0, 1.0) as f32
        } else if self.files_total > 0 {
            self.files_done as f32 / self.files_total as f32
        } else {
            0.0
        }
    }
}

/// Bytes for the progress text: `512 B`, `8.4 MB`, `3.31 GB`.
pub fn format_bytes(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} kB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}

/// The frame count announced by a `.npy` header (axis 0 of a 3-D array, 1
/// for a 2-D one) without reading the data.
pub fn npy_frame_count(path: &Path) -> Option<usize> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 12];
    file.read_exact(&mut head).ok()?;
    if &head[..6] != b"\x93NUMPY" {
        return None;
    }
    let (header_len, offset) = match head[6] {
        1 => (u16::from_le_bytes([head[8], head[9]]) as usize, 10),
        _ => (u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize, 12),
    };
    let mut header = vec![0u8; header_len];
    let already = 12 - offset;
    header[..already].copy_from_slice(&head[offset..]);
    file.read_exact(&mut header[already..]).ok()?;
    let header = String::from_utf8_lossy(&header);
    let shape = header.split("'shape':").nth(1)?;
    let inner = shape.split('(').nth(1)?.split(')').next()?;
    let dims: Vec<usize> = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().ok())
        .collect::<Option<Vec<usize>>>()?;
    match dims.len() {
        2 => Some(1),
        3 => Some(dims[0]),
        _ => None,
    }
}

/// Like [`load_paths`], but with an explicit detector selection, and invokes
/// `on_progress` as the files are read (after every file, and every few
/// megabytes of a large `.npy`), so a caller can drive a progress bar.
pub fn load_paths_with_progress<F>(
    paths: &[PathBuf],
    detector: Selection,
    mut on_progress: F,
) -> Result<ImageStack>
where
    F: FnMut(&LoadProgress),
{
    if paths.is_empty() {
        bail!("No files selected");
    }

    let mut sorted = paths.to_vec();
    sorted.sort();
    let total = sorted.len();

    // Size the work up front: bytes from the file sizes, frames from the
    // `.npy` headers (a TIFF counts as one frame until read).
    let mut progress = LoadProgress {
        files_total: total,
        ..LoadProgress::default()
    };
    let mut announced: Vec<usize> = Vec::with_capacity(total);
    for path in &sorted {
        progress.bytes_total += std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let frames = match ext_of(path).as_str() {
            "npy" => npy_frame_count(path).unwrap_or(1),
            _ => 1,
        };
        announced.push(frames);
        progress.frames_total += frames;
    }
    on_progress(&progress);

    let mut frames: Vec<Array2<f32>> = Vec::new();
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut dims: Option<(usize, usize)> = None;
    // TIFF frames follow the detector in force (guess or override); `.npy`
    // input is re-oriented only when the detector was chosen explicitly.
    let mut orientation = Orientation::Identity;

    for (idx, path) in sorted.iter().enumerate() {
        progress.current = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        on_progress(&progress);
        let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let bytes_before = progress.bytes_done;
        let loaded = match ext_of(path).as_str() {
            "tif" | "tiff" => {
                orientation = detector.orientation();
                load_tiff(path, orientation)?
            }
            "npy" => {
                if let Some(d) = detector.manual {
                    orientation = d.orientation();
                }
                let frames = load_npy(path, &mut |read| {
                    progress.bytes_done = bytes_before + read;
                    on_progress(&progress);
                })?;
                if orientation.is_identity() {
                    frames
                } else {
                    frames.into_iter().map(|f| orientation.apply(f)).collect()
                }
            }
            other => bail!("Unsupported file type '.{other}': {}", path.display()),
        };
        // A multi-page TIFF announces its real page count once read.
        progress.frames_total = progress.frames_total + loaded.len() - announced[idx];
        progress.frames_done += loaded.len();
        progress.bytes_done = bytes_before + file_bytes;
        progress.files_done = idx + 1;

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
        on_progress(&progress);
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
/// `on_read(bytes_so_far)` is called every few megabytes — a stack handed
/// over by NeCTAR is one multi-gigabyte file on a network filesystem.
fn load_npy(path: &Path, on_read: &mut dyn FnMut(u64)) -> Result<Vec<Array2<f32>>> {
    use ndarray_npy::ReadNpyExt;
    use std::io::{Cursor, Read};

    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let size = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
    let mut bytes: Vec<u8> = Vec::with_capacity(size);
    const CHUNK: usize = 8 << 20;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
        on_read(bytes.len() as u64);
    }
    drop(buf);

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
/// a CCD page is flipped vertically and horizontally, QHY is rotated 90° counterclockwise and unknown
/// detectors are kept as-is. The .npy path does not go through here: those
/// arrays are re-oriented in [`load_paths_with_progress`] only when the
/// detector was chosen explicitly.
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
    fn npy_header_announces_the_frames_and_progress_counts_bytes() {
        let dir = tmp_dir("npyprogress");
        let path = dir.join("stack.npy");
        let a = ndarray::Array3::<f32>::zeros((7, 4, 5));
        ndarray_npy::write_npy(&path, &a).unwrap();
        assert_eq!(npy_frame_count(&path), Some(7));
        let flat = dir.join("one.npy");
        ndarray_npy::write_npy(&flat, &Array2::<u16>::zeros((4, 5))).unwrap();
        assert_eq!(npy_frame_count(&flat), Some(1));

        let mut seen: Vec<LoadProgress> = Vec::new();
        let stack = load_paths_with_progress(&[path.clone()], Selection::default(), |p| {
            seen.push(p.clone())
        })
        .unwrap();
        assert_eq!(stack.n_frames(), 7);
        let first = seen.first().unwrap();
        assert_eq!((first.frames_total, first.files_total, first.files_done), (7, 1, 0));
        assert_eq!(first.bytes_total, std::fs::metadata(&path).unwrap().len());
        let last = seen.last().unwrap();
        assert_eq!((last.frames_done, last.files_done), (7, 1));
        assert_eq!(last.bytes_done, last.bytes_total);
        assert_eq!(last.fraction(), 1.0);
        assert_eq!(last.current, "stack.npy");
        assert_eq!(format_bytes(3_310_000_000), "3.31 GB");
        assert_eq!(format_bytes(8_400_000), "8.4 MB");
        let _ = std::fs::remove_dir_all(&dir);
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
    fn npy_is_reoriented_only_when_the_detector_is_explicit() {
        let dir = tmp_dir("npy_orientation");
        let path = dir.join("image.npy");
        // 3 wide × 2 tall: [1 2 3; 4 5 6]
        let a = Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        ndarray_npy::write_npy(&path, &a).unwrap();

        // No explicit detector: as-is, even if the path named one.
        let mut sel = Selection::from_path(&path);
        sel.auto = Some((Detector::Timepix, Source::Path));
        let stack = load_paths_with_progress(&[path.clone()], sel, |_| {}).unwrap();
        assert_eq!(stack.orientation, Orientation::Identity);
        assert_eq!((stack.height, stack.width), (2, 3));
        assert_eq!(stack.frames[0][(1, 0)], 4.0);

        // `--detector timepix`: transposed like a Timepix TIFF.
        let stack = load_paths_as(&[path.clone()], Detector::Timepix).unwrap();
        assert_eq!(stack.orientation, Orientation::Transpose);
        assert_eq!((stack.height, stack.width), (3, 2));
        assert_eq!(stack.frames[0][(1, 0)], 2.0);

        // A 3-D stack is re-oriented plane by plane.
        let stack3 = dir.join("stack.npy");
        let b = ndarray::Array3::from_shape_fn((2, 2, 3), |(k, y, x)| (k * 10 + y * 3 + x) as f32);
        ndarray_npy::write_npy(&stack3, &b).unwrap();
        let stack = load_paths_as(&[stack3], Detector::Timepix).unwrap();
        assert_eq!((stack.n_frames(), stack.height, stack.width), (2, 3, 2));
        assert_eq!(stack.frames[1][(2, 1)], 15.0);
        let _ = std::fs::remove_dir_all(&dir);
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

        // CCD: flipped vertically and horizontally, same size
        let stack = load_paths_as(&[path], Detector::Ccd).unwrap();
        assert_eq!(stack.orientation, Orientation::Rotate180);
        assert_eq!((stack.height, stack.width), (2, 3));
        // rotated 180°: [6 5 4; 3 2 1]
        assert_eq!(stack.frames[0][(0, 0)], 6.0);
    }
}
