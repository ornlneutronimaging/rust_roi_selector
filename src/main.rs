//! ROI Selector — view the integrated image of a stack (TIFF or NumPy), draw
//! regions of interest (rectangle, circle, ellipse), and export a mask file
//! where the selected pixels are 1 and everything else is 0.
//!
//! Designed to be launched either standalone or by an external tool (e.g. a
//! marimo notebook) that passes the input image and the mask destination on
//! the command line.

use ndarray::Array2;
use roi_selector::app::RoiApp;
use roi_selector::loader;
use std::path::PathBuf;

const USAGE: &str = "\
roi_selector — draw ROIs on an integrated image and export a 1/0 mask

USAGE:
  roi_selector [OPTIONS] [INPUT ...]

ARGS:
  INPUT   TIFF file(s), a folder of TIFF/.npy images, or a 2-D .npy image
          (e.g. an integrated image exported by a notebook). Several frames
          are combined (sum/mean/max) into the displayed image. When omitted,
          the data can be opened from within the application.

OPTIONS:
  -o, --output <PATH>     Mask file written by the 'Save mask & quit' button
                          (.tif/.tiff → 8-bit grayscale TIFF, .npy → uint8
                          NumPy array; ROI pixels = 1, others = 0)
  --called-from-python    The app is driven by another application (e.g. a
                          marimo notebook): the save button becomes
                          '↩ Return to main application', which writes the
                          mask to --output (required) and closes the window
                          so the caller can resume.
  --instructions <TEXT>   Instructions shown in a modal window on top of the
                          application at startup (e.g. what region the caller
                          expects to be selected). Reopen any time with the
                          'ℹ Instructions' toolbar button.
  --mask <PATH>           Existing mask file (.tif/.tiff/.npy, non-zero =
                          selected) shown as the starting selection, e.g. the
                          mask of a previous session to edit. Additive ROIs
                          add to it, subtract ROIs carve from it, and it is
                          included in the saved mask.
  --save-dir <PATH>       Starting folder of the 'Save mask as…' dialog
                          (default: the folder the displayed data came from).
                          Useful when the caller hands the data over via a
                          temporary file.
  --single-image          Open on the single-image view (slider through the
                          frames) instead of the integrated image
  -h, --help              Show this help
";

/// Read an existing mask file into a boolean array (non-zero = selected),
/// reusing the image loader so the same formats are accepted.
fn load_initial_mask(path: &PathBuf) -> Result<Array2<bool>, String> {
    let stack = loader::load_paths(&[path.clone()])
        .map_err(|e| format!("cannot read mask {}: {e:#}", path.display()))?;
    let frame = stack
        .frames
        .into_iter()
        .next()
        .ok_or_else(|| format!("no image data in mask {}", path.display()))?;
    Ok(frame.mapv(|v| v > 0.5))
}

type ParsedArgs = (
    Vec<PathBuf>,
    Option<PathBuf>,
    bool,
    Option<String>,
    Option<Array2<bool>>,
    Option<PathBuf>,
    bool,
);

fn parse_args() -> Result<ParsedArgs, String> {
    let mut inputs = Vec::new();
    let mut output = None;
    let mut called_from_python = false;
    let mut instructions = None;
    let mut initial_mask = None;
    let mut save_dir = None;
    let mut single_image = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "-o" | "--output" => {
                let path = args.next().ok_or("--output requires a path")?;
                output = Some(PathBuf::from(path));
            }
            "--called-from-python" | "--called_from_python" => called_from_python = true,
            "--save-dir" | "--save_dir" => {
                let path = args.next().ok_or("--save-dir requires a path")?;
                save_dir = Some(PathBuf::from(path));
            }
            "--single-image" | "--single_image" => single_image = true,
            "--instructions" => {
                let text = args.next().ok_or("--instructions requires a text argument")?;
                instructions = Some(text);
            }
            "--mask" => {
                let path = args.next().ok_or("--mask requires a path")?;
                initial_mask = Some(load_initial_mask(&PathBuf::from(path))?);
            }
            s if s.starts_with('-') => return Err(format!("Unknown option: {s}")),
            _ => inputs.push(PathBuf::from(a)),
        }
    }
    if called_from_python && output.is_none() {
        return Err("--called-from-python requires --output <PATH> (where the mask is returned)".to_owned());
    }
    if let Some(text) = &instructions {
        if text.trim().is_empty() {
            instructions = None;
        }
    }
    Ok((inputs, output, called_from_python, instructions, initial_mask, save_dir, single_image))
}

fn main() -> eframe::Result<()> {
    let (inputs, output, called_from_python, instructions, initial_mask, save_dir, single_image) = match parse_args() {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("Error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    // Expand folders to the supported image files they directly contain, so
    // errors (missing folder, no images) surface on stderr before the GUI opens.
    let mut files: Vec<PathBuf> = Vec::new();
    for input in inputs {
        if input.is_dir() {
            match loader::list_supported_in_dir(&input) {
                Ok(found) => files.extend(found),
                Err(e) => {
                    eprintln!("Error: {e:#}");
                    std::process::exit(1);
                }
            }
        } else {
            files.push(input);
        }
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 860.0])
            .with_title("VENUS ROI Selector"),
        ..Default::default()
    };

    eframe::run_native(
        "VENUS ROI Selector",
        native_options,
        Box::new(move |cc| {
            // Always use the dark theme, regardless of the system/desktop theme.
            cc.egui_ctx.set_theme(egui::Theme::Dark);
            let mut app = RoiApp::with_view(
                output,
                called_from_python,
                instructions,
                initial_mask,
                save_dir,
                !single_image,
            );
            if !files.is_empty() {
                app.start_load(files, &cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    )
}
