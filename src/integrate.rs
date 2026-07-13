//! Combine a stack of frames into a single 2-D "integrated" image.

use crate::loader::ImageStack;
use ndarray::{Array2, Zip};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Integration {
    Sum,
    Mean,
    Max,
}

impl Integration {
    pub const ALL: [Integration; 3] = [Integration::Sum, Integration::Mean, Integration::Max];

    pub fn label(self) -> &'static str {
        match self {
            Integration::Sum => "Sum",
            Integration::Mean => "Mean",
            Integration::Max => "Max",
        }
    }
}

/// Collapse the stack into one image using the chosen reduction.
pub fn integrate(stack: &ImageStack, method: Integration) -> Array2<f32> {
    let (h, w) = (stack.height, stack.width);
    let n = stack.frames.len();

    match method {
        Integration::Sum | Integration::Mean => {
            let mut acc = Array2::<f32>::zeros((h, w));
            for frame in &stack.frames {
                acc += frame;
            }
            if method == Integration::Mean && n > 0 {
                acc /= n as f32;
            }
            acc
        }
        Integration::Max => {
            let mut acc = Array2::<f32>::from_elem((h, w), f32::NEG_INFINITY);
            for frame in &stack.frames {
                Zip::from(&mut acc).and(frame).for_each(|a, &v| {
                    if v > *a {
                        *a = v;
                    }
                });
            }
            acc
        }
    }
}
