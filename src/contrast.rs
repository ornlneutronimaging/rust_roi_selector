//! The display range (contrast) of the image and the ways to change it:
//! the min / max of the data, the 5–95 % range that leaves the outliers
//! out, and the colorbar as a control — dragging its top end moves the
//! maximum, its bottom end the minimum, its middle shifts both, the wheel
//! widens or narrows the range, a double-click restores the auto range.
//! The math is the one of nrad-gui's colorbar, so both tools feel the same.

use ndarray::Array2;

/// The fractions of the sorted pixel values that bound the auto range when
/// it leaves the outliers out: the 5th and 95th percentiles.
pub const AUTO_PERCENTILES: (f64, f64) = (0.05, 0.95);

/// The (min, max) of an image's finite values; (0, 1) when it has none, so
/// the result is always ordered and finite.
pub fn data_range(img: &Array2<f32>) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in img.iter() {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if lo <= hi {
        (lo, hi)
    } else {
        (0.0, 1.0)
    }
}

/// The values at the fractions `lo` and `hi` of the image's sorted finite
/// values (0.05, 0.95 = its 5th and 95th percentiles): a display range that
/// leaves the outliers out. Falls back to `data_range` when nothing is
/// finite or the two percentiles coincide (a flat image, or one where most
/// pixels share a value).
pub fn percentile_range(img: &Array2<f32>, lo: f64, hi: f64) -> (f32, f32) {
    let mut v: Vec<f32> = img.iter().copied().filter(|v| v.is_finite()).collect();
    if v.is_empty() {
        return data_range(img);
    }
    let n = v.len();
    let k_lo = ((lo.clamp(0.0, 1.0) * (n - 1) as f64).round() as usize).min(n - 1);
    let k_hi = ((hi.clamp(0.0, 1.0) * (n - 1) as f64).round() as usize).clamp(k_lo, n - 1);
    // Two selections instead of a sort: the high one first, then the low
    // one among the values left of it.
    let (left, high, _) = v.select_nth_unstable_by(k_hi, |a, b| a.total_cmp(b));
    let high = *high;
    let low = if k_lo < k_hi {
        *left.select_nth_unstable_by(k_lo, |a, b| a.total_cmp(b)).1
    } else {
        high
    };
    if low < high {
        (low, high)
    } else {
        data_range(img)
    }
}

/// Where the colorbar is grabbed: its top end moves the maximum, its bottom
/// end the minimum, the middle shifts both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarZone {
    Top,
    Middle,
    Bottom,
}

/// The zone at `frac` of the bar height (0 = top, 1 = bottom): the ends are
/// the outer quarters.
pub fn bar_zone(frac: f32) -> BarZone {
    if frac < 0.25 {
        BarZone::Top
    } else if frac > 0.75 {
        BarZone::Bottom
    } else {
        BarZone::Middle
    }
}

/// The smallest display span kept, as a fraction of the data range.
const MIN_SPAN_FRAC: f64 = 1e-4;

/// A data range edits can be bounded by: finite and ordered.
fn usable(data: (f64, f64)) -> bool {
    data.0.is_finite() && data.1.is_finite() && data.0 < data.1
}

/// The low end of a range of `span` that wants to start at `lo`: kept
/// inside `data` when it fits there, else sliding only as far as still
/// covers `data`.
fn slide_into(lo: f64, span: f64, data: (f64, f64)) -> f64 {
    let a = data.0;
    let b = data.1 - span;
    lo.clamp(a.min(b), a.max(b))
}

/// The display range after dragging `zone` of a bar spanning `lo` (bottom)
/// to `hi` (top) by `dy_frac` of the bar height (positive = downwards, i.e.
/// towards lower values): a drag of the whole height moves the end by the
/// whole span. The end being dragged stops at the data, the other end stays
/// where it is; a shift keeps the span and slides back into the data. The
/// edit is refused when it would leave no span.
pub fn colorbar_drag(lo: f64, hi: f64, zone: BarZone, dy_frac: f64, data: (f64, f64)) -> (f64, f64) {
    let span = hi - lo;
    if !usable(data) || span.is_nan() || span <= 0.0 || !dy_frac.is_finite() {
        return (lo, hi);
    }
    let d = -dy_frac * span;
    let (nlo, nhi) = match zone {
        BarZone::Top => (lo, (hi + d).clamp(data.0, data.1)),
        BarZone::Bottom => ((lo + d).clamp(data.0, data.1), hi),
        BarZone::Middle => {
            let nlo = slide_into(lo + d, span, data);
            (nlo, nlo + span)
        }
    };
    keep_span(lo, hi, nlo, nhi, data)
}

/// The display range with its span scaled by `factor` around its center
/// (the wheel over the bar), kept inside `data`: a span wider than the data
/// is capped at the data.
pub fn colorbar_scale(lo: f64, hi: f64, factor: f64, data: (f64, f64)) -> (f64, f64) {
    let span = hi - lo;
    if !usable(data) || span.is_nan() || span <= 0.0 || !factor.is_finite() || factor <= 0.0 {
        return (lo, hi);
    }
    let c = (lo + hi) / 2.0;
    let nspan = (span * factor).min(data.1 - data.0);
    let nlo = slide_into(c - nspan / 2.0, nspan, data);
    keep_span(lo, hi, nlo, nlo + nspan, data)
}

/// `(nlo, nhi)` unless they leave less than the minimum span, in which case
/// the previous range stands.
fn keep_span(lo: f64, hi: f64, nlo: f64, nhi: f64, data: (f64, f64)) -> (f64, f64) {
    let min_span = (data.1 - data.0).abs() * MIN_SPAN_FRAC;
    if nlo.is_finite() && nhi.is_finite() && nhi - nlo >= min_span && nlo < nhi {
        (nlo, nhi)
    } else {
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ranges_of_the_data_and_of_its_percentiles() {
        let mut v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        v.push(1.0e6); // one hot pixel
        v.push(-1.0e6); // one dead pixel
        v.push(f32::NAN);
        v.push(f32::INFINITY);
        let img = Array2::from_shape_vec((8, 13), v).unwrap();
        assert_eq!(data_range(&img), (-1.0e6, 1.0e6));
        assert_eq!(percentile_range(&img, 0.05, 0.95), (4.0, 95.0));
        assert_eq!(percentile_range(&img, 0.0, 1.0), (-1.0e6, 1.0e6));
        assert_eq!(percentile_range(&array![[5.0f32, 5.0], [5.0, 5.0]], 0.05, 0.95), (5.0, 5.0));
        assert_eq!(percentile_range(&array![[f32::NAN, f32::INFINITY]], 0.05, 0.95), (0.0, 1.0));
        assert_eq!(data_range(&Array2::<f32>::zeros((0, 0))), (0.0, 1.0));
    }

    #[test]
    fn colorbar_drags_move_one_end_or_shift_both_inside_the_data() {
        let data = (0.0, 100.0);
        assert_eq!(colorbar_drag(20.0, 60.0, BarZone::Top, 0.25, data), (20.0, 50.0));
        assert_eq!(colorbar_drag(20.0, 60.0, BarZone::Bottom, 0.5, data), (0.0, 60.0));
        assert_eq!(colorbar_drag(20.0, 60.0, BarZone::Middle, -0.5, data), (40.0, 80.0));
        // A shift stops at the data.
        assert_eq!(colorbar_drag(20.0, 60.0, BarZone::Middle, -2.0, data), (60.0, 100.0));
        assert_eq!(colorbar_drag(20.0, 60.0, BarZone::Middle, 2.0, data), (0.0, 40.0));
        // An end dragged past the other end is refused: the range stands.
        assert_eq!(colorbar_drag(20.0, 60.0, BarZone::Top, 1.5, data), (20.0, 60.0));
        // An end dragged past the data stops at the data.
        assert_eq!(colorbar_drag(20.0, 60.0, BarZone::Top, -2.0, data), (20.0, 100.0));
        assert_eq!(colorbar_drag(20.0, 60.0, BarZone::Top, f64::NAN, data), (20.0, 60.0));
        assert_eq!(bar_zone(0.1), BarZone::Top);
        assert_eq!(bar_zone(0.5), BarZone::Middle);
        assert_eq!(bar_zone(0.9), BarZone::Bottom);
    }

    #[test]
    fn the_wheel_scales_the_span_around_its_center() {
        let data = (0.0, 100.0);
        assert_eq!(colorbar_scale(20.0, 60.0, 0.5, data), (30.0, 50.0));
        assert_eq!(colorbar_scale(20.0, 60.0, 2.0, data), (0.0, 80.0));
        // Wider than the data: the data.
        assert_eq!(colorbar_scale(20.0, 60.0, 10.0, data), (0.0, 100.0));
        assert_eq!(colorbar_scale(20.0, 60.0, 0.0, data), (20.0, 60.0));
    }
}
