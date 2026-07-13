//! ROI Selector library: stack loading, integration, ROI geometry and mask export.
//!
//! The GUI binary (`main.rs`) is a thin shell around these modules; they are
//! exposed here so they can be unit/integration tested without a display.

pub mod app;
pub mod colormap;
pub mod integrate;
pub mod loader;
pub mod roi;
