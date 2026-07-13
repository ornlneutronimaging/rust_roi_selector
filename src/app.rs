//! The egui/eframe application: toolbar, image viewer, ROI table, and editing.
//!
//! ROIs are kept as a live, ordered list so they can be selected, edited,
//! toggled between add/subtract, and deleted from the side panel. The exported
//! mask (selected pixels = 1, everything else = 0) is the composite of the
//! list, applied in order.

use crate::colormap::Colormap;
use crate::integrate::{integrate, Integration};
use crate::loader::{self, ImageStack};
use crate::roi::{save_mask, Geometry, Tool};

use egui::{Color32, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use ndarray::Array2;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

const UNDO_DEPTH: usize = 24;
const HANDLE_HIT: f32 = 10.0;
const HANDLE_SIZE: f32 = 9.0;

/// Messages sent from the background loading thread to the UI.
enum LoadMsg {
    Progress { done: usize, total: usize },
    Done(anyhow::Result<ImageStack>),
}

/// A loading operation in flight on a background thread.
struct LoadJob {
    rx: Receiver<LoadMsg>,
    done: usize,
    total: usize,
}

/// One region of interest in the selection.
#[derive(Clone)]
struct Roi {
    id: u32,
    geom: Geometry,
    /// `true` adds to the selection, `false` carves it out.
    additive: bool,
}

impl Roi {
    fn kind(&self) -> &'static str {
        match self.geom {
            Geometry::Rect { .. } => "Rectangle",
            Geometry::Ellipse { .. } => "Ellipse",
            Geometry::Circle { .. } => "Circle",
        }
    }
}

/// A resize grab-point on the selected ROI.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Handle {
    Box { hx: i8, vy: i8 },
    Radius,
}

pub struct RoiApp {
    stack: Option<ImageStack>,
    loading: Option<LoadJob>,

    /// Mask destination given on the command line (`--output`); enables the
    /// "Save mask & quit" one-click workflow used when the app is driven by an
    /// external tool (e.g. a marimo notebook).
    output_path: Option<PathBuf>,

    // Integration / display.
    integration: Integration,
    integrated: Option<Array2<f32>>,
    data_min: f32,
    data_max: f32,
    vmin: f32,
    vmax: f32,
    colormap: Colormap,

    // Textures.
    img_tex: Option<TextureHandle>,
    mask_tex: Option<TextureHandle>,
    img_dirty: bool,
    mask_tex_dirty: bool,

    // ROI model.
    rois: Vec<Roi>,
    next_id: u32,
    selected: Option<u32>,
    selected_count: usize,
    undo: Vec<Vec<Roi>>,

    // Tools.
    tool: Tool,
    subtract: bool,

    // Interaction transients.
    drawing: bool,
    moving: bool,
    resizing: Option<Handle>,
    drag_changed: bool, // an undo snapshot was taken for the current drag
    move_last: Option<(f32, f32)>,
    drag_start: Option<(f32, f32)>,

    // View.
    scale: f32,
    fit_requested: bool,
    cursor: Option<(usize, usize, f32)>,

    status: String,
}

impl RoiApp {
    pub fn new(output_path: Option<PathBuf>) -> Self {
        let status = match &output_path {
            Some(p) => format!(
                "Draw the region(s) of interest, then 'Save mask & quit' writes the mask to {}",
                p.display()
            ),
            None => "Open a TIFF stack or a .npy image to begin.".to_owned(),
        };
        Self {
            stack: None,
            loading: None,
            output_path,
            integration: Integration::Sum,
            integrated: None,
            data_min: 0.0,
            data_max: 1.0,
            vmin: 0.0,
            vmax: 1.0,
            colormap: Colormap::Viridis,
            img_tex: None,
            mask_tex: None,
            img_dirty: false,
            mask_tex_dirty: false,
            rois: Vec::new(),
            next_id: 1,
            selected: None,
            selected_count: 0,
            undo: Vec::new(),
            tool: Tool::Rectangle,
            subtract: false,
            drawing: false,
            moving: false,
            resizing: None,
            drag_changed: false,
            move_last: None,
            drag_start: None,
            scale: 1.0,
            fit_requested: false,
            cursor: None,
            status,
        }
    }

    // ----- loading & integration -------------------------------------------

    pub fn start_load(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        if paths.is_empty() {
            return;
        }
        let total = paths.len();
        let (tx, rx) = std::sync::mpsc::channel();
        let progress_tx = tx.clone();
        let ctx = ctx.clone();

        std::thread::spawn(move || {
            let result = loader::load_paths_with_progress(&paths, |done, total| {
                let _ = progress_tx.send(LoadMsg::Progress { done, total });
                ctx.request_repaint();
            });
            let _ = tx.send(LoadMsg::Done(result));
            ctx.request_repaint();
        });

        self.loading = Some(LoadJob { rx, done: 0, total });
        self.status = format!("Loading {total} file(s)…");
    }

    fn poll_load(&mut self) {
        let mut result = None;
        if let Some(job) = &mut self.loading {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    LoadMsg::Progress { done, total } => {
                        job.done = done;
                        job.total = total;
                    }
                    LoadMsg::Done(res) => result = Some(res),
                }
            }
        }
        if let Some(res) = result {
            self.loading = None;
            match res {
                Ok(stack) => self.apply_stack(stack),
                Err(e) => self.status = format!("Load failed: {e:#}"),
            }
        }
    }

    fn apply_stack(&mut self, stack: ImageStack) {
        let n = stack.n_frames();
        let (w, h) = (stack.width, stack.height);
        let first = stack
            .sources
            .first()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.rois.clear();
        self.next_id = 1;
        self.selected = None;
        self.selected_count = 0;
        self.undo.clear();
        self.drawing = false;
        self.moving = false;
        self.resizing = None;
        self.stack = Some(stack);
        self.recompute_integration();
        self.fit_requested = true;
        self.mask_tex_dirty = true;
        self.status = format!("Loaded {n} frame(s), {w}×{h} px (from {first} …).");
    }

    fn recompute_integration(&mut self) {
        let Some(stack) = &self.stack else { return };
        let img = integrate(stack, self.integration);

        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in img.iter() {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            lo = 0.0;
            hi = 1.0;
        }
        self.data_min = lo;
        self.data_max = hi;
        self.vmin = lo;
        self.vmax = hi;
        self.integrated = Some(img);
        self.img_dirty = true;
    }

    // ----- ROI model ----------------------------------------------------------

    fn push_undo(&mut self) {
        self.undo.push(self.rois.clone());
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
    }

    fn undo(&mut self) {
        let Some(prev) = self.undo.pop() else { return };
        self.rois = prev;
        if let Some(sid) = self.selected {
            if !self.rois.iter().any(|r| r.id == sid) {
                self.selected = None;
            }
        }
        self.mask_tex_dirty = true;
    }

    fn clear_rois(&mut self) {
        if self.rois.is_empty() {
            return;
        }
        self.push_undo();
        self.rois.clear();
        self.selected = None;
        self.mask_tex_dirty = true;
    }

    /// Composite the ROI list (in order) into a boolean mask.
    fn composite_mask(&self) -> Array2<bool> {
        let Some(img) = &self.integrated else {
            return Array2::default((0, 0));
        };
        let (h, w) = (img.shape()[0], img.shape()[1]);
        let mut m = Array2::<bool>::default((h, w));
        for roi in &self.rois {
            roi.geom.stamp(&mut m, roi.additive);
        }
        m
    }

    fn selected_geom(&self) -> Option<Geometry> {
        let id = self.selected?;
        self.rois.iter().find(|r| r.id == id).map(|r| r.geom)
    }

    fn selected_geom_mut(&mut self) -> Option<&mut Geometry> {
        let id = self.selected?;
        self.rois.iter_mut().find(|r| r.id == id).map(|r| &mut r.geom)
    }

    /// Topmost (last-drawn) ROI containing an image point.
    fn roi_at(&self, p: (f32, f32)) -> Option<u32> {
        self.rois
            .iter()
            .rev()
            .find(|r| r.geom.contains(p.0, p.1))
            .map(|r| r.id)
    }

    fn delete_selected(&mut self) {
        if let Some(sid) = self.selected {
            self.push_undo();
            self.rois.retain(|r| r.id != sid);
            self.selected = None;
            self.mask_tex_dirty = true;
        }
    }

    // ----- textures ---------------------------------------------------------

    fn ensure_img_texture(&mut self, ctx: &egui::Context) {
        if !self.img_dirty {
            return;
        }
        let Some(img) = &self.integrated else { return };
        let (h, w) = (img.shape()[0], img.shape()[1]);
        let span = (self.vmax - self.vmin).max(1e-12);
        let lut = self.colormap.lut();
        let mut buf = vec![0u8; w * h * 4];
        for (i, &v) in img.iter().enumerate() {
            let t = ((v - self.vmin) / span).clamp(0.0, 1.0);
            let idx = ((t * 255.0).round() as usize).min(255);
            let [r, g, b] = lut[idx];
            buf[i * 4] = r;
            buf[i * 4 + 1] = g;
            buf[i * 4 + 2] = b;
            buf[i * 4 + 3] = 255;
        }
        let color = egui::ColorImage::from_rgba_unmultiplied([w, h], &buf);
        self.img_tex = Some(ctx.load_texture("integrated", color, TextureOptions::NEAREST));
        self.img_dirty = false;
    }

    fn ensure_mask_texture(&mut self, ctx: &egui::Context) {
        if !self.mask_tex_dirty {
            return;
        }
        let Some(img) = &self.integrated else {
            self.mask_tex = None;
            self.selected_count = 0;
            self.mask_tex_dirty = false;
            return;
        };
        let (h, w) = (img.shape()[0], img.shape()[1]);
        let mask = self.composite_mask();
        let mut buf = vec![0u8; w * h * 4];
        let mut count = 0usize;
        for (i, &m) in mask.iter().enumerate() {
            if m {
                count += 1;
                buf[i * 4] = 40;
                buf[i * 4 + 1] = 220;
                buf[i * 4 + 2] = 120;
                buf[i * 4 + 3] = 100;
            }
        }
        self.selected_count = count;
        let color = egui::ColorImage::from_rgba_unmultiplied([w, h], &buf);
        self.mask_tex = Some(ctx.load_texture("mask", color, TextureOptions::NEAREST));
        self.mask_tex_dirty = false;
    }

    // ----- file dialogs -----------------------------------------------------

    fn open_files_dialog(&mut self, ctx: &egui::Context) {
        if let Some(files) = rfd::FileDialog::new()
            .add_filter("Images", loader::SUPPORTED_EXTENSIONS)
            .set_title("Open TIFF / .npy image(s)")
            .pick_files()
        {
            self.start_load(files, ctx);
        }
    }

    fn open_folder_dialog(&mut self, ctx: &egui::Context) {
        if let Some(dir) = rfd::FileDialog::new()
            .set_title("Open a folder of TIFF / .npy images")
            .pick_folder()
        {
            match loader::list_supported_in_dir(&dir) {
                Ok(files) => self.start_load(files, ctx),
                Err(e) => self.status = format!("{e:#}"),
            }
        }
    }

    /// Write the mask to `path`. Returns whether the write succeeded.
    fn write_mask(&mut self, path: &std::path::Path) -> bool {
        let mask = self.composite_mask();
        let count = mask.iter().filter(|&&b| b).count();
        match save_mask(path, &mask) {
            Ok(()) => {
                self.status = format!(
                    "Saved mask ({count} px = 1) to {}",
                    path.display()
                );
                true
            }
            Err(e) => {
                self.status = format!("Save failed: {e:#}");
                false
            }
        }
    }

    fn save_mask_dialog(&mut self) {
        if self.integrated.is_none() {
            self.status = "Nothing to save — load an image first.".to_owned();
            return;
        }
        if self.selected_count == 0 {
            self.status = "Nothing to save — draw at least one region first.".to_owned();
            return;
        }
        let default_name = self
            .output_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "mask.tif".to_owned());
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("TIFF mask", &["tif", "tiff"])
            .add_filter("NumPy mask", &["npy"])
            .set_file_name(default_name)
            .set_title("Save ROI mask (1 inside the regions, 0 outside)")
            .save_file()
        {
            self.write_mask(&path);
        }
    }

    // ----- UI ---------------------------------------------------------------

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let busy = self.loading.is_some();
            let ctx = ui.ctx().clone();
            if ui
                .add_enabled(!busy, egui::Button::new("📂 Open Files…"))
                .clicked()
            {
                self.open_files_dialog(&ctx);
            }
            if ui
                .add_enabled(!busy, egui::Button::new("📁 Open Folder…"))
                .clicked()
            {
                self.open_folder_dialog(&ctx);
            }

            ui.separator();

            let n_frames = self.stack.as_ref().map(|s| s.n_frames()).unwrap_or(0);
            let mut changed = false;
            ui.add_enabled_ui(n_frames > 1, |ui| {
                egui::ComboBox::from_id_salt("integration")
                    .selected_text(format!("Integrate: {}", self.integration.label()))
                    .show_ui(ui, |ui| {
                        for m in Integration::ALL {
                            changed |= ui
                                .selectable_value(&mut self.integration, m, m.label())
                                .changed();
                        }
                    });
            });
            if changed {
                self.recompute_integration();
            }

            ui.separator();

            ui.label("Contrast:");
            let range = self.data_min..=self.data_max;
            let speed = (self.data_max - self.data_min).max(1.0) / 200.0;
            let r1 = ui.add(
                egui::DragValue::new(&mut self.vmin)
                    .speed(speed)
                    .range(range.clone()),
            );
            let r2 = ui.add(egui::DragValue::new(&mut self.vmax).speed(speed).range(range));
            if ui.button("Auto").clicked() {
                self.vmin = self.data_min;
                self.vmax = self.data_max;
                self.img_dirty = true;
            }
            if r1.changed() || r2.changed() {
                self.img_dirty = true;
            }

            ui.separator();

            let mut cmap_changed = false;
            egui::ComboBox::from_id_salt("colormap")
                .selected_text(format!("Colormap: {}", self.colormap.label()))
                .show_ui(ui, |ui| {
                    for c in Colormap::ALL {
                        cmap_changed |= ui
                            .selectable_value(&mut self.colormap, c, c.label())
                            .changed();
                    }
                });
            if cmap_changed {
                self.img_dirty = true;
            }
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Tool:");
            for t in [Tool::Rectangle, Tool::Ellipse, Tool::Circle] {
                ui.selectable_value(&mut self.tool, t, t.label());
            }
            ui.separator();
            ui.checkbox(&mut self.subtract, "Subtract")
                .on_hover_text("New regions carve out of the selection instead of adding");

            ui.separator();
            if ui
                .add_enabled(!self.undo.is_empty(), egui::Button::new("↩ Undo"))
                .clicked()
            {
                self.undo();
            }
            if ui.button("🗑 Clear").clicked() {
                self.clear_rois();
            }

            ui.separator();
            ui.label("Zoom:");
            if ui.button("−").clicked() {
                self.scale = (self.scale / 1.25).max(0.02);
            }
            if ui.button("+").clicked() {
                self.scale = (self.scale * 1.25).min(64.0);
            }
            if ui.button("Fit").clicked() {
                self.fit_requested = true;
            }
            ui.label(format!("{:.0}%", self.scale * 100.0));
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Save buttons sit at the far right; lay them out first so the
            // status text on the left can take the remaining width.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let can_save = self.integrated.is_some() && self.selected_count > 0;
                if let Some(out) = self.output_path.clone() {
                    if ui
                        .add_enabled(can_save, egui::Button::new("✅ Save mask && quit"))
                        .on_hover_text(format!("Write the mask to {} and close", out.display()))
                        .clicked()
                        && self.write_mask(&out)
                    {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
                if ui
                    .add_enabled(can_save, egui::Button::new("💾 Save mask as…"))
                    .on_hover_text("Write the 1/0 mask to a .tif or .npy file of your choice")
                    .clicked()
                {
                    self.save_mask_dialog();
                }
                ui.separator();
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.label(&self.status);
                    ui.separator();
                    ui.label(format!("selected: {} px", self.selected_count));
                    ui.separator();
                    ui.label(format!("ROIs: {}", self.rois.len()));
                    if let Some((x, y, v)) = self.cursor {
                        ui.separator();
                        ui.label(format!("({x}, {y}) = {v:.4}"));
                    }
                });
            });
        });
    }

    /// The right-hand ROI table: select, edit, toggle add/subtract, delete.
    fn roi_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Regions of Interest");
        ui.add_space(4.0);
        if self.rois.is_empty() {
            ui.label("Draw a rectangle, ellipse, or circle on the image to add it here.");
            return;
        }
        ui.label("Click a row to select; edit values, toggle +/−, or delete.");
        ui.separator();

        let selected = self.selected;
        let mut new_selected = self.selected;
        let mut to_remove: Option<u32> = None;
        let mut changed = false;

        egui::ScrollArea::vertical().show(ui, |ui| {
            for roi in self.rois.iter_mut() {
                let id = roi.id;
                let is_sel = selected == Some(id);
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(is_sel, format!("{} #{id}", roi.kind()))
                            .clicked()
                        {
                            new_selected = Some(id);
                        }
                        let sign = if roi.additive { "＋ add" } else { "− sub" };
                        if ui
                            .small_button(sign)
                            .on_hover_text("Toggle add / subtract")
                            .clicked()
                        {
                            roi.additive = !roi.additive;
                            changed = true;
                        }
                        if ui.small_button("Delete").clicked() {
                            to_remove = Some(id);
                        }
                    });
                    if is_sel {
                        ui.add_space(2.0);
                        if edit_geometry(ui, id, &mut roi.geom) {
                            changed = true;
                        }
                    }
                });
            }
        });

        self.selected = new_selected;
        if let Some(rid) = to_remove {
            self.push_undo();
            self.rois.retain(|r| r.id != rid);
            if self.selected == Some(rid) {
                self.selected = None;
            }
            changed = true;
        }
        if changed {
            self.mask_tex_dirty = true;
        }
    }

    fn viewer(&mut self, ui: &mut egui::Ui) {
        if let Some(job) = &self.loading {
            let frac = if job.total > 0 {
                job.done as f32 / job.total as f32
            } else {
                0.0
            };
            ui.centered_and_justified(|ui| {
                ui.add_sized(
                    [320.0, 24.0],
                    egui::ProgressBar::new(frac)
                        .show_percentage()
                        .text(format!("⏳ Loading {} / {} files", job.done, job.total)),
                );
            });
            return;
        }

        let Some(img) = &self.integrated else {
            ui.centered_and_justified(|ui| {
                ui.label("No image loaded.");
            });
            return;
        };
        let (h, w) = (img.shape()[0], img.shape()[1]);

        if self.fit_requested && w > 0 && h > 0 {
            let avail = ui.available_size();
            let s = (avail.x / w as f32).min(avail.y / h as f32);
            self.scale = s.clamp(0.02, 64.0);
            self.fit_requested = false;
        }

        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let size = egui::vec2(w as f32 * self.scale, h as f32 * self.scale);
                let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());

                let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                let painter = ui.painter_at(rect);
                if let Some(t) = &self.img_tex {
                    painter.image(t.id(), rect, full_uv, Color32::WHITE);
                }
                if let Some(t) = &self.mask_tex {
                    painter.image(t.id(), rect, full_uv, Color32::WHITE);
                }

                self.handle_interaction(&painter, rect, &response, w, h);
            });
    }

    fn handle_interaction(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        response: &egui::Response,
        w: usize,
        h: usize,
    ) {
        let scale = self.scale;
        let to_img =
            |p: Pos2| -> (f32, f32) { ((p.x - rect.left()) / scale, (p.y - rect.top()) / scale) };
        let to_screen =
            |ix: f32, iy: f32| -> Pos2 { Pos2::new(rect.left() + ix * scale, rect.top() + iy * scale) };

        // Cursor read-out.
        self.cursor = None;
        if let Some(p) = response.hover_pos() {
            let (ix, iy) = to_img(p);
            let (xi, yi) = (ix.floor() as i64, iy.floor() as i64);
            if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
                let v = self
                    .integrated
                    .as_ref()
                    .map(|m| m[(yi as usize, xi as usize)])
                    .unwrap_or(0.0);
                self.cursor = Some((xi as usize, yi as usize, v));
            }
        }

        // Resize a handle, move the region under the cursor, or draw a new one.
        // Plain click (no drag): select the ROI under the cursor, or deselect.
        if response.clicked() {
            if let Some(p) = response.interact_pointer_pos() {
                self.selected = self.roi_at(to_img(p));
            }
        }

        if response.drag_started() {
            self.drag_changed = false;
            let press = painter
                .ctx()
                .input(|i| i.pointer.press_origin())
                .or_else(|| response.interact_pointer_pos());
            let cur = response.interact_pointer_pos().map(to_img);
            if let Some(sp) = press {
                let start = to_img(sp);
                let mut acted = false;

                // 1) A resize handle on the currently selected ROI.
                if let Some(g) = self.selected_geom() {
                    if let Some(hd) = shape_handles(&g).into_iter().find_map(|(hd, (ix, iy))| {
                        (to_screen(ix, iy).distance(sp) <= HANDLE_HIT).then_some(hd)
                    }) {
                        self.resizing = Some(hd);
                        acted = true;
                    } else if g.contains(start.0, start.1) {
                        self.moving = true;
                        self.move_last = cur.or(Some(start));
                        acted = true;
                    }
                }

                // 2) Otherwise grab whichever ROI is under the cursor.
                if !acted {
                    if let Some(rid) = self.roi_at(start) {
                        self.selected = Some(rid);
                        self.moving = true;
                        self.move_last = cur.or(Some(start));
                        acted = true;
                    }
                }

                // 3) Otherwise start drawing a new ROI (snapshot now, before it
                //    exists, so undo removes it).
                if !acted {
                    self.push_undo();
                    self.drag_changed = true;
                    let id = self.next_id;
                    self.next_id += 1;
                    let geom = Geometry::from_drag(self.tool, start, start);
                    self.rois.push(Roi {
                        id,
                        geom,
                        additive: !self.subtract,
                    });
                    self.selected = Some(id);
                    self.drawing = true;
                    self.drag_start = Some(start);
                }
            }
        }

        if response.dragged() {
            let cur = response.interact_pointer_pos().map(to_img);
            // Snapshot once, on the first real movement of a move/resize.
            if (self.moving || self.resizing.is_some()) && !self.drag_changed {
                self.push_undo();
                self.drag_changed = true;
            }
            if let Some(hd) = self.resizing {
                if let (Some(c), Some(g)) = (cur, self.selected_geom_mut()) {
                    resize_geom(g, hd, c);
                    self.mask_tex_dirty = true;
                }
            } else if self.moving {
                if let (Some(last), Some(c)) = (self.move_last, cur) {
                    if let Some(g) = self.selected_geom_mut() {
                        g.translate(c.0 - last.0, c.1 - last.1);
                        self.mask_tex_dirty = true;
                    }
                    self.move_last = cur;
                }
            } else if self.drawing {
                if let (Some(a), Some(c)) = (self.drag_start, cur) {
                    let new = Geometry::from_drag(self.tool, a, c);
                    if let Some(g) = self.selected_geom_mut() {
                        *g = new;
                        self.mask_tex_dirty = true;
                    }
                }
            }
        }

        if response.drag_stopped() {
            if self.drawing {
                // Drop a region that never grew beyond a click.
                if let Some(a) = self.drag_start {
                    let c = response.interact_pointer_pos().map(to_img).unwrap_or(a);
                    let dist = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
                    if dist < 1.0 {
                        if let Some(sid) = self.selected {
                            self.rois.retain(|r| r.id != sid);
                            self.selected = None;
                            self.undo.pop(); // discard the snapshot for the no-op
                        }
                    }
                }
            }
            self.drawing = false;
            self.moving = false;
            self.resizing = None;
            self.drag_changed = false;
            self.move_last = None;
            self.drag_start = None;
        }

        // Hover cursor: resize over a handle, grab inside the selected ROI.
        if !response.dragged() {
            if let (Some(hp), Some(g)) = (response.hover_pos(), self.selected_geom()) {
                let over = shape_handles(&g).into_iter().find_map(|(hd, (ix, iy))| {
                    (to_screen(ix, iy).distance(hp) <= HANDLE_HIT).then_some(hd)
                });
                if let Some(hd) = over {
                    painter.ctx().set_cursor_icon(cursor_for_handle(hd));
                } else {
                    let (ix, iy) = to_img(hp);
                    if g.contains(ix, iy) {
                        painter.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    }
                }
            }
        }

        // Delete the selected ROI with Delete/Backspace (unless typing).
        if self.selected.is_some()
            && !painter.ctx().egui_wants_keyboard_input()
            && painter.ctx().input(|i| {
                i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)
            })
        {
            self.delete_selected();
        }

        // Draw every ROI; the selected one shows handles.
        for roi in &self.rois {
            let sel = self.selected == Some(roi.id);
            draw_roi(painter, &roi.geom, roi.additive, sel, &to_screen, scale);
        }
    }
}

/// Numeric editor for one ROI's geometry. Returns whether anything changed.
fn edit_geometry(ui: &mut egui::Ui, id: u32, geom: &mut Geometry) -> bool {
    let mut changed = false;
    egui::Grid::new(("roi_grid", id))
        .num_columns(2)
        .spacing([6.0, 2.0])
        .show(ui, |ui| match geom {
            Geometry::Rect { x0, y0, x1, y1 } => {
                let mut x = (*x0).min(*x1);
                let mut y = (*y0).min(*y1);
                let mut w = (*x1 - *x0).abs();
                let mut hh = (*y1 - *y0).abs();
                changed |= field(ui, "X", &mut x);
                changed |= field(ui, "Y", &mut y);
                changed |= field(ui, "Width", &mut w);
                changed |= field(ui, "Height", &mut hh);
                if changed {
                    *x0 = x;
                    *y0 = y;
                    *x1 = x + w.max(0.0);
                    *y1 = y + hh.max(0.0);
                }
            }
            Geometry::Ellipse { cx, cy, rx, ry } => {
                changed |= field(ui, "Center X", cx);
                changed |= field(ui, "Center Y", cy);
                changed |= field(ui, "Radius X", rx);
                changed |= field(ui, "Radius Y", ry);
                if changed {
                    *rx = (*rx).max(0.0);
                    *ry = (*ry).max(0.0);
                }
            }
            Geometry::Circle { cx, cy, r } => {
                changed |= field(ui, "Center X", cx);
                changed |= field(ui, "Center Y", cy);
                changed |= field(ui, "Radius", r);
                if changed {
                    *r = (*r).max(0.0);
                }
            }
        });
    changed
}

fn field(ui: &mut egui::Ui, label: &str, v: &mut f32) -> bool {
    ui.label(label);
    let changed = ui.add(egui::DragValue::new(v).speed(0.5)).changed();
    ui.end_row();
    changed
}

/// Image-space positions of every resize handle for a geometry.
fn shape_handles(geom: &Geometry) -> Vec<(Handle, (f32, f32))> {
    match *geom {
        Geometry::Circle { cx, cy, r } => vec![
            (Handle::Radius, (cx + r, cy)),
            (Handle::Radius, (cx - r, cy)),
            (Handle::Radius, (cx, cy + r)),
            (Handle::Radius, (cx, cy - r)),
        ],
        _ => {
            let (minx, miny, maxx, maxy) = shape_bounds(geom).unwrap();
            let (midx, midy) = ((minx + maxx) * 0.5, (miny + maxy) * 0.5);
            let mut out = Vec::with_capacity(8);
            for vy in [-1i8, 0, 1] {
                for hx in [-1i8, 0, 1] {
                    if hx == 0 && vy == 0 {
                        continue;
                    }
                    let x = match hx {
                        -1 => minx,
                        1 => maxx,
                        _ => midx,
                    };
                    let y = match vy {
                        -1 => miny,
                        1 => maxy,
                        _ => midy,
                    };
                    out.push((Handle::Box { hx, vy }, (x, y)));
                }
            }
            out
        }
    }
}

fn shape_bounds(geom: &Geometry) -> Option<(f32, f32, f32, f32)> {
    match *geom {
        Geometry::Rect { x0, y0, x1, y1 } => Some((x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1))),
        Geometry::Ellipse { cx, cy, rx, ry } => Some((cx - rx, cy - ry, cx + rx, cy + ry)),
        Geometry::Circle { .. } => None,
    }
}

fn resize_geom(geom: &mut Geometry, handle: Handle, p: (f32, f32)) {
    match handle {
        Handle::Radius => {
            if let Geometry::Circle { cx, cy, r } = geom {
                *r = ((p.0 - *cx).powi(2) + (p.1 - *cy).powi(2)).sqrt().max(0.5);
            }
        }
        Handle::Box { hx, vy } => {
            if let Some((mut minx, mut miny, mut maxx, mut maxy)) = shape_bounds(geom) {
                match hx {
                    -1 => minx = p.0,
                    1 => maxx = p.0,
                    _ => {}
                }
                match vy {
                    -1 => miny = p.1,
                    1 => maxy = p.1,
                    _ => {}
                }
                let (x0, x1) = (minx.min(maxx), minx.max(maxx));
                let (y0, y1) = (miny.min(maxy), miny.max(maxy));
                *geom = match *geom {
                    Geometry::Rect { .. } => Geometry::Rect { x0, y0, x1, y1 },
                    Geometry::Ellipse { .. } => Geometry::Ellipse {
                        cx: (x0 + x1) * 0.5,
                        cy: (y0 + y1) * 0.5,
                        rx: (x1 - x0) * 0.5,
                        ry: (y1 - y0) * 0.5,
                    },
                    other => other,
                };
            }
        }
    }
}

fn cursor_for_handle(handle: Handle) -> egui::CursorIcon {
    match handle {
        Handle::Radius => egui::CursorIcon::ResizeHorizontal,
        Handle::Box { hx: 0, .. } => egui::CursorIcon::ResizeVertical,
        Handle::Box { vy: 0, .. } => egui::CursorIcon::ResizeHorizontal,
        Handle::Box { hx, vy } => {
            if hx == vy {
                egui::CursorIcon::ResizeNwSe
            } else {
                egui::CursorIcon::ResizeNeSw
            }
        }
    }
}

/// Draw one ROI outline; if `selected`, add a translucent fill and handles.
fn draw_roi(
    painter: &egui::Painter,
    geom: &Geometry,
    additive: bool,
    selected: bool,
    to_screen: &dyn Fn(f32, f32) -> Pos2,
    scale: f32,
) {
    let (outline, fill) = if additive {
        (
            Color32::from_rgb(0, 220, 160),
            Color32::from_rgba_unmultiplied(0, 220, 160, 60),
        )
    } else {
        (
            Color32::from_rgb(255, 150, 40),
            Color32::from_rgba_unmultiplied(255, 150, 40, 60),
        )
    };
    let stroke = Stroke::new(if selected { 2.0 } else { 1.0 }, outline);

    match *geom {
        Geometry::Rect { x0, y0, x1, y1 } => {
            let r = Rect::from_two_pos(to_screen(x0, y0), to_screen(x1, y1));
            if selected {
                painter.rect_filled(r, egui::CornerRadius::ZERO, fill);
            }
            painter.rect_stroke(r, egui::CornerRadius::ZERO, stroke, egui::StrokeKind::Middle);
        }
        Geometry::Circle { cx, cy, r } => {
            let c = to_screen(cx, cy);
            if selected {
                painter.circle_filled(c, r * scale, fill);
            }
            painter.circle_stroke(c, r * scale, stroke);
        }
        Geometry::Ellipse { cx, cy, rx, ry } => {
            let n = 64;
            let pts: Vec<Pos2> = (0..n)
                .map(|i| {
                    let a = std::f32::consts::TAU * i as f32 / n as f32;
                    to_screen(cx + rx * a.cos(), cy + ry * a.sin())
                })
                .collect();
            if selected {
                painter.add(egui::Shape::convex_polygon(pts.clone(), fill, stroke));
            } else {
                painter.add(egui::Shape::closed_line(pts, stroke));
            }
        }
    }

    if selected {
        let (cx, cy) = geom.center();
        let c = to_screen(cx, cy);
        painter.circle_filled(c, 4.0, outline);
        painter.circle_stroke(c, 4.0, Stroke::new(1.0, Color32::BLACK));

        for (_hd, (ix, iy)) in shape_handles(geom) {
            let r = Rect::from_center_size(to_screen(ix, iy), egui::vec2(HANDLE_SIZE, HANDLE_SIZE));
            painter.rect_filled(r, egui::CornerRadius::ZERO, Color32::WHITE);
            painter.rect_stroke(
                r,
                egui::CornerRadius::ZERO,
                Stroke::new(1.0, Color32::BLACK),
                egui::StrokeKind::Middle,
            );
        }
    }
}

impl eframe::App for RoiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_load();
        if self.loading.is_some() {
            ctx.request_repaint();
        }

        self.ensure_img_texture(&ctx);
        self.ensure_mask_texture(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            self.toolbar(ui);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            self.status_bar(ui);
        });
        egui::Panel::right("rois")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| {
                self.roi_panel(ui);
            });
        egui::CentralPanel::default().show(ui, |ui| {
            self.viewer(ui);
        });
    }
}
