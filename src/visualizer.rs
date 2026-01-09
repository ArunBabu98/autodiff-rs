use crate::engine::Value;



pub struct GraphVisualizer {
    pub root: Value,
    pub centered: bool,
}
impl eframe::App for GraphVisualizer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Directed Acyclic Graph");

            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // Large virtual canvas
                    let canvas_size = egui::Vec2::new(3000.0, 2000.0);
                    let (rect, _) = ui.allocate_exact_size(canvas_size, egui::Sense::hover());

                    // Camera center
                    let origin = rect.center();

                    // Draw graph
                    let root_rect = self.root.render_node(ui, origin);

                    // 👇 Scroll ONCE to center the graph
                    if !self.centered {
                        ui.scroll_to_rect(root_rect, Some(egui::Align::Center));
                        self.centered = true;
                    }
                });
        });
    }
}
