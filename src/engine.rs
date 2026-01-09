use egui::{Color32, Pos2, Stroke, Ui, Vec2};
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

use crate::visualizer::GraphVisualizer;

#[derive(Debug)]
pub enum Ops {
    Add,
    Sub,
    Mul,
    Tanh,
    Exp,
    Log,
    Pow(f64),
    Relu,
}

pub struct Data {
    pub data: f64,
    pub grad: f64,
    pub parents: Vec<Value>,
    pub op: Option<Ops>,
    pub _backward: Option<Box<dyn Fn()>>,
}

impl Debug for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Data")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("op", &self.op)
            .finish()
    }
}

#[derive(Clone)]
pub struct Value(pub Rc<RefCell<Data>>);

impl Value {
    pub fn new(data: f64) -> Self {
        let data = Data {
            data,
            grad: 0.0,
            parents: vec![],
            op: None,
            _backward: None,
        };
        Value(Rc::new(RefCell::new(data)))
    }

    pub fn value(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn tanh(&self) -> Value {
        let x = self.0.borrow().data;
        let t = x.tanh();
        let input_node = self.clone();
        let new_data = Data {
            data: t,
            grad: 0.0,
            parents: vec![self.clone()],
            op: Some(Ops::Tanh),
            _backward: None,
        };
        let out = Value(Rc::new(RefCell::new(new_data)));
        let out_clone = out.clone();

        let backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad;
            let local_derivative = 1.0 - t * t;
            input_node.0.borrow_mut().grad += local_derivative * out_grad;
        });
        out.0.borrow_mut()._backward = Some(backward);
        out
    }

    pub fn relu(&self) -> Value {
        let x = self.0.borrow().data;
        let val = if x < 0.0 { 0.0 } else { x };
        let input_node = self.clone();
        let new_data = Data {
            data: val,
            grad: 0.0,
            parents: vec![self.clone()],
            op: Some(Ops::Relu),
            _backward: None,
        };
        let out = Value(Rc::new(RefCell::new(new_data)));
        let out_clone = out.clone();

        let backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad;
            let local_derivative = if x > 0.0 { 1.0 } else { 0.0 };
            input_node.0.borrow_mut().grad += local_derivative * out_grad;
        });
        out.0.borrow_mut()._backward = Some(backward);
        out
    }

    pub fn pow(&self, exponent: f64) -> Value {
        let x = self.0.borrow().data;
        let out_data = x.powf(exponent);
        let input_node = self.clone();
        let new_data = Data {
            data: out_data,
            grad: 0.0,
            parents: vec![self.clone()],
            op: Some(Ops::Pow(exponent)),
            _backward: None,
        };
        let out = Value(Rc::new(RefCell::new(new_data)));
        let out_clone = out.clone();

        let backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad;
            let local_derivative = exponent * x.powf(exponent - 1.0);
            input_node.0.borrow_mut().grad += local_derivative * out_grad;
        });
        out.0.borrow_mut()._backward = Some(backward);
        out
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn build_topo(
            v: &Value,
            visited: &mut std::collections::HashSet<*const Data>,
            topo: &mut Vec<Value>,
        ) {
            let ptr = v.0.as_ptr() as *const Data;
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for child in &v.0.borrow().parents {
                    build_topo(child, visited, topo);
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut visited, &mut topo);
        self.0.borrow_mut().grad = 1.0;
        for node in topo.iter().rev() {
            if let Some(ref backward_fn) = node.0.borrow()._backward {
                backward_fn();
            }
        }
    }
    pub fn exp(&self) -> Value {
        let x = self.0.borrow().data;
        let out_data = x.exp();
        let input_node = self.clone();
        let new_data = Data {
            data: out_data,
            grad: 0.0,
            parents: vec![self.clone()],
            op: Some(Ops::Exp),
            _backward: None,
        };
        let out = Value(Rc::new(RefCell::new(new_data)));
        let out_clone = out.clone();

        let backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad;
            input_node.0.borrow_mut().grad += out_data * out_grad;
        });
        out.0.borrow_mut()._backward = Some(backward);
        out
    }

    pub fn log(&self) -> Value {
        let x = self.0.borrow().data;
        let out_data = x.ln();
        let input_node = self.clone();
        let new_data = Data {
            data: out_data,
            grad: 0.0,
            parents: vec![self.clone()],
            op: Some(Ops::Log),
            _backward: None,
        };
        let out = Value(Rc::new(RefCell::new(new_data)));
        let out_clone = out.clone();

        let backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad;
            input_node.0.borrow_mut().grad += (1.0 / x) * out_grad;
        });
        out.0.borrow_mut()._backward = Some(backward);
        out
    }

    pub fn draw(&self) {
        let value_to_draw = self.clone();
        let native_options = eframe::NativeOptions {
            event_loop_builder: Some(Box::new(|builder| {
                #[cfg(target_os = "windows")]
                {
                    builder.with_any_thread(true);
                }
            })),
            viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
            ..Default::default()
        };

        let _ = eframe::run_native(
            "Value Graph",
            native_options,
            Box::new(|_cc| {
                Ok(Box::new(GraphVisualizer {
                    root: value_to_draw,
                    centered: false,
                }))
            }),
        );
    }

    pub fn render_node(&self, ui: &mut Ui, pos: Pos2) -> egui::Rect {
        let data = self.0.borrow();
        let box_size = Vec2::new(80.0, 50.0);
        let rect = egui::Rect::from_min_size(pos, box_size);

        ui.painter()
            .rect_filled(rect, 4.0, Color32::from_rgb(30, 30, 30));
        ui.painter().rect_stroke(
            rect,
            4.0,
            Stroke::new(1.0, Color32::WHITE),
            egui::StrokeKind::Outside,
        );

        let label = format!("{:.2}\ng: {:.2}", data.data, data.grad);
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(12.0),
            Color32::WHITE,
        );

        if let Some(ref op) = data.op {
            let op_center = pos + Vec2::new(-40.0, box_size.y / 2.0);
            let op_radius = 15.0;

            self.draw_arrow(
                ui,
                op_center + Vec2::new(op_radius, 0.0),
                rect.left_center(),
            );

            ui.painter()
                .circle_filled(op_center, op_radius, Color32::from_rgb(70, 70, 70));
            ui.painter()
                .circle_stroke(op_center, op_radius, Stroke::new(1.0, Color32::LIGHT_GRAY));

            let op_char = match op {
                Ops::Add => "+".to_string(),
                Ops::Sub => "-".to_string(),
                Ops::Mul => "*".to_string(),
                Ops::Tanh => "tanh".to_string(),
                Ops::Exp => "e".to_string(),
                Ops::Log => "log".to_string(),
                Ops::Pow(n) => format!("**{}", n),
                Ops::Relu => "ReLU".to_string(),
            };
            ui.painter().text(
                op_center,
                egui::Align2::CENTER_CENTER,
                op_char,
                egui::FontId::monospace(14.0),
                Color32::WHITE,
            );

            let mut child_y_offset = -40.0;
            for child in &data.parents {
                let child_pos = op_center + Vec2::new(-120.0, child_y_offset - (box_size.y / 2.0));
                let child_rect = child.render_node(ui, child_pos);
                self.draw_arrow(
                    ui,
                    child_rect.right_center(),
                    op_center - Vec2::new(op_radius, 0.0),
                );
                child_y_offset += 80.0;
            }
        }
        rect
    }

    fn draw_arrow(&self, ui: &mut Ui, start: Pos2, end: Pos2) {
        let stroke = Stroke::new(1.0, Color32::GRAY);
        ui.painter().line_segment([start, end], stroke);
        let vec = end - start;
        if vec.length() < 1.0 {
            return;
        }
        let base_angle = vec.angle();
        let tip = end;
        let arrow_angle = 0.5;
        let length = 10.0;
        let p1 = tip + Vec2::angled(base_angle + std::f32::consts::PI + arrow_angle) * length;
        let p2 = tip + Vec2::angled(base_angle + std::f32::consts::PI - arrow_angle) * length;
        ui.painter().line_segment([tip, p1], stroke);
        ui.painter().line_segment([tip, p2], stroke);
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.borrow().fmt(f)
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        &self * -1.0
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Add<&Value> for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Self::Output {
        let sum = self.0.borrow().data + rhs.0.borrow().data;
        let left = self.clone();
        let right = rhs.clone();
        let new_data = Data {
            data: sum,
            grad: 0.0,
            parents: vec![left, right],
            op: Some(Ops::Add),
            _backward: None,
        };
        let out = Value(Rc::new(RefCell::new(new_data)));
        let out_clone = out.clone();
        let left_node = self.clone();
        let right_node = rhs.clone();
        let backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad;
            left_node.0.borrow_mut().grad += out_grad;
            right_node.0.borrow_mut().grad += out_grad;
        });
        out.0.borrow_mut()._backward = Some(backward);
        out
    }
}

impl Add<f64> for &Value {
    type Output = Value;
    fn add(self, rhs: f64) -> Self::Output {
        self + &Value::new(rhs)
    }
}

impl Add<&Value> for f64 {
    type Output = Value;
    fn add(self, rhs: &Value) -> Self::Output {
        &Value::new(self) + rhs
    }
}

impl Sub<&Value> for &Value {
    type Output = Value;
    fn sub(self, rhs: &Value) -> Self::Output {
        self + &(-rhs)
    }
}

impl Mul<&Value> for &Value {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Self::Output {
        let product = self.0.borrow().data * rhs.0.borrow().data;
        let left = self.clone();
        let right = rhs.clone();
        let new_data = Data {
            data: product,
            grad: 0.0,
            parents: vec![left, right],
            op: Some(Ops::Mul),
            _backward: None,
        };
        let out = Value(Rc::new(RefCell::new(new_data)));
        let out_clone = out.clone();
        let left_node = self.clone();
        let right_node = rhs.clone();
        let backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad;
            let l_data = left_node.0.borrow().data;
            let r_data = right_node.0.borrow().data;
            left_node.0.borrow_mut().grad += r_data * out_grad;
            right_node.0.borrow_mut().grad += l_data * out_grad;
        });
        out.0.borrow_mut()._backward = Some(backward);
        out
    }
}

impl Mul<f64> for &Value {
    type Output = Value;
    fn mul(self, rhs: f64) -> Self::Output {
        self * &Value::new(rhs)
    }
}

impl Mul<&Value> for f64 {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Self::Output {
        &Value::new(self) * rhs
    }
}

impl Div<&Value> for &Value {
    type Output = Value;
    fn div(self, rhs: &Value) -> Self::Output {
        self * &rhs.pow(-1.0)
    }
}

pub struct SGD {
    pub params: Vec<Value>,
    pub lr: f64,
}

impl SGD {
    pub fn new(params: Vec<Value>, lr: f64) -> Self {
        Self { params, lr }
    }

    pub fn step(&self) {
        for p in &self.params {
            let mut data = p.0.borrow_mut();
            data.data -= self.lr * data.grad;
        }
    }
}
