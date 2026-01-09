
use rand::Rng;

use crate::engine::Value;

pub trait Module {
    fn parameters(&self) -> Vec<Value>;
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.0.borrow_mut().grad = 0.0;
        }
    }
}

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: u64, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();
        let w: Vec<Value> = (0..nin)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0)))
            .collect();
        let b = Value::new(0.0);
        Self { w, b, nonlin }
    }

    pub fn call(&self, x: &[Value]) -> Value {
        let act = self
            .w
            .iter()
            .zip(x.iter())
            .map(|(wi, xi)| wi * xi)
            .fold(self.b.clone(), |acc, val| &acc + &val);

        if self.nonlin { act.tanh() } else { act }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut p = self.w.clone();
        p.push(self.b.clone());
        p
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: u64, nout: u64, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, nonlin)).collect();
        Self { neurons }
    }

    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: u64, nouts: Vec<u64>) -> Self {
        let mut sz = vec![nin];
        sz.extend(&nouts);
        let layers = (0..nouts.len())
            .map(|i| Layer::new(sz[i], sz[i + 1], i != nouts.len() - 1))
            .collect();
        Self { layers }
    }

    pub fn call(&self, mut x: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            x = layer.call(&x);
        }
        x
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
