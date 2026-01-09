mod engine;
mod nn;
mod visualizer;

#[cfg(test)]
mod tests {
    use crate::{
        engine::*,
        nn::{Layer, MLP, Module, Neuron},
    };

    #[test]
    fn test_add() {
        let a = Value::new(2.0);
        let b = Value::new(1.0);
        let c = &a + &b;
        let d = &a + &c;
        assert_eq!(c.value(), 3.0);
        assert_eq!(d.value(), 5.0);
    }

    #[test]
    fn test_sub() {
        let a = Value::new(2.0);
        let b = Value::new(1.0);
        let c = &a - &b;
        assert_eq!(c.value(), 1.0);
    }

    #[test]
    fn test_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = &a * &b;
        assert_eq!(c.value(), 6.0);
    }

    #[test]
    fn test_debug_print() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = Value::new(10.0);
        let e = &a * &b;
        let d = &e + &c;
        d.draw();
    }

    #[test]
    fn test_tanh() {
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        let b = Value::new(6.7);

        let x1w1 = &x1 * &w1;
        let x2w2 = &x2 * &w2;
        let x1w1x2w2 = &(&x1w1 + &x2w2) + &b;
        let o = x1w1x2w2.tanh();
        o.draw();
    }

    #[test]
    fn test_backward() {
        let x = Value::new(2.0);
        let y = Value::new(3.0);

        let f = &x / &y;
        f.backward();
        assert!((x.0.borrow().grad - 0.33333333).abs() < 1e-6);
        assert!((y.0.borrow().grad - (-0.22222222)).abs() < 1e-6);

        f.draw();
    }

    #[test]
    fn test_neuron_call() {
        let x = vec![Value::new(2.0), Value::new(3.0)];
        let n = Neuron::new(2, true);
        let z = n.call(&x);
        z.backward();
        z.draw();
    }

    #[test]
    fn test_layer_call() {
        let x = vec![Value::new(2.0), Value::new(3.0)];
        let l = Layer::new(2, 3, true);
        let outs = l.call(&x);
        assert_eq!(outs.len(), 3);
        let loss = outs.iter().fold(Value::new(0.0), |acc, v| &acc + v);
        loss.backward();
        loss.draw();
    }

    #[test]
    fn test_mlp_and_params() {
        let x = vec![Value::new(2.0), Value::new(3.0)];
        let model = MLP::new(2, vec![4, 4, 1]);
        let out = model.call(x);

        assert_eq!(out.len(), 1);

        model.zero_grad();
        out[0].backward();

        let params = model.parameters();
        assert_eq!(params.len(), 37);

        out[0].draw();
    }

    #[test]
    fn test_grad_check() {
        let x = Value::new(1.234);
        let y = Value::new(-2.345);

        let f = (&x * &y).tanh();
        f.backward();

        let eps = 1e-6;
        let x_plus = Value::new(1.234 + eps);
        let x_minus = Value::new(1.234 - eps);

        let f_plus = (&x_plus * &y).tanh().value();
        let f_minus = (&x_minus * &y).tanh().value();

        let numerical = (f_plus - f_minus) / (2.0 * eps);
        let autodiff = x.0.borrow().grad;

        assert!((numerical - autodiff).abs() < 1e-3);
    }

    #[test]
    fn test_xor_training_showcase() {
        let model = MLP::new(2, vec![4, 4, 1]);
        let optimizer = SGD::new(model.parameters(), 0.1);

        // XOR dataset
        let inputs = vec![
            vec![Value::new(0.0), Value::new(0.0)],
            vec![Value::new(0.0), Value::new(1.0)],
            vec![Value::new(1.0), Value::new(0.0)],
            vec![Value::new(1.0), Value::new(1.0)],
        ];
        let targets = vec![0.0, 1.0, 1.0, 0.0];

        println!("Starting XOR Training...");

        for epoch in 0..100 {
            let mut total_loss = Value::new(0.0);

            for (x, y_true) in inputs.iter().zip(targets.iter()) {
                // Forward pass
                let y_pred = &model.call(x.clone())[0];

                // Mean Squared Error Loss: (pred - true)^2
                let diff = y_pred - &Value::new(*y_true);
                let loss = &diff * &diff;
                total_loss = &total_loss + &loss;
            }

            // Backward pass
            model.zero_grad();
            total_loss.backward();

            // Update weights
            optimizer.step();

            if epoch % 20 == 0 {
                println!("Epoch {}: Loss {:.4}", epoch, total_loss.value());
            }
        }

        // Final Verification
        for (x, y_true) in inputs.iter().zip(targets.iter()) {
            let pred = model.call(x.clone())[0].value();
            println!(
                "In: {:?} Target: {} Pred: {:.4}",
                x.iter().map(|v| v.value()).collect::<Vec<_>>(),
                y_true,
                pred
            );
            assert!((pred - y_true).abs() < 0.2);
        }

        // Visualize the final learned state of the computational graph
        let final_pred = &model.call(inputs[1].clone())[0];
        final_pred.draw();
    }
}
