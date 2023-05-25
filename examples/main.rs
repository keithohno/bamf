use std::vec;

use bamf::vector::Vector;
use bamf::{Layer, NeuralNetwork};

fn main() {
    repeat_nn_single_input();
}

fn repeat_nn_single_input() {
    let weights1 = Layer::random((3, 5), (-0.5, 1.0));
    let weights2 = Layer::random((5, 2), (-0.5, 1.0));
    let mut nn = NeuralNetwork::new(vec![weights1, weights2]);
    for _ in 0..20 {
        let input = Vector::from(vec![0.0, 1.0, 2.0]);
        let target = Vector::from(vec![0.0, 1.0]);
        let loss = nn.train(input, &target);
        println!("loss: {:?}", loss);
    }
}
