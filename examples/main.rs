use std::vec;

use bamf::matrix::Matrix;
use bamf::vector::Vector;
use bamf::{Layer, NeuralNetwork};

fn main() {
    repeat_nn_single_input();
}

fn repeat_nn_single_input() {
    let weights1 = Layer::new(
        Matrix::from(vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 1.0, 0.0, -1.0],
            vec![-2.5, -1.3, -1.1, -0.9, -0.7],
        ]),
        vec![5.4, 3.2, 1.0, 1.2, 3.4],
    );
    let weights2 = Layer::new(
        Matrix::from(vec![
            vec![1.0, 2.0],
            vec![0.5, 0.8],
            vec![-1.5, -0.5],
            vec![-2.0, -1.0],
            vec![3.0, 5.0],
        ]),
        vec![1.2, -1.8],
    );
    let mut nn = NeuralNetwork::new(vec![weights1, weights2], Vector::from(vec![0.0, 1.0]));
    for _ in 0..20 {
        let input = Vector::from(vec![0.0, 1.0, 2.0]);
        nn.train(input);
        println!("loss: {:?}", nn.loss);
    }
    println!("loss: {:?}", nn.loss);
    println!("weight1: {:?}", nn.layers[0].weights);
    println!("bias1: {:?}", nn.layers[0].biases);
    println!("weight2: {:?}", nn.layers[1].weights);
    println!("bias2: {:?}", nn.layers[1].biases);
    println!("output: {:?}", nn.intermediates.last().unwrap());
}
