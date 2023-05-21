use std::vec;

use bamf::matrix::{Matrix, Multiply, Scale};
use bamf::vector::Vector;
use bamf::{Layer, NeuralNetwork};

fn main() {
    let input = Vector::from(vec![0.0, 1.0, 2.0]);
    let weights = Layer::new(
        Matrix::new(
            vec![3, 2],
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        ),
        Vector::from(vec![5.0, 0.0]),
    );
    let output = weights.forward(&input);
    let softmaxed = output.softmax();
    let bad_target = Vector::from(vec![0.5, 0.5]);
    let good_target = Vector::from(vec![0.01, 0.99]);
    let big_loss = softmaxed.cross_entropy_loss(&bad_target);
    let small_loss = softmaxed.cross_entropy_loss(&good_target);
    println!("input: {:?}", input);
    println!("weights: {:?}", weights);
    println!("output: {:?}", output);
    println!("softmaxed: {:?}", softmaxed);
    println!("big_loss: {:?}", big_loss);
    println!("small_loss: {:?}", small_loss);

    let matrix = Matrix::new(vec![2, 2], vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let vector = Vector::from(vec![1.0, 2.0]);
    println!("product1: {:?}", matrix.multiply(&vector));
    println!("product2: {:?}", matrix.transpose().multiply(&vector));
    println!("product3: {:?}", matrix.scale(&vector));

    repeat_nn_single_input();
}

fn repeat_nn_single_input() {
    let weights = Layer::new(
        Matrix::new(
            vec![3, 2],
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        ),
        vec![5.0, 0.0],
    );
    let mut nn1 = NeuralNetwork::new(vec![weights], Vector::from(vec![0.0, 1.0]));
    for _ in 0..100 {
        let input = Vector::from(vec![0.0, 1.0, 2.0]);
        nn1.train(input);
    }
    println!("loss: {:?}", nn1.loss);
    println!("weights: {:?}", nn1.layers[0].weights);
    println!("biases: {:?}", nn1.layers[0].biases);
    println!("output: {:?}", nn1.intermediates.last().unwrap());
}
