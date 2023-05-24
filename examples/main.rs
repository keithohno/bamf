use std::vec;

use bamf::matrix::{Matrix, Multiply};
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

    repeat_nn_single_input();
}

fn repeat_nn_single_input() {
    let weights1 = Layer::new(
        Matrix::new(
            vec![3, 5],
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![1.0, 2.0, 1.0, 0.0, -1.0],
                vec![-2.5, -1.3, -1.1, -0.9, -0.7],
            ],
        ),
        vec![5.4, 3.2, 1.0, 1.2, 3.4],
    );
    let weights2 = Layer::new(
        Matrix::new(
            vec![5, 2],
            vec![
                vec![1.0, 2.0],
                vec![0.5, 0.8],
                vec![-1.5, -0.5],
                vec![-2.0, -1.0],
                vec![3.0, 5.0],
            ],
        ),
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
