use std::vec;

use bamf::matrix::{Matrix, Multiply};
use bamf::{NeuralNetwork, NeuronLayer, WeightLayer};

fn main() {
    let input = NeuronLayer::from_vec(vec![0.0, 1.0, 2.0]);
    let mut weights = WeightLayer::new(
        Matrix::new(
            vec![3, 2],
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        ),
        vec![5.0, 0.0],
    );
    let output = weights.forward(&input);
    let softmaxed = output.softmax();
    let bad_target = NeuronLayer::from_vec(vec![0.5, 0.5]);
    let good_target = NeuronLayer::from_vec(vec![0.01, 0.99]);
    let big_loss = softmaxed.cross_entropy_loss(&bad_target);
    let small_loss = softmaxed.cross_entropy_loss(&good_target);
    println!("input: {:?}", input);
    println!("weights: {:?}", weights);
    println!("output: {:?}", output);
    println!("softmaxed: {:?}", softmaxed);
    println!("big_loss: {:?}", big_loss);
    println!("small_loss: {:?}", small_loss);

    let matrix = Matrix::new(vec![2, 2], vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let vector = vec![1.0, 2.0];
    println!("product1: {:?}", matrix.multiply(&vector));
    println!("product2: {:?}", matrix.transpose().multiply(&vector));
    println!("product3: {:?}", matrix.multiply_across(&vector));

    let mut nn1 = NeuralNetwork::new(&mut weights, &input, bad_target);
    nn1.forward();
    println!("nn1: {:?}", nn1.backward());
    let mut nn2 = NeuralNetwork::new(&mut weights, &input, good_target);
    nn2.forward();
    println!("nn2: {:?}", nn2.backward());

    repeat_nn_single_input();
}

fn repeat_nn_single_input() {
    let mut weights = WeightLayer::new(
        Matrix::new(
            vec![3, 2],
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        ),
        vec![5.0, 0.0],
    );
    let input = NeuronLayer::from_vec(vec![0.0, 1.0, 2.0]);
    let mut nn1 = NeuralNetwork::new(&mut weights, &input, NeuronLayer::from_vec(vec![0.0, 1.0]));
    for _ in 0..100 {
        nn1.train();
    }
    println!("loss: {:?}", nn1.loss);
    println!("weights: {:?}", nn1.layer.weights);
    println!("output: {:?}", nn1.intermediates.last().unwrap());
}
