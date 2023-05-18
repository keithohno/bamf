use std::vec;

use bamf::matrix::{Matrix, Multiply};
use bamf::{NeuronLayer, WeightLayer};

fn main() {
    let input = NeuronLayer::from_vec(vec![0.0, 1.0, 2.0]);
    let weights = WeightLayer::new(
        Matrix::new(
            vec![3, 2],
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        ),
        vec![5.0, 0.0],
    );
    let output = weights.forward(&input);
    let softmaxed = output.softmax();
    let big_loss = softmaxed.cross_entropy_loss(&NeuronLayer::from_vec(vec![0.5, 0.5]));
    let small_loss = softmaxed.cross_entropy_loss(&NeuronLayer::from_vec(vec![0.01, 0.99]));
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
}
