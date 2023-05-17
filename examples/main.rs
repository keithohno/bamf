use bamf::tensor::flatten::{flatten, Node};
use bamf::tensor::Tensor;
use bamf::{NeuronLayer, WeightLayer};

fn main() {
    let input = NeuronLayer::from_vec(vec![0.0, 1.0, 2.0]);
    let weights = WeightLayer::new(
        vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]],
        vec![5.0, 0.0],
    );
    let output = weights.forward(&input);
    let softmaxed = output.softmax();
    let big_loss = softmaxed.cross_entropy_loss(NeuronLayer::from_vec(vec![0.5, 0.5]));
    let small_loss = softmaxed.cross_entropy_loss(NeuronLayer::from_vec(vec![0.01, 0.99]));
    println!("input: {:?}", input);
    println!("weights: {:?}", weights);
    println!("output: {:?}", output);
    println!("softmaxed: {:?}", softmaxed);
    println!("big_loss: {:?}", big_loss);
    println!("small_loss: {:?}", small_loss);

    let data = Node::from(vec![Node::from(vec![1.0, 2.0]), Node::from(vec![3.0, 4.0])]);
    let tensor = Tensor::new(vec![2, 2], Node::from(data));
    println!("tensor: {:?}", tensor);
}
