use bamf::{NeuronLayer, WeightLayer};

fn main() {
    let input = NeuronLayer::from_vec(vec![0.0, 1.0, 2.0]);
    let weights: WeightLayer = WeightLayer::new(
        vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]],
        vec![0.0, 1.0],
    );
    let output = weights.forward(&input);
    println!("input: {:?}", input);
    println!("weights: {:?}", weights);
    println!("output: {:?}", output);
}
