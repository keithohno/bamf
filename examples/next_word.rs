use std::fs::read_to_string;

use bamf::{
    activation::LELU,
    language::{clean, Embedding},
    matrix::Matrix,
    vector::Vector,
    Layer, NeuralNetwork,
};

fn main() {
    let input_text = read_to_string("examples/your_name.txt").unwrap();
    let embedding = Embedding::builder(input_text)
        .window(5)
        .dim(10)
        .train(100000);

    // first (dynamic) nn layer: input -> embedding
    let dict_size = embedding.num_to_word.len();
    let nn_l1 = Layer::random((dict_size, embedding.dim), (0.0, 1.0)).with_activation(LELU);

    // second (static) nn layer: embedding -> prediction
    let mut embeddings_arr: Vec<Vec<f64>> = vec![];
    for i in 0..dict_size {
        embeddings_arr.push(
            embedding
                .word_to_embed
                .get(&embedding.num_to_word[i])
                .unwrap()
                .data
                .clone(),
        );
    }
    let embedding_to_prediction_matrix = Matrix::from(embeddings_arr).transpose();
    let nn_l2 = Layer::new(embedding_to_prediction_matrix, Vector::zero(dict_size));

    // combine layers into nn
    let mut nn = NeuralNetwork::new(vec![nn_l1, nn_l2]);

    let input_text = read_to_string("examples/your_name.txt").unwrap();
    let words = clean(input_text)
        .split_whitespace()
        .map(|s| s.to_owned())
        .collect::<Vec<String>>();

    for i in 0..words.len() - 1 {
        let embedding1 = embedding.get(&words[i]);
        let embedding2 = embedding.get(&words[i + 1]);
        // train nn: input=embedding1, target=embedding2
    }
}
