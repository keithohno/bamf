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
        .window(3)
        .dim(32)
        .train(500000);

    // first (dynamic) nn layer: input -> embedding
    let dict_size = embedding.num_to_word.len();
    let nn_l1 = Layer::random((embedding.dim, embedding.dim), (0.0, 1.0)).with_activation(LELU);

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
    let mut nn_l2 = Layer::new(embedding_to_prediction_matrix, Vector::zero(dict_size));
    nn_l2.set_constant();

    // combine layers into nn
    let mut nn = NeuralNetwork::new(vec![nn_l1, nn_l2]);

    // process training data / input text
    let input_text = read_to_string("examples/your_name.txt").unwrap();
    let words = clean(input_text)
        .split_whitespace()
        .map(|s| s.to_owned())
        .collect::<Vec<String>>();

    // train nn (100 epochs)
    for _ in 0..100 {
        for i in 0..words.len() - 1 {
            let input_embedding = embedding.get(&words[i]).unwrap().clone();
            let mut target_one_hot = Vector::zero(dict_size);
            target_one_hot.data[*embedding.word_to_num.get(&words[i + 1]).unwrap()] = 1.0;
            nn.train(input_embedding, &target_one_hot);
        }
    }

    // predict next word for all words in dictionary
    for word in &embedding.num_to_word {
        let output = nn.forward(embedding.get(word).unwrap().clone());
        let prediction = &embedding.num_to_word[max_index(&output.data)];
        println!("{} -> {}", word, prediction);
    }
}

fn max_index(vec: &Vec<f64>) -> usize {
    let mut max = 0.0;
    let mut max_index = 0;
    for (i, val) in vec.iter().enumerate() {
        if *val > max {
            max = *val;
            max_index = i;
        }
    }
    max_index
}
