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
    let nn_l1 = Layer::random((embedding.dim, embedding.dim), (0.0, 1.0)).with_activation(LELU);
    println!("{:?}", nn_l1.weights.dims);

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
    println!("{:?}", embedding_to_prediction_matrix.dims);
    let nn_l2 = Layer::new(embedding_to_prediction_matrix, Vector::zero(dict_size));

    // combine layers into nn
    let mut nn = NeuralNetwork::new(vec![nn_l1, nn_l2]);

    let input_text = read_to_string("examples/your_name.txt").unwrap();
    let words = clean(input_text)
        .split_whitespace()
        .map(|s| s.to_owned())
        .collect::<Vec<String>>();

    for i in 0..words.len() - 1 {
        let input_embedding = embedding.get(&words[i]).unwrap().clone();
        let mut target_one_hot = Vector::zero(dict_size);
        target_one_hot.data[*embedding.word_to_num.get(&words[i + 1]).unwrap()] = 1.0;
        nn.train(input_embedding, &target_one_hot);
    }

    let mut word = "your".to_owned();
    for _ in 0..100 {
        print!("{} ", word);
        let output = nn.forward(embedding.get(&word).unwrap().clone());
        let prediction = max_index(&output.data);
        word = embedding.num_to_word[prediction].clone();
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
