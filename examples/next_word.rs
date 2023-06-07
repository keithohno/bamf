use std::fs::read_to_string;

use bamf::language::{clean, Embedding};

fn main() {
    let input_text = read_to_string("examples/your_name.txt").unwrap();
    let embedding = Embedding::builder(input_text)
        .window(5)
        .dim(10)
        .train(1000000);

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
