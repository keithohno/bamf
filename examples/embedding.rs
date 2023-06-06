use std::fs::read_to_string;
use std::io;

use bamf::language::Embedding;

fn main() {
    let input_text = read_to_string("examples/your_name.txt").unwrap();
    let embedding = Embedding::builder(input_text)
        .window(5)
        .dim(10)
        .train(1000000);
    loop {
        let mut word1 = String::new();
        let mut word2 = String::new();

        println!("word 1: ");
        io::stdin().read_line(&mut word1).unwrap();
        println!("word 2: ");
        io::stdin().read_line(&mut word2).unwrap();

        let embedding1 = embedding.get(&word1.trim());
        let embedding2 = embedding.get(&word2.trim());
        match (embedding1, embedding2) {
            (Some(embedding1), Some(embedding2)) => {
                println!("similarity: {}", embedding1.dot(&embedding2));
            }
            _ => println!("word not found"),
        }
    }
}
