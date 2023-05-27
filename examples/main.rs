use std::fs::read_to_string;

use bamf::language::EmbeddingBuilder;

fn main() {
    let input_text = read_to_string("examples/your_name.txt").unwrap();
    let embedding_builder = EmbeddingBuilder::new(input_text);
    println!("{:?}", embedding_builder);
    println!("{:?}", embedding_builder.random_pairing());
}
