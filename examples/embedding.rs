use std::fs::read_to_string;

use bamf::language::Embedding;

fn main() {
    let input_text = read_to_string("examples/your_name.txt").unwrap();
    let embedding = Embedding::builder(input_text)
        .window(2)
        .dim(10)
        .train(50000);
    println!("{:?}", embedding.get("your").unwrap());
    println!("{:?}", embedding.get("name").unwrap());
    println!("{:?}", embedding.get("taki").unwrap());
    println!("{:?}", embedding.get("mitsuha").unwrap());
    println!("{:?}", embedding.get("asdfasdf"));
}
