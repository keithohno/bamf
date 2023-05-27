use std::fs::read_to_string;

use bamf::language::{clean, tokenize};

fn main() {
    let input_text = read_to_string("examples/your_name.txt").unwrap();
    println!("{:?}", tokenize(clean(input_text)));
}
