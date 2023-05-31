use std::fs::read_to_string;

use bamf::language::clean;

fn main() {
    let input_text = read_to_string("examples/your_name.txt").unwrap();
    println!("{}", clean(input_text));
}
