use regex::Regex;
use std::collections::HashMap;

#[derive(Debug)]
pub struct EmbeddingBuilder {
    vocab: HashMap<String, usize>,
    vocab_size: usize,
    codex: Vec<usize>,
}

impl EmbeddingBuilder {
    pub fn new(input: String) -> EmbeddingBuilder {
        let (codex, vocab) = tokenize(clean(input));
        let vocab_size = vocab.len();
        EmbeddingBuilder {
            vocab,
            vocab_size,
            codex,
        }
    }
}

pub fn clean(input: String) -> String {
    let re = Regex::new("[.,:;?!-\"]").unwrap();
    let input = re.replace_all(&input, "");
    let re = Regex::new("-").unwrap();
    re.replace_all(&input, " ").to_lowercase().to_string()
}

pub fn tokenize(input: String) -> (Vec<usize>, HashMap<String, usize>) {
    let words = input.split_whitespace();
    let mut nums = vec![];
    let mut word_to_num = HashMap::new();
    for (word) in words {
        match word_to_num.get(word) {
            Some(token) => nums.push(*token),
            None => {
                let token = word_to_num.len();
                nums.push(token);
                word_to_num.insert(word.to_owned(), token);
            }
        }
    }
    (nums, word_to_num)
}
