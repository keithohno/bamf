use rand::{thread_rng, Rng};
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug)]
pub struct EmbeddingBuilder {
    vocab: HashMap<String, usize>,
    vocab_size: usize,
    codex: Vec<usize>,
    codex_size: usize,
    window: usize,
}

impl EmbeddingBuilder {
    pub fn new(input: String) -> EmbeddingBuilder {
        let (codex, vocab) = tokenize(clean(input));
        let vocab_size = vocab.len();
        let codex_size = codex.len();
        EmbeddingBuilder {
            vocab,
            vocab_size,
            codex,
            codex_size,
            window: 1,
        }
    }

    pub fn set_window(&mut self, window: usize) {
        self.window = window;
    }

    pub fn random_pairing(&self) -> (Vec<f64>, Vec<f64>) {
        let mut rng = thread_rng();
        let offset = rng.gen_range(1..=self.window);
        let index = rng.gen_range(0..(self.codex_size - offset));
        let (num1, num2) = match rng.gen::<bool>() {
            true => (self.codex[index], self.codex[index + offset]),
            false => (self.codex[index + offset], self.codex[index]),
        };
        (self.one_hot(num1), self.one_hot(num2))
    }

    pub fn one_hot(&self, num: usize) -> Vec<f64> {
        let mut one_hot = vec![0.0; self.vocab_size];
        one_hot[num] = 1.0;
        one_hot
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
