use rand::{thread_rng, Rng};
use regex::Regex;
use std::collections::HashMap;

use crate::{vector::Vector, Layer, NeuralNetwork};

pub struct Embedding {
    pub map: HashMap<String, Vector>,
}

impl Embedding {
    pub fn builder(codex: String) -> EmbeddingBuilder {
        EmbeddingBuilder::new(codex)
    }

    pub fn get(&self, word: &str) -> Option<&Vector> {
        self.map.get(word)
    }
}

#[derive(Debug)]
pub struct EmbeddingBuilder {
    vocab: HashMap<String, usize>,
    pub vocab_size: usize,
    codex: Vec<usize>,
    codex_size: usize,
    window: usize,
    dim: usize,
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
            dim: 1,
        }
    }

    pub fn window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }

    pub fn dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    fn random_pairing(&self) -> (Vector, Vector) {
        let mut rng = thread_rng();
        let offset = rng.gen_range(1..=self.window);
        let index = rng.gen_range(0..(self.codex_size - offset));
        let (num1, num2) = match rng.gen::<bool>() {
            true => (self.codex[index], self.codex[index + offset]),
            false => (self.codex[index + offset], self.codex[index]),
        };
        (self.one_hot(num1), self.one_hot(num2))
    }

    fn one_hot(&self, num: usize) -> Vector {
        let mut one_hot = vec![0.0; self.vocab_size];
        one_hot[num] = 1.0;
        one_hot.into()
    }

    pub fn train(&mut self, runs: usize) -> Embedding {
        let nn_l1 = Layer::random((self.vocab_size, self.dim), (0.0, 1.0));
        let nn_l2 = Layer::random((self.dim, self.vocab_size), (0.0, 1.0));
        let mut nn = NeuralNetwork::new(vec![nn_l1, nn_l2]);

        let mut loss_sum = 0.0;
        for i in 0..runs {
            let (input, output) = self.random_pairing();
            let loss = nn.train(input, &output);
            loss_sum += loss;
            // TODO: remove average loss printing
            match i % 10000 {
                0 => {
                    println!("{}", loss_sum / 10000.0);
                    loss_sum = 0.0;
                }
                _ => {}
            }
        }

        // extract embedding from hidden layer
        let mut embedding_dict = HashMap::new();
        for (word, num) in &self.vocab {
            nn.forward(self.one_hot(*num));
            let embedding = nn.intermediates[1].clone();
            embedding_dict.insert(word.to_owned(), embedding);
        }

        Embedding {
            map: embedding_dict,
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
