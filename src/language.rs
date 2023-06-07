use rand::{thread_rng, Rng};
use regex::Regex;
use std::collections::HashMap;

use crate::{activation::LELU, vector::Vector, Layer, NeuralNetwork};

pub struct Embedding {
    pub word_to_embed: HashMap<String, Vector>,
    pub word_to_num: HashMap<String, usize>,
    pub num_to_word: Vec<String>,
    pub dim: usize,
}

impl Embedding {
    pub fn builder(codex: String) -> EmbeddingBuilder {
        EmbeddingBuilder::new(codex)
    }

    pub fn get(&self, word: &str) -> Option<&Vector> {
        self.word_to_embed.get(word)
    }
}

#[derive(Debug)]
pub struct EmbeddingBuilder {
    word_to_num: HashMap<String, usize>,
    num_to_word: Vec<String>,
    pub dict_size: usize,
    codex: Vec<usize>,
    codex_size: usize,
    window: usize,
    dim: usize,
}

impl EmbeddingBuilder {
    pub fn new(input: String) -> EmbeddingBuilder {
        let (codex, word_to_num, num_to_word) = tokenize(clean(input));
        let vocab_size = word_to_num.len();
        let codex_size = codex.len();
        EmbeddingBuilder {
            word_to_num,
            num_to_word,
            dict_size: vocab_size,
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
        let mut one_hot = vec![0.0; self.dict_size];
        one_hot[num] = 1.0;
        one_hot.into()
    }

    pub fn train(&mut self, runs: usize) -> Embedding {
        let nn_l1 = Layer::random((self.dict_size, self.dim), (0.0, 1.0)).with_activation(LELU);
        let nn_l2 = Layer::random((self.dim, self.dict_size), (0.0, 1.0));
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
        let mut word_to_embed = HashMap::new();
        for (word, num) in &self.word_to_num {
            nn.forward(self.one_hot(*num));
            let embedding = nn.intermediates[1].clone();
            word_to_embed.insert(word.to_owned(), embedding);
        }

        Embedding {
            word_to_embed,
            num_to_word: self.num_to_word.clone(),
            word_to_num: self.word_to_num.clone(),
            dim: self.dim,
        }
    }
}

pub fn clean(input: String) -> String {
    let re = Regex::new("[.,:;?!-\"]").unwrap();
    let input = re.replace_all(&input, "");
    let re = Regex::new("-").unwrap();
    re.replace_all(&input, " ").to_lowercase().to_string()
}

pub fn tokenize(input: String) -> (Vec<usize>, HashMap<String, usize>, Vec<String>) {
    let words = input.split_whitespace();
    let mut nums = vec![];
    let mut word_to_num = HashMap::new();
    let mut num_to_word = vec![];
    for word in words {
        match word_to_num.get(word) {
            Some(num) => nums.push(*num),
            None => {
                let num = word_to_num.len();
                nums.push(num);
                word_to_num.insert(word.to_owned(), num);
                num_to_word.push(word.to_owned());
            }
        }
    }
    (nums, word_to_num, num_to_word)
}
