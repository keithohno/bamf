use regex::Regex;
use std::collections::HashMap;

pub fn clean(input: String) -> String {
    let re = Regex::new("[.,:;?!-\"]").unwrap();
    let input = re.replace_all(&input, "");
    let re = Regex::new("-").unwrap();
    re.replace_all(&input, " ").to_lowercase().to_string()
}

pub fn tokenize(input: String) -> Vec<usize> {
    let words = input.split_whitespace();
    let mut tokens = vec![];
    let mut word_token_dict = HashMap::new();
    for (word) in words {
        match word_token_dict.get(&word) {
            Some(token) => tokens.push(*token),
            None => {
                let token = word_token_dict.len();
                tokens.push(token);
                word_token_dict.insert(word, token);
            }
        }
    }
    tokens
}
