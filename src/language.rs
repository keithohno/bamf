use regex::Regex;

pub fn clean(input: String) -> String {
    let re = Regex::new("[.,?\"]").unwrap();
    re.replace_all(&input, "").to_lowercase().to_string()
}
