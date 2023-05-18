pub fn subtract(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    assert!(a.len() == b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
