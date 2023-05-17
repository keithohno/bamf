use self::flatten::Node;

pub mod flatten;

#[derive(Debug)]
pub struct Tensor {
    data: Vec<f64>,
    dims: Vec<usize>,
    skips: Vec<usize>,
}

impl Tensor {
    pub fn new(dims: Vec<usize>, data: Node<f64>) -> Tensor {
        let size = dims.iter().product::<usize>();
        let skips = dims
            .iter()
            .scan(size, |acc, &x| {
                *acc /= x;
                Some(*acc)
            })
            .collect::<Vec<usize>>();

        let data = flatten::flatten(&data);
        assert!(data.len() == size);
        Tensor { data, dims, skips }
    }
}
