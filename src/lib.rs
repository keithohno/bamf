use matrix::{Matrix, Multiply};

pub mod matrix;

#[derive(Debug)]
pub struct NeuronLayer {
    dim: usize,
    vals: Vec<f64>,
}

impl NeuronLayer {
    fn new(dim: usize) -> NeuronLayer {
        NeuronLayer {
            dim: dim,
            vals: vec![0.0; dim],
        }
    }

    pub fn from_vec(vals: Vec<f64>) -> NeuronLayer {
        let dim = vals.len();
        NeuronLayer { dim, vals }
    }

    pub fn softmax(&self) -> NeuronLayer {
        let exp_sum = self.vals.iter().map(|x| x.exp()).sum::<f64>();
        NeuronLayer {
            dim: self.dim,
            vals: self.vals.iter().map(|x| x.exp() / exp_sum).collect(),
        }
    }

    pub fn cross_entropy_loss(&self, expected: &NeuronLayer) -> f64 {
        let cross_entropy_fn = |expected: f64, actual: f64| {
            -expected * actual.ln() - (1.0 - expected) * (1.0 - actual).ln()
        };
        expected
            .vals
            .iter()
            .zip(self.vals.iter())
            .map(|(e, a)| cross_entropy_fn(*e, *a))
            .sum::<f64>()
    }
}

#[derive(Debug)]
pub struct WeightLayer {
    dim: (usize, usize),
    weights: Matrix,
    biases: Vec<f64>,
}

impl WeightLayer {
    pub fn new(weights: Matrix, biases: Vec<f64>) -> WeightLayer {
        assert!(weights.dims[1] == biases.len());
        WeightLayer {
            dim: (weights.dims[0], weights.dims[1]),
            weights,
            biases,
        }
    }

    pub fn forward(&self, input: &NeuronLayer) -> NeuronLayer {
        let product = self.weights.transpose().multiply(&input.vals);
        NeuronLayer::from_vec(product)
    }
}
