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

    pub fn cross_entropy_loss(&self, expected: NeuronLayer) -> f64 {
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
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl WeightLayer {
    pub fn new(weights: Vec<Vec<f64>>, biases: Vec<f64>) -> WeightLayer {
        let dim_out = weights.len();
        assert!(dim_out > 0);
        assert!(biases.len() == dim_out);
        let dim_in: usize = weights[0].len();
        assert!(weights.iter().all(|v| v.len() == dim_in));
        WeightLayer {
            dim: (dim_in, dim_out),
            weights,
            biases,
        }
    }

    pub fn forward(&self, input: &NeuronLayer) -> NeuronLayer {
        assert!(input.dim == self.dim.0);
        let mut output = NeuronLayer::new(self.dim.1);
        for i in 0..self.dim.1 {
            output.vals[i] = self.biases[i]
                + self.weights[i]
                    .iter()
                    .zip(input.vals.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
        }
        output
    }
}
