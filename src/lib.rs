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
