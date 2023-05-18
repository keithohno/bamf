use matrix::{Matrix, Multiply};
use misc::subtract;

pub mod matrix;
pub mod misc;

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
        let cross_entropy_fn = |expected: f64, actual: f64| -expected * actual.ln();
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

pub struct NeuralNetwork<'a> {
    layer: &'a WeightLayer,
    input: &'a NeuronLayer,
    intermediates: Vec<NeuronLayer>,
    target: NeuronLayer,
    loss: f64,
}

impl<'a> NeuralNetwork<'a> {
    pub fn new(
        layer: &'a WeightLayer,
        input: &'a NeuronLayer,
        target: NeuronLayer,
    ) -> NeuralNetwork<'a> {
        NeuralNetwork {
            layer,
            input,
            intermediates: Vec::new(),
            target,
            loss: 0.0,
        }
    }

    pub fn forward(&mut self) {
        self.intermediates.push(self.layer.forward(&self.input));
        self.intermediates
            .push(self.intermediates.last().unwrap().softmax());
        self.loss = self.intermediates[1].cross_entropy_loss(&self.target);
    }

    pub fn backward(&mut self) -> Matrix {
        let dl_dy = subtract(&self.intermediates.last().unwrap().vals, &self.target.vals);
        let dvec = (0..self.input.vals.len())
            .map(|x| vec![self.input.vals[x]; self.target.vals.len()])
            .collect::<Vec<Vec<f64>>>();
        let mut dw_dy = Matrix::new(vec![self.input.vals.len(), self.target.vals.len()], dvec);
        for i in 0..dw_dy.dims[0] {
            for j in 0..dw_dy.dims[1] {
                dw_dy.data[i * dw_dy.step[0] + j * dw_dy.step[1]] = self.input.vals[i]
            }
        }
        dw_dy.transpose().multiply_across(&dl_dy)
    }
}
