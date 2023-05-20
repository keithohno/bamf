use matrix::{Matrix, Multiply, Scale};
use vector::Vector;

pub mod matrix;
pub mod vector;

#[derive(Debug)]
pub struct WeightLayer {
    dim: (usize, usize),
    pub weights: Matrix,
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

    pub fn forward(&self, input: &Vector) -> Vector {
        self.weights.transpose().multiply(&input)
    }
}

pub struct NeuralNetwork<'a> {
    pub layer: &'a mut WeightLayer,
    input: &'a Vector,
    pub intermediates: Vec<Vector>,
    target: Vector,
    pub loss: f64,
}

impl<'a> NeuralNetwork<'a> {
    pub fn new(layer: &'a mut WeightLayer, input: &'a Vector, target: Vector) -> NeuralNetwork<'a> {
        NeuralNetwork {
            layer,
            input,
            intermediates: Vec::new(),
            target,
            loss: 0.0,
        }
    }

    pub fn forward(&mut self) {
        self.intermediates = Vec::new();
        self.intermediates.push(self.layer.forward(&self.input));
        self.intermediates
            .push(self.intermediates.last().unwrap().softmax());
        self.loss = self
            .intermediates
            .last()
            .unwrap()
            .cross_entropy_loss(&self.target);
    }

    pub fn backward(&mut self) -> Matrix {
        let dl_dy = self.intermediates.last().unwrap().subtract(&self.target);
        let dvec = (0..self.input.vals.len())
            .map(|x| vec![self.input.vals[x]; self.target.vals.len()])
            .collect::<Vec<Vec<f64>>>();
        let mut dw_dy = Matrix::empty(vec![self.input.vals.len(), self.target.vals.len()]);
        for i in 0..dw_dy.dims[0] {
            for j in 0..dw_dy.dims[1] {
                dw_dy.data[i * dw_dy.step[0] + j * dw_dy.step[1]] = self.input.vals[i]
            }
        }
        dw_dy.transpose().scale(&dl_dy)
    }

    pub fn train(&mut self) {
        self.forward();
        let dw_dy = self.backward();
        self.layer.weights = self.layer.weights.subtract(&dw_dy.transpose());
    }
}
