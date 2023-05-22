use matrix::{Matrix, Multiply, Scale};
use vector::Vector;

pub mod matrix;
pub mod vector;

#[derive(Debug)]
pub struct WeightLayer {
    pub weights: Matrix,
    pub biases: Vector,
}

impl WeightLayer {
    pub fn new<V>(weights: Matrix, biases: V) -> WeightLayer
    where
        V: Into<Vector>,
    {
        let biases = biases.into();
        assert!(weights.dims[1] == biases.len());
        WeightLayer { weights, biases }
    }

    pub fn forward(&self, input: &Vector) -> Vector {
        self.weights.transpose().multiply(&input).add(&self.biases)
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

    pub fn backward(&mut self) -> (Matrix, Vector) {
        let dl_dy = self.intermediates.last().unwrap().subtract(&self.target);
        let mut dy_dw = Matrix::empty(vec![self.input.vals.len(), self.target.vals.len()]);
        for i in 0..dy_dw.dims[0] {
            for j in 0..dy_dw.dims[1] {
                dy_dw.data[i * dy_dw.step[0] + j * dy_dw.step[1]] = self.input.vals[i]
            }
        }
        let dy_db = Vector::from(vec![1.0; self.target.vals.len()]);
        let dl_dw = dy_dw.transpose().scale(&dl_dy);
        let dl_db = dy_db.scale(&dl_dy);
        (dl_dw, dl_db)
    }

    pub fn train(&mut self) {
        self.forward();
        let (dl_dw, dl_db) = self.backward();
        self.layer.weights = self.layer.weights.subtract(&dl_dw.transpose());
        self.layer.biases = self.layer.biases.subtract(&dl_db);
    }
}
