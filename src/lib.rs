use activation::RELU;
use matrix::{Matrix, Multiply};
use ops::Scale;
use vector::Vector;

pub mod activation;
pub mod language;
pub mod matrix;
pub mod ops;
pub mod vector;

#[derive(Debug)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Vector,
}

impl Layer {
    pub fn new<V>(weights: Matrix, biases: V) -> Layer
    where
        V: Into<Vector>,
    {
        let biases = biases.into();
        assert!(weights.dims[1] == biases.len());
        Layer { weights, biases }
    }

    pub fn random(dims: (usize, usize), bounds: (f64, f64)) -> Layer {
        Layer::new(Matrix::random(dims, bounds), Vector::random(dims.1, bounds))
    }

    pub fn forward(&self, input: &Vector) -> Vector {
        activation::apply(
            self.weights.transpose().multiply(&input).add(&self.biases),
            RELU,
        )
    }

    /// Computes the gradient of the loss wrt the input of the layer (x), the weights (w), and the biases (b).
    ///
    /// Returns the tuple (dl_dx, dl_dw, dl_db)
    ///
    /// # Arguments
    ///
    /// * `dl_dz` - gradient of loss wrt output of the layer (z)
    /// * `x` - input to the layer
    /// * `y` - output of weight multiplication, input to activation
    /// * `z` - output of the layer
    fn backward(
        &self,
        dl_dz: &Vector,
        x: &Vector,
        y: Option<&Vector>,
        z: Option<&Vector>,
    ) -> (Vector, Matrix, Vector) {
        let dl_dy = activation::backwards(dl_dz, y, z, RELU);

        let dl_dx = self.weights.multiply(&dl_dy);
        let mut dl_dw = Matrix::zero((self.weights.dims[0], self.weights.dims[1]));
        for i in 0..self.weights.dims[0] {
            for j in 0..self.weights.dims[1] {
                *dl_dw.get_mut(i, j) = x[i] * dl_dy[j];
            }
        }
        let dl_db = dl_dy;
        (dl_dx, dl_dw, dl_db)
    }
}

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub intermediates: Vec<Vector>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> NeuralNetwork {
        NeuralNetwork {
            layers,
            intermediates: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: Vector) -> &Vector {
        self.intermediates = vec![input];
        for layer in &self.layers {
            let output = layer.forward(self.intermediates.last().unwrap());
            self.intermediates.push(output);
        }
        self.intermediates
            .push(self.intermediates.last().unwrap().softmax());
        return self.intermediates.last().unwrap();
    }

    pub fn backward(&mut self, target: &Vector) -> Vec<(Matrix, Vector)> {
        let mut dl_dz = self.intermediates.pop().unwrap().subtract(target);
        let mut gradients = Vec::new();
        for i in (0..self.layers.len()).rev() {
            let input = &self.intermediates[i];
            let output = &self.intermediates[i + 1];
            let layer = &self.layers[i];
            let (dl_dx, dl_dw, dl_db) = layer.backward(&dl_dz, input, None, Some(output));
            gradients.push((dl_dw, dl_db));
            dl_dz = dl_dx;
        }
        gradients.reverse();
        gradients
    }

    pub fn loss(&self, target: &Vector) -> f64 {
        self.intermediates
            .last()
            .unwrap()
            .cross_entropy_loss(target)
    }

    pub fn train(&mut self, input: Vector, target: &Vector) -> f64 {
        self.forward(input);
        let loss = self.loss(target);
        let gradients = self.backward(target);
        for (i, (dl_dw, dl_db)) in gradients.iter().enumerate() {
            self.layers[i].weights = self.layers[i].weights.subtract(&dl_dw.scale(0.1));
            self.layers[i].biases = self.layers[i].biases.subtract(&dl_db.scale(0.1));
        }
        loss
    }
}
