use activation::RELU;
use matrix::{Matrix, Multiply};
use ops::Scale;
use vector::Vector;

pub mod activation;
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
    target: Vector,
    pub loss: f64,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, target: Vector) -> NeuralNetwork {
        NeuralNetwork {
            layers,
            intermediates: Vec::new(),
            target,
            loss: 0.0,
        }
    }

    pub fn forward(&mut self, input: Vector) {
        self.intermediates = vec![input];
        for layer in &self.layers {
            let output = layer.forward(self.intermediates.last().unwrap());
            self.intermediates.push(output);
        }
        self.intermediates
            .push(self.intermediates.last().unwrap().softmax());
        self.loss = self
            .intermediates
            .last()
            .unwrap()
            .cross_entropy_loss(&self.target);
    }

    pub fn backward(&mut self) -> Vec<(Matrix, Vector)> {
        let mut dl_dz = self.intermediates.pop().unwrap().subtract(&self.target);
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

    pub fn train(&mut self, input: Vector) {
        self.forward(input);
        let gradients = self.backward();
        for (i, (dl_dw, dl_db)) in gradients.iter().enumerate() {
            self.layers[i].weights = self.layers[i].weights.subtract(&dl_dw.scale(0.1));
            self.layers[i].biases = self.layers[i].biases.subtract(&dl_db.scale(0.1));
        }
    }
}
