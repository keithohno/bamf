use activation::RELU;
use matrix::{Matrix, Multiply};
use vector::Vector;

pub mod activation;
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
        let mut dl_dw = Matrix::empty(self.weights.dims.clone());
        for i in 0..self.weights.dims[0] {
            for j in 0..self.weights.dims[1] {
                *dl_dw.get_mut(i, j) = x[i] * dl_dy[j];
            }
        }
        let dl_db = dl_dy;
        (dl_dx, dl_dw, dl_db)
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
        let dl_dz = self.intermediates.last().unwrap().subtract(&self.target);
        let (_, dl_dw, dl_db) =
            self.layer
                .backward(&dl_dz, self.input, None, Some(&self.intermediates[0]));
        (dl_dw, dl_db)
    }

    pub fn train(&mut self) {
        self.forward();
        let (dl_dw, dl_db) = self.backward();
        self.layer.weights = self.layer.weights.subtract(&dl_dw);
        self.layer.biases = self.layer.biases.subtract(&dl_db);
    }
}
