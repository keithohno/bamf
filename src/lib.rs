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
        self.weights
            .transpose()
            .multiply(&input)
            .add(&self.biases)
            // ReLU
            .into_iter()
            .map(|x| if x > 0.0 { x } else { 0.0 })
            .collect::<Vec<f64>>()
            .into()
    }

    /// Computes the gradient of the loss wrt the input of the layer (x), the weights (w), and the biases (b).
    ///
    /// Returns the tuple (dl_dx, dl_dw, dl_db)
    ///
    /// # Arguments
    ///
    /// * `dl_dy` - gradient of loss wrt output of the layer (y)
    /// * `x` - input to the layer
    /// * `y` - output of the layer
    fn backward(&self, dl_dy: &Vector, x: &Vector, y: &Vector) -> (Vector, Matrix, Vector) {
        let dl_dx = self.weights.multiply(&dl_dy);
        let dl_db = Vector::from(
            y.iter()
                // ReLU
                .map(|i| if *i > 0.0 { 1.0 } else { 0.0 })
                .zip(dl_dy.iter())
                .map(|(i, j)| i * j)
                .collect::<Vec<f64>>(),
        );
        let mut dl_dw = Matrix::empty(self.weights.dims.clone());
        for j in 0..self.weights.dims[1] {
            // ReLU
            if y[j] > 0.0 {
                for i in 0..self.weights.dims[0] {
                    *dl_dw.get_mut(i, j) = x[i] * dl_dy[j];
                }
            }
        }
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
        let dl_dy = self.intermediates.last().unwrap().subtract(&self.target);
        let (dl_dx, dl_dw, dl_db) = self
            .layer
            .backward(&dl_dy, self.input, &self.intermediates[0]);
        (dl_dw, dl_db)
    }

    pub fn train(&mut self) {
        self.forward();
        let (dl_dw, dl_db) = self.backward();
        self.layer.weights = self.layer.weights.subtract(&dl_dw);
        self.layer.biases = self.layer.biases.subtract(&dl_db);
    }
}
