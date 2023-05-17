#[derive(Debug)]
pub struct Matrix {
    data: Vec<f64>,
    dims: Vec<usize>,
    step: Vec<usize>,
    size: usize,
}

impl Matrix {
    pub fn new(dims: Vec<usize>, data: Vec<Vec<f64>>) -> Matrix {
        let mut matrix = Matrix::empty(dims);
        matrix.data = data.into_iter().flatten().collect();
        assert!(matrix.data.len() == matrix.size);
        matrix
    }

    pub fn empty(dims: Vec<usize>) -> Matrix {
        let size = dims[0] * dims[1];
        let data = vec![0.0; size];
        let step = vec![dims[1], 1];

        Matrix {
            data,
            dims,
            step,
            size,
        }
    }

    pub fn transpose(mut self) -> Matrix {
        self.dims.reverse();
        self.step.reverse();
        self
    }
}

pub trait Multiply<S, T> {
    fn multiply(&self, arg: &S) -> T;
}

impl Multiply<Vec<f64>, Vec<f64>> for Matrix {
    fn multiply(&self, vec: &Vec<f64>) -> Vec<f64> {
        assert!(self.dims[1] == vec.len());
        let mut result = vec![0.0; self.dims[0]];
        for i in 0..self.dims[0] {
            for j in 0..self.dims[1] {
                result[i] += self.data[i * self.step[0] + j * self.step[1]] * vec[j];
            }
        }
        result
    }
}
