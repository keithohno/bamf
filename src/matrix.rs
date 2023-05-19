#[derive(Debug)]
pub struct Matrix {
    pub data: Box<Vec<f64>>,
    pub dims: Vec<usize>,
    pub step: Vec<usize>,
    size: usize,
}

pub trait Multiply<S, T> {
    fn multiply(&self, arg: &S) -> T;
}

impl Matrix {
    pub fn new(dims: Vec<usize>, data: Vec<Vec<f64>>) -> Matrix {
        let mut matrix = Matrix::empty(dims);
        *matrix.data = data.into_iter().flatten().collect();
        assert!(matrix.data.len() == matrix.size);
        matrix
    }

    pub fn empty(dims: Vec<usize>) -> Matrix {
        let size = dims[0] * dims[1];
        let data = vec![0.0; size];
        let step = vec![dims[1], 1];

        Matrix {
            data: Box::new(data),
            dims,
            step,
            size,
        }
    }

    pub fn transpose(&self) -> Matrix {
        Matrix {
            data: self.data.clone(),
            dims: vec![self.dims[1], self.dims[0]],
            step: vec![self.step[1], self.step[0]],
            size: self.size,
        }
    }

    pub fn dot(&self, vec: &Vec<f64>) -> Matrix {
        assert!(self.dims[0] == vec.len());
        let mut result = Matrix::empty(self.dims.clone());
        for i in 0..self.dims[0] {
            for j in 0..self.dims[1] {
                result.data[i * self.step[0] + j * self.step[1]] =
                    self.data[i * self.step[0] + j * self.step[1]] * vec[i];
            }
        }
        result
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert!(self.dims == other.dims);
        let mut result = Matrix::empty(self.dims.clone());
        for i in 0..self.size {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }

    pub fn multiply(&self, vec: &Vec<f64>) -> Vec<f64> {
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
