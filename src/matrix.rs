use crate::ops::Scale;
use crate::vector::Vector;
use rand::Rng;

#[derive(Debug)]
pub struct Matrix {
    pub data: Box<Vec<f64>>,
    pub dims: Vec<usize>,
    pub step: Vec<usize>,
    pub size: usize,
}

pub trait Multiply<S, T> {
    fn multiply(&self, arg: &S) -> T;
}

impl Matrix {
    pub fn zero(dims: (usize, usize)) -> Matrix {
        let size = dims.0 * dims.1;
        let data = vec![0.0; size];
        let step = vec![dims.1, 1];

        Matrix {
            data: Box::new(data),
            dims: vec![dims.0, dims.1],
            step,
            size,
        }
    }

    pub fn random(dims: (usize, usize), bounds: (f64, f64)) -> Matrix {
        let mut result = Matrix::zero(dims);
        let range = bounds.1 - bounds.0;
        for i in 0..result.size {
            result.data[i] = rand::thread_rng().gen::<f64>() * range + bounds.0;
        }
        result
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.step[0] + j * self.step[1]]
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.data[i * self.step[0] + j * self.step[1]]
    }

    pub fn transpose(&self) -> Matrix {
        Matrix {
            data: self.data.clone(),
            dims: vec![self.dims[1], self.dims[0]],
            step: vec![self.step[1], self.step[0]],
            size: self.size,
        }
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert!(self.dims == other.dims);
        let mut result = Matrix::zero((self.dims[0], self.dims[1]));
        for i in 0..self.size {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }
}

impl From<Vec<Vec<f64>>> for Matrix {
    fn from(data: Vec<Vec<f64>>) -> Matrix {
        let dims = vec![data.len(), data[0].len()];
        let step = vec![dims[1], 1];
        let size = dims[0] * dims[1];
        let data: Vec<f64> = data.into_iter().flatten().collect();
        assert!(data.len() == size);
        Matrix {
            data: Box::new(data),
            dims,
            step,
            size,
        }
    }
}

impl Multiply<Vector, Vector> for Matrix {
    fn multiply(&self, vec: &Vector) -> Vector {
        assert!(self.dims[1] == vec.len());
        let mut res = vec![0.0; self.dims[0]];
        for i in 0..self.dims[0] {
            for j in 0..self.dims[1] {
                res[i] += self.data[i * self.step[0] + j * self.step[1]] * vec[j];
            }
        }
        Vector::from(res)
    }
}

impl Scale<&Vector> for Matrix {
    fn scale(&self, vec: &Vector) -> Matrix {
        assert!(self.dims[0] == vec.len());
        let mut result = Matrix::zero((self.dims[0], self.dims[1]));
        for i in 0..self.dims[0] {
            for j in 0..self.dims[1] {
                result.data[i * self.step[0] + j * self.step[1]] =
                    self.data[i * self.step[0] + j * self.step[1]] * vec.data[i];
            }
        }
        result
    }
}

impl Scale<f64> for Matrix {
    fn scale(&self, c: f64) -> Matrix {
        let mut result = Matrix::zero((self.dims[0], self.dims[1]));
        result.data = Box::new(self.data.iter().map(|x| x * c).collect::<Vec<f64>>());
        result
    }
}

#[cfg(test)]
mod tests {

    use super::{Matrix, Multiply, Scale};
    use crate::vector::Vector;

    #[test]
    fn test_matrix_multiply() {
        let matrix = Matrix::from(vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]]);
        let vec = Vector::from(vec![1.0, 2.0, 3.0]);
        let result = matrix.multiply(&vec);
        assert_eq!(result, vec![14.0, 26.0]);
    }

    #[test]
    fn test_matrix_scale() {
        let matrix = Matrix::from(vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]]);
        let vec = Vector::from(vec![1.0, 2.0]);
        let result = matrix.scale(&vec);
        assert_eq!(*result.data, vec![1.0, 2.0, 3.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_matrix_subtract() {
        let matrix1 = Matrix::from(vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
        let matrix2 = Matrix::from(vec![vec![0.0, 1.0, 2.0], vec![3.0, 2.0, 1.0]]);
        let result = matrix1.subtract(&matrix2);
        assert_eq!(*result.data, vec![1.0, 1.0, 1.0, -1.0, 1.0, 3.0]);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = Matrix::from(vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
        let result = matrix.transpose();
        assert_eq!(*result.data, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.dims, vec![3, 2]);
    }

    #[test]
    fn test_matrix_transposed_multiply() {
        let matrix = Matrix::from(vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
        let vec = Vector::from(vec![1.0, 2.0]);
        let result = matrix.transpose().multiply(&vec);
        assert_eq!(result, vec![5.0, 8.0, 11.0]);
    }
}
