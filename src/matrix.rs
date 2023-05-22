use crate::vector::Vector;

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

pub trait Scale<S> {
    fn scale(&self, arg: &S) -> Matrix;
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

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert!(self.dims == other.dims);
        let mut result = Matrix::empty(self.dims.clone());
        for i in 0..self.size {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
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

impl Scale<Vector> for Matrix {
    fn scale(&self, vec: &Vector) -> Matrix {
        assert!(self.dims[0] == vec.len());
        let mut result = Matrix::empty(self.dims.clone());
        for i in 0..self.dims[0] {
            for j in 0..self.dims[1] {
                result.data[i * self.step[0] + j * self.step[1]] =
                    self.data[i * self.step[0] + j * self.step[1]] * vec.vals[i];
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {

    use crate::matrix::{Matrix, Multiply, Scale};
    use crate::vector::Vector;

    #[test]
    fn test_matrix_multiply() {
        let matrix = Matrix::new(vec![2, 3], vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]]);
        let vec = Vector::from(vec![1.0, 2.0, 3.0]);
        let result = matrix.multiply(&vec);
        assert_eq!(result, vec![14.0, 26.0]);
    }

    #[test]
    fn test_matrix_scale() {
        let matrix = Matrix::new(vec![2, 3], vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]]);
        let vec = Vector::from(vec![1.0, 2.0]);
        let result = matrix.scale(&vec);
        assert_eq!(*result.data, vec![1.0, 2.0, 3.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_matrix_subtract() {
        let matrix1 = Matrix::new(vec![2, 3], vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
        let matrix2 = Matrix::new(vec![2, 3], vec![vec![0.0, 1.0, 2.0], vec![3.0, 2.0, 1.0]]);
        let result = matrix1.subtract(&matrix2);
        assert_eq!(*result.data, vec![1.0, 1.0, 1.0, -1.0, 1.0, 3.0]);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = Matrix::new(vec![2, 3], vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
        let result = matrix.transpose();
        assert_eq!(*result.data, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.dims, vec![3, 2]);
    }

    #[test]
    fn test_matrix_transposed_multiply() {
        let matrix = Matrix::new(vec![2, 3], vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
        let vec = Vector::from(vec![1.0, 2.0]);
        let result = matrix.transpose().multiply(&vec);
        assert_eq!(result, vec![5.0, 8.0, 11.0]);
    }
}
