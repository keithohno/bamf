use rand::Rng;

use crate::ops::Scale;
use std::{ops::Index, slice::Iter};

#[derive(Debug, PartialEq, Clone)]
pub struct Vector {
    pub size: usize,
    pub data: Vec<f64>,
}

impl Vector {
    pub fn zero(size: usize) -> Vector {
        Vector {
            size,
            data: vec![0.0; size],
        }
    }

    pub fn random(size: usize, bounds: (f64, f64)) -> Vector {
        let mut result = Vector::zero(size);
        let range = bounds.1 - bounds.0;
        for i in 0..result.size {
            result.data[i] = rand::thread_rng().gen::<f64>() * range + bounds.0;
        }
        result
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn softmax(&self) -> Vector {
        let exp_sum = self.data.iter().map(|x| x.exp()).sum::<f64>();
        Vector {
            size: self.size,
            data: self.data.iter().map(|x| x.exp() / exp_sum).collect(),
        }
    }

    pub fn cross_entropy_loss(&self, expected: &Vector) -> f64 {
        let cross_entropy_fn = |expected: f64, actual: f64| -expected * actual.ln();
        expected
            .data
            .iter()
            .zip(self.data.iter())
            .map(|(e, a)| cross_entropy_fn(*e, *a))
            .sum::<f64>()
    }

    pub fn subtract(&self, vec2: &Vector) -> Vector {
        assert!(self.size == vec2.size);
        let res = self
            .data
            .iter()
            .zip(vec2.data.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>();
        Vector::from(res)
    }

    pub fn add(&self, vec2: &Vector) -> Vector {
        assert!(self.size == vec2.size);
        let res = self
            .data
            .iter()
            .zip(vec2.data.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<f64>>();
        Vector::from(res)
    }

    pub fn iter(&self) -> Iter<f64> {
        return self.data.iter();
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.data[index]
    }
}

impl PartialEq<Vec<f64>> for Vector {
    fn eq(&self, vec2: &Vec<f64>) -> bool {
        self.size == vec2.len() && self.data.iter().zip(vec2.iter()).all(|(x, y)| x == y)
    }
}

impl From<Vec<f64>> for Vector {
    fn from(data: Vec<f64>) -> Vector {
        let size = data.len();
        Vector { size, data }
    }
}

impl IntoIterator for Vector {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl Scale<&Vector> for Vector {
    fn scale(&self, vec2: &Vector) -> Vector {
        assert!(self.size == vec2.size);
        let res = self
            .data
            .iter()
            .zip(vec2.data.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<f64>>();
        Vector::from(res)
    }
}

impl Scale<f64> for Vector {
    fn scale(&self, c: f64) -> Vector {
        let res = self.data.iter().map(|x| x * c).collect::<Vec<f64>>();
        Vector::from(res)
    }
}

#[cfg(test)]
mod tests {
    use super::Vector;
    use crate::ops::Scale;

    const EPSILON: f64 = 0.00001;

    #[test]
    fn test_subtract() {
        let vec1 = Vector::from(vec![2.0, 1.0, 2.0]);
        let vec2 = Vector::from(vec![1.2, -2.5, 3.0]);
        let expected = Vector::from(vec![0.8, 3.5, -1.0]);
        assert_eq!(vec1.subtract(&vec2), expected);
    }

    #[test]
    fn test_add() {
        let vec1 = Vector::from(vec![2.0, 3.0, -2.0]);
        let vec2 = Vector::from(vec![1.2, -2.5, -3.1]);
        let expected = Vector::from(vec![3.2, 0.5, -5.1]);
        assert_eq!(vec1.add(&vec2), expected);
    }

    #[test]
    fn test_scale_scalar() {
        let vec = Vector::from(vec![2.0, 3.0, -1.5]);
        let result = vec.scale(0.3);
        let expected = Vector::from(vec![0.6, 0.9, -0.45]);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < EPSILON);
        }
    }

    #[test]
    fn test_scale_vector() {
        let vec1 = Vector::from(vec![1.0, 3.0, -1.5]);
        let vec2 = Vector::from(vec![-0.5, 4.0, -0.6]);
        let result = vec1.scale(&vec2);
        let expected = Vector::from(vec![-0.5, 12.0, 0.9]);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < EPSILON);
        }
    }

    #[test]
    fn test_softmax() {
        let vec = Vector::from(vec![-1.0, 0.0, 0.5]);
        let result = vec.softmax();
        let expected = Vector::from(vec![
            0.12195165230972886,
            0.3314989604240915,
            0.5465493872661796,
        ]);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < EPSILON);
        }
    }

    #[test]
    fn test_cross_entropy_loss() {
        let vec = Vector::from(vec![0.1, 0.3, 0.6]);
        let target = Vector::from(vec![0.0, 0.0, 1.0]);
        let result = vec.cross_entropy_loss(&target);
        let expected = 0.510825623765990;
        assert!((result - expected).abs() < EPSILON);
    }
}
