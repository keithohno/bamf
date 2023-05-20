use std::ops::Index;

#[derive(Debug, PartialEq)]
pub struct Vector {
    pub dim: usize,
    pub vals: Vec<f64>,
}

impl Vector {
    pub fn from_vec(vals: Vec<f64>) -> Vector {
        let dim = vals.len();
        Vector { dim, vals }
    }

    pub fn len(&self) -> usize {
        self.dim
    }

    pub fn softmax(&self) -> Vector {
        let exp_sum = self.vals.iter().map(|x| x.exp()).sum::<f64>();
        Vector {
            dim: self.dim,
            vals: self.vals.iter().map(|x| x.exp() / exp_sum).collect(),
        }
    }

    pub fn cross_entropy_loss(&self, expected: &Vector) -> f64 {
        let cross_entropy_fn = |expected: f64, actual: f64| -expected * actual.ln();
        expected
            .vals
            .iter()
            .zip(self.vals.iter())
            .map(|(e, a)| cross_entropy_fn(*e, *a))
            .sum::<f64>()
    }

    pub fn subtract(&self, vec2: &Vector) -> Vector {
        assert!(self.dim == vec2.dim);
        let res = self
            .vals
            .iter()
            .zip(vec2.vals.iter())
            .map(|(x, y)| x - y)
            .collect();
        Vector::from_vec(res)
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.vals[index]
    }
}

impl PartialEq<Vec<f64>> for Vector {
    fn eq(&self, vec2: &Vec<f64>) -> bool {
        self.dim == vec2.len() && self.vals.iter().zip(vec2.iter()).all(|(x, y)| x == y)
    }
}