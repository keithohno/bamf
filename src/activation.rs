use crate::vector::Vector;

#[derive(Debug)]
pub struct Activation {
    pub function: fn(x: f64) -> f64,
    pub derivative: fn(dl_dz: f64, y: Option<f64>, z: Option<f64>) -> f64,
}

impl Activation {
    pub fn apply(&self, vec: Vector) -> Vector {
        vec.data
            .iter()
            .map(|x| (self.function)(*x))
            .collect::<Vec<f64>>()
            .into()
    }

    pub fn backpropagate(&self, dl_dz: &Vector, y: Option<&Vector>, z: Option<&Vector>) -> Vector {
        if let Some(y) = y {
            dl_dz
                .iter()
                .zip(y.iter())
                .map(|(i, j)| (self.derivative)(*i, Some(*j), None))
                .collect::<Vec<f64>>()
                .into()
        } else if let Some(z) = z {
            dl_dz
                .iter()
                .zip(z.iter())
                .map(|(i, j)| (self.derivative)(*i, None, Some(*j)))
                .collect::<Vec<f64>>()
                .into()
        } else {
            panic!()
        }
    }
}

pub const RELU: Activation = Activation {
    function: |x| {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    },
    derivative: |dl_dz, _, z| {
        if let Some(z) = z {
            if z > 0.0 {
                dl_dz
            } else {
                0.0
            }
        } else {
            panic!("RELU::backwards: z argument is None")
        }
    },
};

pub const LELU: Activation = Activation {
    function: |x| {
        if x > 0.0 {
            x
        } else {
            x / 10.0
        }
    },
    derivative: |dl_dz, _, z| {
        if let Some(z) = z {
            if z > 0.0 {
                dl_dz
            } else {
                dl_dz / 10.0
            }
        } else {
            panic!("LELU::backwards: z argument is None")
        }
    },
};
