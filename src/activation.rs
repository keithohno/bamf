use crate::vector::Vector;

pub struct ReLU {}

pub trait Activation {
    fn apply(x: f64) -> f64;
    fn backwards(dl_dz: f64, y: Option<f64>, z: Option<f64>) -> f64;
}

impl Activation for ReLU {
    fn apply(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn backwards(dl_dz: f64, _: Option<f64>, z: Option<f64>) -> f64 {
        if let Some(z) = z {
            if z > 0.0 {
                dl_dz
            } else {
                0.0
            }
        } else {
            panic!()
        }
    }
}

pub fn apply<T>(vec: Vector, _: T) -> Vector
where
    T: Activation,
{
    vec.vals
        .iter()
        .map(|x| T::apply(*x))
        .collect::<Vec<f64>>()
        .into()
}

pub fn backwards<T>(dl_dz: &Vector, y: Option<&Vector>, z: Option<&Vector>, _: T) -> Vector
where
    T: Activation,
{
    if let Some(y) = y {
        dl_dz
            .iter()
            .zip(y.iter())
            .map(|(i, j)| T::backwards(*i, Some(*j), None))
            .collect::<Vec<f64>>()
            .into()
    } else if let Some(z) = z {
        dl_dz
            .iter()
            .zip(z.iter())
            .map(|(i, j)| T::backwards(*i, None, Some(*j)))
            .collect::<Vec<f64>>()
            .into()
    } else {
        panic!()
    }
}

pub const RELU: ReLU = ReLU {};
