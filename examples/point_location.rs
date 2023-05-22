use bamf::vector::Vector;
use bamf::{Layer, NeuralNetwork};
use rand::Rng;

fn main() {
    println!("in-circle accuracy rate: {}", run_circle_nn());
    println!("near-axis accuracy rate: {}", run_axis_nn());
}

fn run_axis_nn() -> f64 {
    const RADIUS: f64 = 0.4;

    let mut nn = NeuralNetwork::new(vec![
        Layer::random((2, 12), (0.0, 1.0)),
        Layer::random((12, 12), (0.0, 1.0)),
        Layer::random((12, 2), (0.0, 1.0)),
    ]);

    for _ in 0..100000 {
        let point = random_point();
        let near_axis = is_near_axis(point, RADIUS);
        let input = Vector::from(vec![point.0, point.1]);
        let target = if near_axis {
            Vector::from(vec![1.0, 0.0])
        } else {
            Vector::from(vec![0.0, 1.0])
        };
        nn.train(input, &target);
    }

    let mut correct = 0;
    let mut total = 0;
    for _ in 0..1000 {
        let point = random_point();
        let near_axis = is_near_axis(point, RADIUS);
        let input = Vector::from(vec![point.0, point.1]);
        let result = nn.forward(input);
        let predicted_in_circle = result.data[0] > result.data[1];
        if predicted_in_circle == near_axis {
            correct += 1;
        }
        total += 1;
    }

    correct as f64 / total as f64
}

fn run_circle_nn() -> f64 {
    const RADIUS: f64 = 0.8;

    let mut nn = NeuralNetwork::new(vec![
        Layer::random((2, 12), (0.0, 1.0)),
        Layer::random((12, 12), (0.0, 1.0)),
        Layer::random((12, 2), (0.0, 1.0)),
    ]);

    for _ in 0..100000 {
        let point = random_point();
        let in_circle = is_in_circle(point, RADIUS);
        let input = Vector::from(vec![point.0, point.1]);
        let target = if in_circle {
            Vector::from(vec![1.0, 0.0])
        } else {
            Vector::from(vec![0.0, 1.0])
        };
        nn.train(input, &target);
    }

    let mut correct = 0;
    let mut total = 0;
    for _ in 0..1000 {
        let point = random_point();
        let in_circle = is_in_circle(point, RADIUS);
        let input = Vector::from(vec![point.0, point.1]);
        let result = nn.forward(input);
        let predicted_in_circle = result.data[0] > result.data[1];
        if predicted_in_circle == in_circle {
            correct += 1;
        }
        total += 1;
    }

    correct as f64 / total as f64
}

fn random_point() -> (f64, f64) {
    let mut rng = rand::thread_rng();
    (rng.gen::<f64>() * 2.0 - 1.0, rng.gen::<f64>() * 2.0 - 1.0)
}

fn is_in_circle(point: (f64, f64), radius: f64) -> bool {
    let (x, y) = point;
    x * x + y * y < radius * radius
}

fn is_near_axis(point: (f64, f64), radius: f64) -> bool {
    let (x, y) = point;
    x.abs() < radius && y.abs() < radius
}
