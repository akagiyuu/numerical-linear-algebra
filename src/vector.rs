use num_complex::{Complex64, ComplexFloat};

use crate::Matrix;

pub type Vector<const N: usize> = Matrix<N, 1>;

impl<const N: usize> Vector<N> {
    pub fn dot(&self, rhs: &Vector<N>) -> Complex64 {
        self.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn norm(&self, p: u32) -> f64 {
        self.iter()
            .map(|x_i| x_i.abs().powi(p as i32))
            .sum::<f64>()
            .powf(1. / p as f64)
    }

    pub fn normalize(self) -> Self {
        let length = Complex64::new(self.norm(2), 0.);
        1. / length * self
    }

    pub fn project(self, rhs: Self) -> Self {
        self.dot(&rhs) / rhs.dot(&rhs) * rhs
    }

    pub fn gram_schimidt_process<const M: usize>(vectors: [Vector<N>; M]) -> [Vector<N>; M] {
        let mut basis_vectors = vectors.into_iter().enumerate().fold(
            [Vector::<N>::ZEROS; M],
            |mut basis_vectors, (i, vector)| {
                basis_vectors[i] = vector.clone()
                    - basis_vectors
                        .iter()
                        .take(i)
                        .map(|basis_vector| vector.clone().project(basis_vector.clone()))
                        .sum();

                basis_vectors
            },
        );

        basis_vectors
            .iter_mut()
            .for_each(|vector| *vector = vector.clone().normalize());

        basis_vectors
    }
}
