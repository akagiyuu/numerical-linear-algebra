use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use num_complex::{Complex64, ComplexFloat};
use tabled::{
    builder::Builder,
    settings::{Margin, Style},
};

#[derive(Debug, Clone)]
pub struct Matrix<const M: usize, const N: usize> {
    data: [[Complex64; N]; M],
}

pub type Vector<const N: usize> = Matrix<1, N>;

impl<const M: usize, const N: usize> Index<[usize; 2]> for Matrix<M, N> {
    type Output = Complex64;

    fn index(&self, [i, j]: [usize; 2]) -> &Self::Output {
        &self.data[i][j]
    }
}

impl<const M: usize, const N: usize> IndexMut<[usize; 2]> for Matrix<M, N> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.data[index[0]][index[1]]
    }
}

impl<const M: usize, const N: usize> AddAssign for Matrix<M, N> {
    fn add_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(left, right)| *left += right);
    }
}

impl<const M: usize, const N: usize> Add for Matrix<M, N> {
    type Output = Matrix<M, N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        result += rhs;
        result
    }
}

impl<const M: usize, const N: usize> Sum for Matrix<M, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Matrix::ZEROS)
    }
}

impl<const M: usize, const N: usize> SubAssign for Matrix<M, N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(left, right)| *left -= right);
    }
}

impl<const M: usize, const N: usize> Sub for Matrix<M, N> {
    type Output = Matrix<M, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        result -= rhs;
        result
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<Matrix<N, P>> for Matrix<M, N> {
    type Output = Matrix<M, P>;

    fn mul(self, rhs: Matrix<N, P>) -> Self::Output {
        let mut result = Matrix::ZEROS;

        for i in 0..M {
            for j in 0..P {
                for k in 0..N {
                    result[[i, j]] += self[[i, k]] * rhs[[k, j]];
                }
            }
        }

        result
    }
}

impl<const M: usize, const N: usize> MulAssign<Complex64> for Matrix<M, N> {
    fn mul_assign(&mut self, rhs: Complex64) {
        self.iter_mut().for_each(|x| *x *= rhs);
    }
}

impl<const M: usize, const N: usize> Mul<Complex64> for Matrix<M, N> {
    type Output = Matrix<M, N>;

    fn mul(self, rhs: Complex64) -> Self::Output {
        let mut result = self.clone();
        result *= rhs;
        result
    }
}

impl<const M: usize, const N: usize> Mul<Matrix<M, N>> for Complex64 {
    type Output = Matrix<M, N>;

    fn mul(self, rhs: Matrix<M, N>) -> Self::Output {
        rhs * self
    }
}

impl<const M: usize, const N: usize> Display for Matrix<M, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut table_builder = Builder::default();
        for i in 0..M {
            table_builder.push_record(self.data[i].iter().map(|x| x.to_string()));
        }
        let mut table = table_builder.build();
        table.with(Style::blank());
        table.with(Margin::new(4, 0, 0, 0));

        write!(f, "[\n{}\n]", table)
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    #[inline]
    pub const fn new(data: [[Complex64; N]; M]) -> Self {
        Self { data }
    }

    pub fn from_row_vectors(rows: [Vector<N>; M]) -> Self {
        Matrix::new(rows.map(|row| row.data[0]))
    }

    pub fn to_row_vectors(self) -> [Vector<N>; M] {
        self.data.map(|row| Matrix::new([row]))
    }

    pub fn from_column_vectors(columns: [Vector<M>; N]) -> Self {
        Matrix::<N, M>::from_row_vectors(columns).transpose()
    }

    pub fn to_column_vectors(self) -> [Vector<M>; N] {
        self.transpose().to_row_vectors()
    }

    pub const ZEROS: Self = Matrix::new([[Complex64::ZERO; N]; M]);

    pub fn iter(&self) -> impl Iterator<Item = &Complex64> {
        self.data.iter().flatten()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Complex64> {
        self.data.iter_mut().flatten()
    }

    pub fn transpose(&self) -> Matrix<N, M> {
        let mut result = Matrix::ZEROS;

        for i in 0..N {
            for j in 0..M {
                result[[j, i]] = self[[i, j]];
            }
        }

        result
    }
}

impl<const M: usize> Matrix<M, M> {
    pub fn diagonal(data: [Complex64; M]) -> Self {
        let mut result = Matrix::ZEROS;
        for i in 0..M {
            result[[i, i]] = data[i];
        }

        result
    }

    pub fn identity() -> Self {
        Self::diagonal([Complex64::ONE; M])
    }
}

impl<const N: usize> Vector<N> {
    pub fn dot(&self, rhs: &Vector<N>) -> Complex64 {
        self.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn norm(&self, p: u32) -> f64 {
        self.data[0]
            .iter()
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
