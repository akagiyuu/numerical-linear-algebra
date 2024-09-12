use std::{
    fmt::Display,
    ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use num_complex::Complex64;
use tabled::{
    builder::Builder,
    settings::{Margin, Style},
};

#[derive(Debug, Clone)]
pub struct Matrix<const M: usize, const N: usize> {
    data: [[Complex64; N]; M],
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    #[inline]
    pub const fn new(data: [[Complex64; N]; M]) -> Self {
        Self { data }
    }

    pub const ZEROS: Self = Matrix::new([[Complex64::ZERO; N]; M]);
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
        for i in 0..M {
            for j in 0..N {
                self[[i, j]] += rhs[[i, j]];
            }
        }
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

impl<const M: usize, const N: usize> SubAssign for Matrix<M, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self[[i, j]] -= rhs[[i, j]];
            }
        }
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
        for i in 0..M {
            for j in 0..N {
                self[[i, j]] *= rhs;
            }
        }
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
