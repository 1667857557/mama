//! Allocation-bounded implementation of the numerical analysis in `core_mama.py`.

const TWEAK_FACTOR: f64 = 0.99;

fn min_eigenvalue_symmetric(a: &[f64], n: usize) -> Option<f64> {
    if n == 0 {
        return None;
    }
    let mut x = a.to_vec();
    // Cyclic Jacobi iteration is deliberately used here: unlike Cholesky it
    // also classifies positive-semidefinite matrices with zero eigenvalues.
    let scale = x.iter().fold(0.0_f64, |v, z| v.max(z.abs())).max(1.0);
    let tolerance = f64::EPSILON * scale * n as f64;
    for _ in 0..(50 * n * n).max(1) {
        let mut row = 0;
        let mut col = 0;
        let mut largest = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                if x[i * n + j].abs() > largest {
                    largest = x[i * n + j].abs();
                    row = i;
                    col = j;
                }
            }
        }
        if largest <= tolerance {
            return x.iter().step_by(n + 1).copied().reduce(f64::min);
        }
        let app = x[row * n + row];
        let aqq = x[col * n + col];
        let apq = x[row * n + col];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let (s, c) = angle.sin_cos();
        for k in 0..n {
            if k != row && k != col {
                let akp = x[k * n + row];
                let akq = x[k * n + col];
                let new_p = c * akp - s * akq;
                let new_q = s * akp + c * akq;
                x[k * n + row] = new_p;
                x[row * n + k] = new_p;
                x[k * n + col] = new_q;
                x[col * n + k] = new_q;
            }
        }
        x[row * n + row] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        x[col * n + col] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        x[row * n + col] = 0.0;
        x[col * n + row] = 0.0;
    }
    None
}

fn is_positive_definite(a: &[f64], n: usize) -> bool {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = a[i * n + j];
            for k in 0..j {
                value -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !value.is_finite() || value <= 0.0 {
                    return false;
                }
                l[i * n + j] = value.sqrt();
            } else {
                l[i * n + j] = value / l[j * n + j];
            }
        }
    }
    true
}

/// Implements Python's `tweak_omega` in place, including its 0.99 shrink loop.
pub fn tweak_omega(omega: &mut [f64], p: usize) -> bool {
    if omega.len() != p * p || (0..p).any(|i| omega[i * p + i] <= 0.0) {
        return false;
    }
    let diagonal: Vec<f64> = (0..p).map(|i| omega[i * p + i]).collect();
    for i in 0..p {
        for j in 0..p {
            omega[i * p + j] = omega[i * p + j].min((diagonal[i] * diagonal[j]).sqrt());
        }
    }
    while min_eigenvalue_symmetric(omega, p).is_some_and(|v| v < 0.0) {
        for value in omega.iter_mut() {
            *value *= TWEAK_FACTOR;
        }
        for i in 0..p {
            omega[i * p + i] = diagonal[i];
        }
    }
    min_eigenvalue_symmetric(omega, p).is_some_and(|v| v >= 0.0)
}

/// Equivalent to `qc_omega`; returns `(valid, tweaked)` for one SNP.
pub fn qc_omega(omega: &mut [f64], p: usize) -> (bool, bool) {
    if min_eigenvalue_symmetric(omega, p).is_some_and(|v| v >= 0.0) {
        return (true, false);
    }
    if (0..p).any(|i| omega[i * p + i] <= 0.0) {
        return (false, false);
    }
    (tweak_omega(omega, p), true)
}

fn inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut work = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            work[i * 2 * n + j] = a[i * n + j];
        }
        work[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| {
            work[i * 2 * n + col]
                .abs()
                .total_cmp(&work[j * 2 * n + col].abs())
        })?;
        let pivot_value = work[pivot * 2 * n + col];
        if pivot_value == 0.0 || !pivot_value.is_finite() {
            return None;
        }
        if pivot != col {
            for j in 0..2 * n {
                work.swap(col * 2 * n + j, pivot * 2 * n + j);
            }
        }
        let divisor = work[col * 2 * n + col];
        for j in 0..2 * n {
            work[col * 2 * n + j] /= divisor;
        }
        for i in 0..n {
            if i != col {
                let factor = work[i * 2 * n + col];
                for j in 0..2 * n {
                    work[i * 2 * n + j] -= factor * work[col * 2 * n + j];
                }
            }
        }
    }
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = work[i * 2 * n + n + j];
        }
    }
    Some(result)
}

fn run_one(beta: &[f64], omega: &[f64], sigma: &[f64], p: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut result_beta = vec![0.0; p];
    let mut result_se = vec![0.0; p];
    for target in 0..p {
        let diagonal = omega[target * p + target];
        if diagonal == 0.0 {
            return None;
        }
        let scaled: Vec<f64> = (0..p).map(|j| omega[target * p + j] / diagonal).collect();
        let center_inverse: Vec<f64> = (0..p * p)
            .map(|index| {
                let j = index / p;
                let k = index % p;
                omega[index] + sigma[index] - scaled[j] * omega[target * p + k]
            })
            .collect();
        // Match np.linalg.inv in the reference rather than changing the estimator.
        let center = inverse(&center_inverse, p)?;
        let left: Vec<f64> = (0..p)
            .map(|k| (0..p).map(|j| scaled[j] * center[j * p + k]).sum())
            .collect();
        let denominator: f64 = (0..p).map(|j| left[j] * scaled[j]).sum();
        result_beta[target] = (0..p).map(|j| left[j] * beta[j]).sum::<f64>() / denominator;
        result_se[target] = (1.0 / denominator).sqrt();
    }
    Some((result_beta, result_se))
}

/// Direct counterpart of `run_mama_method`, using R column-major inputs.
pub fn run(
    betas: &[f64],
    omega: &[f64],
    sigma: &[f64],
    m: usize,
    p: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if betas.len() != m * p || omega.len() != m * p * p || sigma.len() != omega.len() {
        return None;
    }
    let mut rb = vec![0.0; m * p];
    let mut rs = vec![0.0; m * p];
    for snp in 0..m {
        let b: Vec<f64> = (0..p).map(|j| betas[snp + m * j]).collect();
        let o: Vec<f64> = (0..p * p).map(|x| omega[snp + m * x]).collect();
        let s: Vec<f64> = (0..p * p).map(|x| sigma[snp + m * x]).collect();
        let (ob, os) = run_one(&b, &o, &s, p)?;
        for j in 0..p {
            rb[snp + m * j] = ob[j];
            rs[snp + m * j] = os[j];
        }
    }
    Some((rb, rs))
}

/// Full post-regression analysis: create Omega/Sigma, reproduce QC and identity
/// substitution, and run the estimator without allocating M matrices at once.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn analyze(
    betas: &[f64],
    ses: &[f64],
    ldscores: &[f64],
    ld_coef: &[f64],
    const_coef: &[f64],
    se2_coef: &[f64],
    m: usize,
    p: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>)> {
    if betas.len() != m * p
        || ses.len() != m * p
        || ldscores.len() != m * p * p
        || [ld_coef, const_coef, se2_coef]
            .iter()
            .any(|x| x.len() != p * p)
    {
        return None;
    }
    let mut rb = vec![0.0; m * p];
    let mut rs = vec![0.0; m * p];
    let mut keep = vec![0; m];
    let mut tweaked = vec![0; m];
    for snp in 0..m {
        let b: Vec<f64> = (0..p).map(|j| betas[snp + m * j]).collect();
        let se: Vec<f64> = (0..p).map(|j| ses[snp + m * j]).collect();
        let mut omega: Vec<f64> = (0..p * p)
            .map(|x| ldscores[snp + m * x] * ld_coef[x])
            .collect();
        let mut sigma: Vec<f64> = (0..p * p)
            .map(|x| {
                let i = x / p;
                let j = x % p;
                se[i] * se[j] * se2_coef[x] + const_coef[x]
            })
            .collect();
        let (omega_ok, was_tweaked) = if p == 1 {
            (true, false)
        } else {
            qc_omega(&mut omega, p)
        };
        let sigma_ok = is_positive_definite(&sigma, p);
        if !omega_ok || !sigma_ok {
            omega.fill(0.0);
            sigma.fill(0.0);
            for i in 0..p {
                omega[i * p + i] = 1.0;
                sigma[i * p + i] = 1.0;
            }
        } else {
            keep[snp] = 1;
        }
        tweaked[snp] = i32::from(was_tweaked);
        let (ob, os) = run_one(&b, &omega, &sigma, p)?;
        for j in 0..p {
            rb[snp + m * j] = ob[j];
            rs[snp + m * j] = os[j];
        }
    }
    Some((rb, rs, keep, tweaked))
}

fn least_squares(design: &[f64], response: &[f64], rows: usize, cols: usize) -> Option<Vec<f64>> {
    if design.len() != rows * cols || response.len() != rows {
        return None;
    }
    if cols == 0 {
        return Some(Vec::new());
    }
    // One-sided Jacobi SVD.  This retains the minimum-norm/rank-deficient
    // semantics of numpy.linalg.lstsq without forming X'X.
    let mut a = design.to_vec();
    let mut v = vec![0.0; cols * cols];
    for i in 0..cols {
        v[i * cols + i] = 1.0;
    }
    for _ in 0..(100 * cols * cols) {
        let mut changed = false;
        for p in 0..cols {
            for q in p + 1..cols {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..rows {
                    let x = a[i * cols + p];
                    let y = a[i * cols + q];
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma.abs() <= f64::EPSILON.sqrt() * (alpha * beta).sqrt() {
                    continue;
                }
                changed = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..rows {
                    let x = a[i * cols + p];
                    let y = a[i * cols + q];
                    a[i * cols + p] = c * x - s * y;
                    a[i * cols + q] = s * x + c * y;
                }
                for i in 0..cols {
                    let x = v[i * cols + p];
                    let y = v[i * cols + q];
                    v[i * cols + p] = c * x - s * y;
                    v[i * cols + q] = s * x + c * y;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let norms: Vec<f64> = (0..cols)
        .map(|j| {
            (0..rows)
                .map(|i| a[i * cols + j].powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let largest = norms.iter().copied().fold(0.0, f64::max);
    let cutoff = f64::EPSILON * rows.max(cols) as f64 * largest;
    let projected: Vec<f64> = (0..cols)
        .map(|j| {
            if norms[j] > cutoff {
                (0..rows)
                    .map(|i| a[i * cols + j] * response[i])
                    .sum::<f64>()
                    / (norms[j] * norms[j])
            } else {
                0.0
            }
        })
        .collect();
    Some(
        (0..cols)
            .map(|i| (0..cols).map(|j| v[i * cols + j] * projected[j]).sum())
            .collect(),
    )
}

fn regression(
    y: &[f64],
    x: &[f64],
    weights: &[f64],
    fixed: &[f64],
    rows: usize,
    cols: usize,
) -> Option<Vec<f64>> {
    if y.len() != rows || x.len() != rows * cols || weights.len() != rows || fixed.len() != cols {
        return None;
    }
    if weights.iter().any(|w| *w < 0.0) {
        return None;
    }
    let free: Vec<usize> = (0..cols).filter(|j| fixed[*j].is_nan()).collect();
    let mut adjusted = y.to_vec();
    for i in 0..rows {
        for j in 0..cols {
            if !fixed[j].is_nan() {
                adjusted[i] -= x[i * cols + j] * fixed[j];
            }
        }
    }
    let mut design = vec![0.0; rows * free.len()];
    for i in 0..rows {
        let w = weights[i].sqrt();
        adjusted[i] *= w;
        for (k, j) in free.iter().enumerate() {
            design[i * free.len() + k] = w * x[i * cols + *j];
        }
    }
    let fitted = least_squares(&design, &adjusted, rows, free.len())?;
    let mut result = fixed.to_vec();
    for (k, j) in free.iter().enumerate() {
        result[*j] = fitted[k];
    }
    Some(result)
}

/// Reproduces `run_ldscore_regressions` for explicit fixed-coefficient
/// matrices (`NaN` means unconstrained), followed by the complete analysis.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn fit(
    betas: &[f64],
    ses: &[f64],
    ld: &[f64],
    ld_fixed: &[f64],
    const_fixed: &[f64],
    se_fixed: &[f64],
    m: usize,
    p: usize,
) -> Option<(
    Vec<f64>,
    Vec<f64>,
    Vec<i32>,
    Vec<i32>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    if betas.len() != m * p
        || ses.len() != m * p
        || ld.len() != m * p * p
        || [ld_fixed, const_fixed, se_fixed]
            .iter()
            .any(|x| x.len() != p * p)
    {
        return None;
    }
    let mut lc = vec![0.0; p * p];
    let mut cc = vec![0.0; p * p];
    let mut sc = vec![0.0; p * p];
    for p1 in 0..p {
        for p2 in p1..p {
            let mut x = vec![0.0; m * 3];
            let mut y = vec![0.0; m];
            let mut weights = vec![0.0; m];
            for snp in 0..m {
                let score = ld[snp + m * (p1 + p * p2)];
                x[snp * 3] = score;
                x[snp * 3 + 1] = 1.0;
                x[snp * 3 + 2] = ses[snp + m * p1] * ses[snp + m * p2];
                y[snp] = betas[snp + m * p1] * betas[snp + m * p2];
                weights[snp] = if p1 == p2 {
                    if score > 1.0 {
                        1.0 / score
                    } else if score > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                } else if score != 0.0 {
                    (1.0 / score).abs()
                } else {
                    0.0
                };
            }
            let index = p1 * p + p2;
            let fixed = [ld_fixed[index], const_fixed[index], se_fixed[index]];
            let coef = regression(&y, &x, &weights, &fixed, m, 3)?;
            for &(i, j) in &[(p1, p2), (p2, p1)] {
                lc[i * p + j] = coef[0];
                cc[i * p + j] = coef[1];
                sc[i * p + j] = coef[2];
            }
        }
    }
    let (b, se, k, t) = analyze(betas, ses, ld, &lc, &cc, &sc, m, p)?;
    Some((b, se, k, t, lc, cc, sc))
}

#[unsafe(no_mangle)]
/// C ABI for the low-level estimator.
///
/// # Safety
///
/// All pointers must address non-overlapping buffers with the lengths implied
/// by `m` and `p`; output buffers must be writable.
pub unsafe extern "C" fn mamars_run(
    b: *const f64,
    o: *const f64,
    s: *const f64,
    m: usize,
    p: usize,
    rb: *mut f64,
    rs: *mut f64,
) -> i32 {
    if [b, o, s].iter().any(|x| x.is_null()) || rb.is_null() || rs.is_null() {
        return -1;
    }
    let Some((x, y)) = run(
        unsafe { std::slice::from_raw_parts(b, m * p) },
        unsafe { std::slice::from_raw_parts(o, m * p * p) },
        unsafe { std::slice::from_raw_parts(s, m * p * p) },
        m,
        p,
    ) else {
        return 1;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), rb, x.len());
        std::ptr::copy_nonoverlapping(y.as_ptr(), rs, y.len());
    }
    0
}

#[unsafe(no_mangle)]
/// C ABI for full post-regression analysis.
///
/// # Safety
///
/// All pointers must address non-overlapping buffers with the lengths implied
/// by `m` and `p`; output buffers must be writable.
pub unsafe extern "C" fn mamars_analyze(
    b: *const f64,
    se: *const f64,
    ld: *const f64,
    lc: *const f64,
    cc: *const f64,
    sc: *const f64,
    m: usize,
    p: usize,
    rb: *mut f64,
    rs: *mut f64,
    keep: *mut i32,
    tweak: *mut i32,
) -> i32 {
    if [b, se, ld, lc, cc, sc].iter().any(|x| x.is_null())
        || [rb, rs].iter().any(|x| x.is_null())
        || keep.is_null()
        || tweak.is_null()
    {
        return -1;
    }
    let sl = |x| unsafe { std::slice::from_raw_parts(x, m * p) };
    let ml = |x| unsafe { std::slice::from_raw_parts(x, m * p * p) };
    let cl = |x| unsafe { std::slice::from_raw_parts(x, p * p) };
    let Some((x, y, k, t)) = analyze(sl(b), sl(se), ml(ld), cl(lc), cl(cc), cl(sc), m, p) else {
        return 1;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), rb, x.len());
        std::ptr::copy_nonoverlapping(y.as_ptr(), rs, y.len());
        std::ptr::copy_nonoverlapping(k.as_ptr(), keep, m);
        std::ptr::copy_nonoverlapping(t.as_ptr(), tweak, m);
    }
    0
}

#[unsafe(no_mangle)]
/// C ABI for regression plus analysis.
///
/// # Safety
/// All pointers must reference non-overlapping readable/writable buffers whose
/// lengths follow from `m` and `p`.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn mamars_fit(
    b: *const f64,
    se: *const f64,
    ld: *const f64,
    lf: *const f64,
    cf: *const f64,
    sf: *const f64,
    m: usize,
    p: usize,
    rb: *mut f64,
    rs: *mut f64,
    keep: *mut i32,
    tweak: *mut i32,
    lc: *mut f64,
    cc: *mut f64,
    sc: *mut f64,
) -> i32 {
    if [b, se, ld, lf, cf, sf].iter().any(|x| x.is_null())
        || [rb, rs, lc, cc, sc].iter().any(|x| x.is_null())
        || keep.is_null()
        || tweak.is_null()
    {
        return -1;
    }
    let sl = |x| unsafe { std::slice::from_raw_parts(x, m * p) };
    let ml = |x| unsafe { std::slice::from_raw_parts(x, m * p * p) };
    let cl = |x| unsafe { std::slice::from_raw_parts(x, p * p) };
    let Some((x, y, k, t, l, c, s)) = fit(sl(b), sl(se), ml(ld), cl(lf), cl(cf), cl(sf), m, p)
    else {
        return 1;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), rb, x.len());
        std::ptr::copy_nonoverlapping(y.as_ptr(), rs, y.len());
        std::ptr::copy_nonoverlapping(k.as_ptr(), keep, m);
        std::ptr::copy_nonoverlapping(t.as_ptr(), tweak, m);
        std::ptr::copy_nonoverlapping(l.as_ptr(), lc, l.len());
        std::ptr::copy_nonoverlapping(c.as_ptr(), cc, c.len());
        std::ptr::copy_nonoverlapping(s.as_ptr(), sc, s.len());
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_population_matches_closed_form() {
        let (b, se) = run(&[2.0], &[3.0], &[4.0], 1, 1).unwrap();
        assert_eq!(b, [2.0]);
        assert!((se[0] - 2.0).abs() < 1e-12);
    }
    #[test]
    fn omega_qc_matches_python_examples() {
        let mut valid = vec![2., -1., 0., -1., 2., -1., 0., -1., 2.];
        assert_eq!(qc_omega(&mut valid, 3), (true, false));
        let mut invalid = vec![-2., -1., 0., -1., -2., -1., 0., -1., -2.];
        assert_eq!(qc_omega(&mut invalid, 3), (false, false));
        let mut tweak = vec![2., 2., 2., 2., 1., 3., 2., 3., 2.];
        assert_eq!(qc_omega(&mut tweak, 3), (true, true));
    }
    #[test]
    fn full_analysis_constructs_matrices() {
        let (x, se, k, _) = analyze(&[2.], &[2.], &[3.], &[1.], &[0.], &[1.], 1, 1).unwrap();
        assert_eq!(x, [2.]);
        assert!((se[0] - 2.).abs() < 1e-12);
        assert_eq!(k, [1]);
    }
    #[test]
    fn least_squares_and_fixed_coefficients() {
        let x = [1., 1., 2., 1., 3., 1.];
        let y = [3., 5., 7.];
        let got = regression(&y, &x, &[1.; 3], &[f64::NAN, f64::NAN], 3, 2).unwrap();
        assert!((got[0] - 2.).abs() < 1e-10);
        assert!((got[1] - 1.).abs() < 1e-10);
        let fixed = regression(&y, &x, &[1.; 3], &[2., f64::NAN], 3, 2).unwrap();
        assert_eq!(fixed[0], 2.);
        assert!((fixed[1] - 1.).abs() < 1e-10);
    }
}
