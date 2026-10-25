// Compiled kernels of the uniformization class _unif (unif.py).
//
// _unif computes the matrix exponential of a sub-intensity (or Van Loan
// block) matrix T over many times y at once,
//
//   exp(Ty) = sum_{k=0}^{K} w_k(mu) P^k,   P = T/Gamma + I,  mu = Gamma*y,
//
// with w_k(mu) = exp(-mu) mu^k / k! the Poisson weights. The Python class
// chooses everything that decides the accuracy: Gamma and P, the truncation
// point K from the tolerance, and the error bound it reports. This module does
// the arithmetic that follows from those choices, for every time:
//
//   powersum   the leading rows of exp(Ty), for _unif.run(). Used for the
//              E-steps (the Van Loan blocks of ecph and fitcphreg) and
//              wherever a whole exp(Ty) is needed.
//   vectorsum  the products l exp(Ty) R with a row vector l and the columns of
//              R (or the row l exp(Ty) when R is omitted), for
//              _unif.vectorrun(). Each time costs a weighted sum of the
//              scalars l P^k R instead of a matrix. Used for densities
//              pi exp(Ty) t, survival functions pi exp(Ty) e and the rows
//              pi exp(Ty), in fitcphreg, fitreg, dist and fitcph2dist.
//   normbound  C = max_k ||P^k||_inf (from the row sums P^k e), which bounds
//              the truncation error when P is not sub-stochastic, as for the
//              Van Loan block of a censored E-step.
//   poisson    one Poisson weight, exposed for the tests.
//
// Each time's weights are formed at the mode with Loader's saddle-point
// expansion, and from there outwards by the recurrences
// w_{k+1} = w_k mu/(k+1) and w_{k-1} = w_k k/mu. Only each time's window of
// terms is summed: a tail is left out once a geometric bound on its whole
// weight falls below `cut`, which the class sets so that the two tails change
// exp(Ty) by at most a millionth of the tolerance.
//
// No fast-math: The summation order is fixed, and the results agree with the
// numpy path of the class to rounding, not bit for bit.
//
// To build (from the repository root): python3 -m pip install -e .

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

using Matrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Vector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

namespace {

const double NaN = std::numeric_limits<double>::quiet_NaN();
const double LOG2PI = 1.8378770664093454835606594728112;

// The error of Stirling's formula, log(n!) - log(sqrt(2 pi n) (n/e)^n), for
// n = 0,...,15 (the entry for 0 is not used), computed to 25 digits.
const double SFERR[16] = {
    0.0,
    0.08106146679532725821967026,
    0.04134069595540929409382208,
    0.02767792568499833914878929,
    0.02079067210376509311152277,
    0.01664469118982119216319487,
    0.01387612882307074799874573,
    0.01189670994589177009505572,
    0.01041126526197209649747857,
    0.009255462182712732917728637,
    0.008330563433362871256469319,
    0.007573675487951840794972024,
    0.006942840107209529865664153,
    0.006408994188004207068439631,
    0.005951370112758847735624416,
    0.00555473355196280137103869,
};

// Stirling's error for an integer n >= 1: the table, then the asymptotic series.
double stirlerr(double n) {
    const double S0 = 1.0 / 12.0, S1 = 1.0 / 360.0, S2 = 1.0 / 1260.0;
    const double S3 = 1.0 / 1680.0, S4 = 1.0 / 1188.0;
    if (n <= 15.0) {
        return SFERR[static_cast<int>(n)];
    }
    const double nn = n * n;
    if (n > 500.0) return (S0 - S1 / nn) / n;
    if (n > 80.0) return (S0 - (S1 - S2 / nn) / nn) / n;
    if (n > 35.0) return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n;
    return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n;
}

// x log(x/m) + m - x without the cancellation near x = m.
double bd0(double x, double m) {
    if (std::fabs(x - m) < 0.1 * (x + m)) {
        double v = (x - m) / (x + m);
        double s = (x - m) * v;
        double ej = 2.0 * x * v;
        const double v2 = v * v;
        for (int j = 1; j < 1000; j++) {
            ej *= v2;
            const double s1 = s + ej / (2 * j + 1);
            if (s1 == s) return s1;
            s = s1;
        }
    }
    return x * std::log(x / m) + m - x;
}

// The Poisson probability exp(-mu) mu^k / k! (Loader 2000).
double poisson(int k, double mu) {
    if (k == 0) return std::exp(-mu);
    const double x = static_cast<double>(k);
    return std::exp(-stirlerr(x) - bd0(x, mu) - 0.5 * (LOG2PI + std::log(x)));
}

// One time's window: the weights w[lo..hi] of the terms that are summed.
struct Window {
    int lo;
    int hi;
    std::size_t offset;   // position of w[lo] in the shared weight buffer
};

// Appends the weights of the window of mu to `weights`. The right tail from
// hi+1 has ratios mu/(j+1) <= mu/(hi+2) < 1, the left tail below lo ratios
// j/mu <= (lo-1)/mu < 1, so each tail weighs at most its first term over one
// minus its first ratio; a tail is dropped once that is at most `cut`.
Window window(double mu, int K, double cut, std::vector<double> &weights) {
    int mode = static_cast<int>(std::floor(mu));
    if (mode > K) mode = K;

    std::vector<double> right, left;
    const double wm = poisson(mode, mu);

    double w = wm;
    int hi = mode;
    while (hi < K) {
        const double next = w * mu / (hi + 1);
        const double ratio = mu / (hi + 2);
        if (ratio < 1.0 && next / (1.0 - ratio) <= cut) break;
        right.push_back(next);
        w = next;
        hi++;
    }

    w = wm;
    int lo = mode;
    while (lo > 0) {
        const double next = w * lo / mu;
        const double ratio = (lo - 1) / mu;
        if (next / (1.0 - ratio) <= cut) break;
        left.push_back(next);
        w = next;
        lo--;
    }

    Window out{lo, hi, weights.size()};
    weights.insert(weights.end(), left.rbegin(), left.rend());
    weights.push_back(wm);
    weights.insert(weights.end(), right.begin(), right.end());
    return out;
}

// The windows of every time. As in the numpy path, a time that is not positive
// (not a number included) gets P^0 only, and an infinite one not a number;
// lo = -1 marks both.
std::vector<Window> windows(const double *mus, std::size_t m, int K, double cut,
                            std::vector<double> &weights, int &highest) {
    std::vector<Window> out(m);
    highest = 0;
    for (std::size_t t = 0; t < m; t++) {
        const double mu = mus[t];
        if (!(mu > 0.0)) {
            out[t] = Window{-1, 0, 0};
        } else if (std::isinf(mu)) {
            out[t] = Window{-1, -1, 0};
        } else {
            out[t] = window(mu, K, cut, weights);
            highest = std::max(highest, out[t].hi);
        }
    }
    return out;
}

void checkinputs(const Matrix &P, std::size_t mus, int K) {
    if (P.shape(0) != P.shape(1)) throw std::invalid_argument("The matrix is not square.");
    if (K < 0) throw std::invalid_argument("The truncation point is negative.");
    (void)mus;
}

// The weighted sums of the leading nrows rows of P^k, one block per time.
nb::ndarray<nb::numpy, double, nb::ndim<3>> powersum(Matrix P, Vector mus, int K, int nrows, double cut) {
    checkinputs(P, mus.shape(0), K);
    const std::size_t n = P.shape(0), m = mus.shape(0);
    const std::size_t r = static_cast<std::size_t>(nrows);
    if (nrows < 1 || r > n) throw std::invalid_argument("nrows is out of range.");
    const std::size_t rn = r * n;

    double *out = new double[std::max<std::size_t>(m * rn, 1)];
    nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<double *>(p); });

    {
        nb::gil_scoped_release release;
        const double *p = P.data();
        std::vector<double> weights;
        int highest = 0;
        const std::vector<Window> win = windows(mus.data(), m, K, cut, weights, highest);

        // the leading rows of P^0,...,P^highest
        std::vector<double> powers(static_cast<std::size_t>(highest + 1) * rn, 0.0);
        for (std::size_t i = 0; i < r; i++) powers[i * n + i] = 1.0;
        for (int k = 1; k <= highest; k++) {
            const double *prev = powers.data() + static_cast<std::size_t>(k - 1) * rn;
            double *next = powers.data() + static_cast<std::size_t>(k) * rn;
            for (std::size_t i = 0; i < r; i++) {
                double *row = next + i * n;
                for (std::size_t l = 0; l < n; l++) {
                    const double a = prev[i * n + l];
                    const double *prow = p + l * n;
                    for (std::size_t j = 0; j < n; j++) row[j] += a * prow[j];
                }
            }
        }

        for (std::size_t t = 0; t < m; t++) {
            double *o = out + t * rn;
            const Window &w = win[t];
            if (w.lo == -1) {
                if (w.hi == 0) {
                    std::copy(powers.begin(), powers.begin() + rn, o);
                } else {
                    std::fill(o, o + rn, NaN);
                }
                continue;
            }
            std::fill(o, o + rn, 0.0);
            const double *wk = weights.data() + w.offset;
            for (int k = w.lo; k <= w.hi; k++) {
                const double weight = wk[k - w.lo];
                const double *pk = powers.data() + static_cast<std::size_t>(k) * rn;
                for (std::size_t q = 0; q < rn; q++) o[q] += weight * pk[q];
            }
        }
    }

    return nb::ndarray<nb::numpy, double, nb::ndim<3>>(out, {m, r, n}, owner);
}

// The weighted sums of c_k = l P^k R (R the identity when not given), one row
// per time.
nb::ndarray<nb::numpy, double, nb::ndim<2>> vectorsum(Matrix P, Vector mus, int K, Vector left,
                                                     std::optional<Matrix> right, double cut) {
    checkinputs(P, mus.shape(0), K);
    const std::size_t n = P.shape(0), m = mus.shape(0);
    if (left.shape(0) != n) throw std::invalid_argument("The left vector does not match the matrix.");
    if (right && right->shape(0) != n) throw std::invalid_argument("The right vectors do not match the matrix.");
    const std::size_t q = right ? right->shape(1) : n;

    double *out = new double[std::max<std::size_t>(m * q, 1)];
    nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<double *>(p); });

    {
        nb::gil_scoped_release release;
        const double *p = P.data();
        const double *R = right ? right->data() : nullptr;
        std::vector<double> weights;
        int highest = 0;
        const std::vector<Window> win = windows(mus.data(), m, K, cut, weights, highest);

        // c_k for k = 0,...,highest, from v_k = l P^k
        std::vector<double> c(static_cast<std::size_t>(highest + 1) * q);
        std::vector<double> v(left.data(), left.data() + n), next(n);
        for (int k = 0; k <= highest; k++) {
            double *ck = c.data() + static_cast<std::size_t>(k) * q;
            if (R) {
                std::fill(ck, ck + q, 0.0);
                for (std::size_t l = 0; l < n; l++) {
                    const double a = v[l];
                    const double *rrow = R + l * q;
                    for (std::size_t j = 0; j < q; j++) ck[j] += a * rrow[j];
                }
            } else {
                std::copy(v.begin(), v.end(), ck);
            }
            if (k < highest) {
                std::fill(next.begin(), next.end(), 0.0);
                for (std::size_t l = 0; l < n; l++) {
                    const double a = v[l];
                    const double *prow = p + l * n;
                    for (std::size_t j = 0; j < n; j++) next[j] += a * prow[j];
                }
                v.swap(next);
            }
        }

        for (std::size_t t = 0; t < m; t++) {
            double *o = out + t * q;
            const Window &w = win[t];
            if (w.lo == -1) {
                if (w.hi == 0) {
                    std::copy(c.begin(), c.begin() + q, o);
                } else {
                    std::fill(o, o + q, NaN);
                }
                continue;
            }
            std::fill(o, o + q, 0.0);
            const double *wk = weights.data() + w.offset;
            for (int k = w.lo; k <= w.hi; k++) {
                const double weight = wk[k - w.lo];
                const double *ck = c.data() + static_cast<std::size_t>(k) * q;
                for (std::size_t j = 0; j < q; j++) o[j] += weight * ck[j];
            }
        }
    }

    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(out, {m, q}, owner);
}

// C = max_k ||P^k||_inf over k = 0,...,horizon, from r <- P r with r = e.
double normbound(Matrix P, int horizon) {
    if (P.shape(0) != P.shape(1)) throw std::invalid_argument("The matrix is not square.");
    const std::size_t n = P.shape(0);
    const double *p = P.data();
    nb::gil_scoped_release release;
    std::vector<double> r(n, 1.0), next(n);
    double largest = 1.0;
    for (int k = 0; k < horizon; k++) {
        // the largest row sum, not a number if any is (as numpy's max)
        double value = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; i++) {
            double s = 0.0;
            const double *prow = p + i * n;
            for (std::size_t j = 0; j < n; j++) s += prow[j] * r[j];
            next[i] = s;
            if (std::isnan(s) || s > value) value = std::isnan(value) ? value : s;
        }
        if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
        largest = std::max(largest, value);
        r.swap(next);
    }
    return largest;
}

}  // namespace

NB_MODULE(_unifkernel, module) {
    module.doc() = "Compiled kernels of the uniformization class _unif.";
    module.def("powersum", &powersum, nb::arg("P"), nb::arg("mus"), nb::arg("K"), nb::arg("nrows"),
               nb::arg("cut"));
    module.def("vectorsum", &vectorsum, nb::arg("P"), nb::arg("mus"), nb::arg("K"), nb::arg("left"),
               nb::arg("right").none(), nb::arg("cut"));
    module.def("normbound", &normbound, nb::arg("P"), nb::arg("horizon"));
    module.def("poisson", &poisson, nb::arg("k"), nb::arg("mu"));
}
