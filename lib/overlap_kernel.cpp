#include <eigen_helper.h>
#include <overlap_kernel.h>

namespace quda
{
  /**
   * @brief Calculates the Chebyshev polynomial of the first kind, T_n(x).
   * Uses trigonometric definition for |x|<=1, and recurrence relation for |x|>1.
   */
  double Tn(double x, int n)
  {
    if (abs(x) <= 1.0) { return cos(n * std::acos(x)); }

    double T0 = 1.0, T1 = x, Tk = 2 * x * x - 1;
    switch (n) {
    case 0: return T0;
    case 1: return T1;
    case 2: return Tk;
    default:
      for (int k = 3; k <= n; ++k) {
        T0 = T1;
        T1 = Tk;
        Tk = 2 * x * T1 - T0;
      }
      return Tk;
    }
  }

  /**
   * @brief Evaluates sum c_i * T_i(x) using the stable Clenshaw algorithm.
   */
  double ciTi(double x, const std::vector<double> &c)
  {
    double b2 = 0.0, b1 = 0.0, bk;
    int n = c.size() - 1;
    for (int k = n; k >= 1; --k) {
      bk = c[k] + 2 * x * b1 - b2;
      b2 = b1;
      b1 = bk;
    }
    return c[0] + x * b1 - b2;
  }

  /**
   * @brief Evaluates the derivative of the Chebyshev sum, also using Clenshaw algorithm.
   */
  double iciUim1(double x, const std::vector<double> &c)
  {
    double b2 = 0.0, b1 = 0.0, bk;
    int n = c.size() - 1;
    if (n < 1) return 0.0;
    for (int k = n - 1; k >= 1; --k) {
      bk = (k + 1) * c[k + 1] + 2 * x * b1 - b2;
      b2 = b1;
      b1 = bk;
    }
    return c[1] + 2 * x * b1 - b2;
  }

  /**
   * @brief Calculates the residual error R(x) = 1 - sqrt(x)*P(z) or its derivative.
   */
  double residual(double x, const std::vector<double> &c, double epsilon, bool derivative)
  {
    // Map approximation interval [epsilon, 1] to Chebyshev domain [-1, 1].
    const double z = (x * 2 - (1 + epsilon)) / (1 - epsilon);
    if (derivative) {
      // R'(x)
      return -1 / (2 * sqrt(x)) * ciTi(z, c) - sqrt(x) * iciUim1(z, c) * (2 / (1 - epsilon));
    } else {
      // R(x)
      return 1 - sqrt(x) * ciTi(z, c);
    }
  }

  /**
   * @brief Finds a root in [x_l, x_r] using the Secant method.
   */
  double findRoot(double x_l, double x_r, const std::vector<double> &c, double epsilon, bool derivative)
  {
    double x_m, res_r, res_l, res_m;

    res_l = residual(x_l, c, epsilon, derivative);
    res_r = residual(x_r, c, epsilon, derivative);
    if (abs(res_l) < 1e-15) return x_l;
    if (abs(res_r) < 1e-15) return x_r;
    if (res_r * res_l > 0) {
      errorQuda("ERROR: findRoot with derivative=%d called with wrong ends: (%e %e)->(%e %e)\n", derivative, x_l, x_r,
                res_l, res_r);
      return (x_l + x_r) / 2.0;
    }
    for (int i = 0; i < 20; i++) {
      x_m = (res_l * x_r - res_r * x_l) / (res_l - res_r);
      res_m = residual(x_m, c, epsilon, derivative);
      if (res_m * res_l > 0) {
        x_l = x_m;
        res_l = res_m;
      } else {
        x_r = x_m;
        res_r = res_m;
      }
    }
    return (res_l * x_r - res_r * x_l) / (res_l - res_r);
  }

  /**
   * @brief Core implementation of the Remez algorithm for a fixed order 'n'.
   */
  bool minimaxApproximationRemezImpl(std::vector<double> &c_out, double delta, double epsilon, int n)
  {
    constexpr int max_iter = 5;
    std::vector<double> y(n + 1), z(n + 1), c(n), b(n + 1);
    std::vector<double> c_system_solution(n + 1);

    // Use Eigen::Map to treat raw C++ vectors as Eigen objects without copy.
    Eigen::Map<Eigen::VectorXd> b_eigen(b.data(), b.size());
    Eigen::Map<Eigen::VectorXd> c_eigen(c_system_solution.data(), c_system_solution.size());
    Eigen::MatrixXd M_eigen(n + 1, n + 1);

    // Initial guess for extrema are the extrema of T_n(x).
    for (int i = 0; i < n + 1; ++i) {
      z[i] = cos(M_PI * i / n);
      y[i] = (z[i] * (1 - epsilon) + (1 + epsilon)) / 2;
    }

    int iter = 0;
    while (iter < max_iter) {
      // Step 1: Solve the linear system for new coefficients 'c' and error 'E'.
      for (int i = 0; i < n + 1; ++i) {
        for (int j = 0; j < n; ++j) { M_eigen(i, j) = sqrt(y[i]) * Tn(z[i], j); }
        M_eigen(i, n) = i % 2 == 0 ? 1 : -1; // Last column is for the alternating error term E.
        b_eigen(i) = 1.0;
      }
      c_eigen = M_eigen.lu().solve(b_eigen);

      std::copy(c_system_solution.begin(), c_system_solution.begin() + n, c.begin());

      // Step 2: Find the new extrema of the error function.
      std::vector<double> roots(n - 1);
      for (int i = 0; i < n - 1; ++i) { roots[i] = findRoot(y[i + 1], y[i], c, epsilon, false); }
      for (int i = 0; i < n - 2; ++i) { y[i + 1] = findRoot(roots[i + 1], roots[i], c, epsilon, true); }

      // Step 3: Update z points from new y points.
      for (int i = 0; i < n + 1; ++i) { z[i] = (2 * y[i] - (1 + epsilon)) / (1 - epsilon); }

      // Step 4: Check for convergence.
      double current_max_error = 0.0;
      for (int i = 0; i < n + 1; ++i) {
        b[i] = abs(residual(y[i], c, epsilon, false));
        if (b[i] > current_max_error) current_max_error = b[i];
      }

      if (current_max_error <= delta) {
        c_out = c;
        return true; // Converged
      }
      iter += 1;
    }

    return false; // Failed to converge
  }

  /**
   * @brief High-level wrapper to find the optimal minimax polynomial.
   * It performs an adaptive search for the minimum required polynomial order 'n'.
   */
  std::vector<double> minimaxApproximationRemez(double delta, double epsilon)
  {
    printfQuda("minimaxApproximationRemez called with -> delta: %e, epsilon: %e\n", delta, epsilon);

    // 1. Estimate a reference order.
    int n_ref = static_cast<int>(ceil(-log(delta / 0.41) / (2.083 * sqrt(epsilon)))) + 3;

    std::vector<double> final_coeffs;

    // 2. Loop with increasing order 'n' until the core routine converges.
    for (int n = n_ref; n < n_ref * 1.2 && n < n_ref + 20; ++n) {
      if (minimaxApproximationRemezImpl(final_coeffs, delta, epsilon, n)) {
        printfQuda("minimaxApproximationRemez converged with order n = %d\n", n);
        return final_coeffs;
      }
    }

    errorQuda("minimaxApproximationRemez can not converge");
    return {};
  }

  /**
   * @brief Main constructor: pre-calculates Remez coefficients for given tolerances.
   */
  OverlapKernel::OverlapKernel(std::vector<ColorSpinorField> &evecs, const std::vector<Complex> &evals, double kappa,
                               const std::vector<double> remez_tol) :
    evals(evals.size()),
    kappa(kappa),
    epsilon(pow(evals.back().real() / (1.0 + 8.0 * kappa), 2)),
    remez_tol(remez_tol),
    remez_coeff(remez_tol.size()),
    remez_order(remez_tol.size())
  {
    this->evecs = std::move(evecs);
    for (size_t i = 0; i < evals.size(); i++) { this->evals[i] = evals[i].real(); }
    for (size_t i = 0; i < remez_tol.size(); i++) {
      remez_coeff[i] = minimaxApproximationRemez(remez_tol[i], epsilon);
      remez_order[i] = remez_coeff[i].size() - 1;
    }
  }

  /**
   * @brief Copy constructor: creates a new kernel instance, possibly with a new precision.
   */
  OverlapKernel::OverlapKernel(const OverlapKernel *overlap_kernel, QudaPrecision precision) :
    evals(overlap_kernel->evals),
    kappa(overlap_kernel->kappa),
    epsilon(overlap_kernel->epsilon),
    remez_tol(overlap_kernel->remez_tol),
    remez_coeff(overlap_kernel->remez_coeff),
    remez_order(overlap_kernel->remez_order)
  {
    ColorSpinorParam param(overlap_kernel->evecs[0]);
    param.setPrecision(precision, precision, true);
    evecs.resize(overlap_kernel->evecs.size(), ColorSpinorField(param));
    for (size_t i = 0; i < overlap_kernel->evecs.size(); i++) { evecs[i].copy(overlap_kernel->evecs[i]); }
  }
} // namespace quda