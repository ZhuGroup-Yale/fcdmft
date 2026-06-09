"""Least-squares Fourier transform weights on minimax grids."""

import numpy as np


def _build_log_energy_grid(e_min, e_max, min_num_points, num_points_per_magnitude=200):
    """Build a logarithmic auxiliary energy grid.
    
    Parameters
    ----------
    e_min : float
        Lower bound of the auxiliary energy interval.
    e_max : float
        Upper bound of the auxiliary energy interval.
    min_num_points : int
        Minimum number of grid points to generate.
    num_points_per_magnitude : int, optional
        Number of points used per order of magnitude.

    Returns
    -------
    numpy.ndarray
        Logarithmically spaced auxiliary energy grid.

    Raises
    ------
    ValueError
        If the energy bounds do not define a positive interval.
    """
    if e_min <= 0.0 or e_max <= 0.0 or e_max < e_min:
        raise ValueError("Expected 0 < E_min <= E_max when building the auxiliary grid")

    num_energy_nodes = (int(np.log10(e_max / e_min)) + 1) * num_points_per_magnitude
    num_energy_nodes = max(num_energy_nodes, min_num_points)

    if num_energy_nodes == 1:
        return np.array([e_min], dtype=np.float64)

    grid_scale = np.power(e_max / e_min, 1.0 / (float(num_energy_nodes) - 1.0))
    return e_min * np.power(grid_scale, np.arange(num_energy_nodes, dtype=np.float64))


def _get_max_abs_residual(design_matrix, weight_vector, target_values):
    """Return the maximum absolute residual for a least-squares fit.

    Parameters
    ----------
    design_matrix : numpy.ndarray
        Matrix used in the least-squares fit.
    weight_vector : numpy.ndarray
        Fitted weight vector.
    target_values : numpy.ndarray
        Right-hand side of the least-squares problem.

    Returns
    -------
    float
        Maximum absolute residual.
    """
    fitted_values = design_matrix @ weight_vector
    return float(np.max(np.abs(target_values - fitted_values)))


def _solve_least_squares_svd(design_matrix, target_values):
    """Solve an overdetermined linear system with a reduced SVD.

    Parameters
    ----------
    design_matrix : numpy.ndarray
        Matrix of the linear least-squares problem.
    target_values : numpy.ndarray
        Right-hand side vector.

    Returns
    -------
    numpy.ndarray
        Least-squares solution vector.
    """
    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        design_matrix,
        full_matrices=False,
    )
    transformed_targets = left_vectors.T @ target_values
    scaled_targets = transformed_targets / singular_values
    return right_vectors_t.T @ scaled_targets


def _fit_transform_weights(base_matrix, row_factors, target_rows):
    """Solve a family of related least-squares systems.

    Parameters
    ----------
    base_matrix : numpy.ndarray
        Shared part of the design matrix with shape ``(n_x, n_grid)``.
    row_factors : numpy.ndarray
        Row-specific multiplicative factors with shape ``(n_systems, n_grid)``.
    target_rows : numpy.ndarray
        Right-hand sides with shape ``(n_systems, n_x)``.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Weight matrix and the maximum fit residual across all systems.
    """
    num_systems, num_grid_points = row_factors.shape
    weights = np.empty((num_systems, num_grid_points), dtype=np.float64)
    max_residuals = np.empty(num_systems, dtype=np.float64)

    for system_index, row_factor in enumerate(row_factors):
        design_matrix = base_matrix * row_factor[None, :]
        target_values = target_rows[system_index]
        weight_vector = _solve_least_squares_svd(design_matrix, target_values)
        weights[system_index, :] = weight_vector
        max_residuals[system_index] = _get_max_abs_residual(
            design_matrix,
            weight_vector,
            target_values,
        )

    return weights, float(np.max(max_residuals))


def get_l_sq_wghts_cos_tf_t_to_w(tau_tj, omega_tj, E_min, E_max, num_points_per_magnitude=200):
    """Compute cosine-transform weights from the time grid to the frequency grid.

    Parameters
    ----------
    tau_tj : numpy.ndarray
        Imaginary-time quadrature grid.
    omega_tj : numpy.ndarray
        Imaginary-frequency quadrature grid.
    E_min : float
        Lower bound of the auxiliary energy interval.
    E_max : float
        Upper bound of the auxiliary energy interval.
    num_points_per_magnitude : int, optional
        Number of auxiliary fit points used per decade.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Cosine-transform weights with shape ``(n_omega, n_tau)`` and the maximum
        absolute fit residual.
    """
    tau_grid = np.asarray(tau_tj, dtype=np.float64)
    omega_grid = np.asarray(omega_tj, dtype=np.float64)
    energy_grid = _build_log_energy_grid(
        E_min,
        E_max,
        tau_grid.size,
        num_points_per_magnitude=num_points_per_magnitude,
    )

    energy_sq = np.square(energy_grid)
    omega_sq = np.square(omega_grid)
    exponential_decay = np.exp(-energy_grid[:, None] * tau_grid[None, :])
    cosine_phases = np.cos(np.multiply.outer(omega_grid, tau_grid))
    target_rows = 2.0 * energy_grid[None, :] / (energy_sq[None, :] + omega_sq[:, None])

    return _fit_transform_weights(exponential_decay, cosine_phases, target_rows)


def get_l_sq_wghts_cos_tf_w_to_t(tau_tj, omega_tj, E_min, E_max, num_points_per_magnitude=200):
    """Compute cosine-transform weights from the frequency grid to the time grid.

    Parameters
    ----------
    tau_tj : numpy.ndarray
        Imaginary-time quadrature grid.
    omega_tj : numpy.ndarray
        Imaginary-frequency quadrature grid.
    E_min : float
        Lower bound of the auxiliary energy interval.
    E_max : float
        Upper bound of the auxiliary energy interval.
    num_points_per_magnitude : int, optional
        Number of auxiliary fit points used per decade.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Cosine-transform weights with shape ``(n_tau, n_omega)`` and the maximum
        absolute fit residual.
    """
    tau_grid = np.asarray(tau_tj, dtype=np.float64)
    omega_grid = np.asarray(omega_tj, dtype=np.float64)
    energy_grid = _build_log_energy_grid(
        E_min,
        E_max,
        tau_grid.size,
        num_points_per_magnitude=num_points_per_magnitude,
    )

    energy_sq = np.square(energy_grid)
    omega_sq = np.square(omega_grid)
    denominator = energy_sq[:, None] + omega_sq[None, :]
    cosine_kernel = 2.0 * energy_grid[:, None] / denominator
    cosine_phases = np.cos(np.multiply.outer(tau_grid, omega_grid))
    target_rows = np.exp(-np.multiply.outer(tau_grid, energy_grid))

    return _fit_transform_weights(cosine_kernel, cosine_phases, target_rows)


def get_l_sq_wghts_sin_tf_t_to_w(tau_tj, omega_tj, E_min, E_max, num_points_per_magnitude=200):
    """Compute sine-transform weights from the time grid to the frequency grid.

    Parameters
    ----------
    tau_tj : numpy.ndarray
        Imaginary-time quadrature grid.
    omega_tj : numpy.ndarray
        Imaginary-frequency quadrature grid.
    E_min : float
        Lower bound of the auxiliary energy interval.
    E_max : float
        Upper bound of the auxiliary energy interval.
    num_points_per_magnitude : int, optional
        Number of auxiliary fit points used per decade.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Sine-transform weights with shape ``(n_omega, n_tau)`` and the maximum
        absolute fit residual.
    """
    tau_grid = np.asarray(tau_tj, dtype=np.float64)
    omega_grid = np.asarray(omega_tj, dtype=np.float64)
    energy_grid = _build_log_energy_grid(
        E_min,
        E_max,
        tau_grid.size,
        num_points_per_magnitude=num_points_per_magnitude,
    )

    energy_sq = np.square(energy_grid)
    omega_sq = np.square(omega_grid)
    exponential_decay = np.exp(-energy_grid[:, None] * tau_grid[None, :])
    sine_phases = np.sin(np.multiply.outer(omega_grid, tau_grid))
    target_rows = 2.0 * omega_grid[:, None] / (energy_sq[None, :] + omega_sq[:, None])

    return _fit_transform_weights(exponential_decay, sine_phases, target_rows)


def get_l_sq_wghts_sin_tf_w_to_t(tau_tj, omega_tj, E_min, E_max, num_points_per_magnitude=200):
    """Compute sine-transform weights from the frequency grid to the time grid.

    Parameters
    ----------
    tau_tj : numpy.ndarray
        Imaginary-time quadrature grid.
    omega_tj : numpy.ndarray
        Imaginary-frequency quadrature grid.
    E_min : float
        Lower bound of the auxiliary energy interval.
    E_max : float
        Upper bound of the auxiliary energy interval.
    num_points_per_magnitude : int, optional
        Number of auxiliary fit points used per decade.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Sine-transform weights with shape ``(n_tau, n_omega)`` and the maximum
        absolute fit residual.
    """
    tau_grid = np.asarray(tau_tj, dtype=np.float64)
    omega_grid = np.asarray(omega_tj, dtype=np.float64)
    energy_grid = _build_log_energy_grid(
        E_min,
        E_max,
        tau_grid.size,
        num_points_per_magnitude=num_points_per_magnitude,
    )

    energy_sq = np.square(energy_grid)
    omega_sq = np.square(omega_grid)
    denominator = energy_sq[:, None] + omega_sq[None, :]
    sine_kernel = 2.0 * omega_grid[None, :] / denominator
    sine_phases = np.sin(np.multiply.outer(tau_grid, omega_grid))
    target_rows = np.exp(-np.multiply.outer(tau_grid, energy_grid))

    return _fit_transform_weights(sine_kernel, sine_phases, target_rows)
