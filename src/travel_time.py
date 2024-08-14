import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def get_const_layer_dxdt(p, h, v):
    """
    Calculate horizontal distance and travel time of a ray in constant velocity layer 

    Parameters
    -------
    p : float
        ray parameter or hoizontal slowness in s/km
    h : float
        layer thickness in km
    v : float
        layer velocity in km / s

    Returns
    -------
    dx : float
        range offset in km
    dt : float
        travel time in s
    """
    # convert velocity to slowness
    u = 1. / v
    # calc. vertical slowness 
    eta = np.sqrt(u**2 - p**2)
    # calc. range offset and travel time
    dx = h * p / eta
    dt = h * u**2 / eta
    return dx, dt


def prop_ray(p, h_i, v_i):
    """
    Calculate horizontal distance and travel time of a ray 
    through a set of constant velocity layers 

    Parameters
    -------
    p : float
        ray parameter or hoizontal slowness in s/km
    h_i : array
        layer thicknesses in km
    v : array
        layer velocities in km / s

    Returns
    -------
    X : float
        Total ange offset in km
    T : float
        Total travel time in s
    """
    X, T = 0., 0.
    for ii, (h, v) in enumerate(zip(h_i, v_i)):
        dx, dt = get_const_layer_dxdt(p, h, v)
        X += dx 
        T += dt
    return X, T


def calc_range_offset_wTakeoff(theta, h_k, lmd_k):
    """
    Calculate horizontal distance (aka range offset) of a ray 
    through a set of constant velocity layers 

    Parameters
    -------
    theta : float
        takeoff angle in radians
    h_i : array
        layer thicknesses in km
    lmd_i : array
        layer velocity ratios(aka unitlesss normalized layer velocities v_i / v_source

    Returns
    -------
    X : float
        Total range offset in km
    """
    sin_theta = np.sin(theta)
    numer = lmd_k * h_k * sin_theta
    denom = np.sqrt(1 - (lmd_k * sin_theta)**2)
    return np.sum(numer / denom)


def get_equivalent_layer_thicknesses(layer_depths, src_depth, path_type='direct'):
    """
    Calculate the vertical distance traveled by rays through each layer for direct
    and refelcted wave paths.

    Parameters
    -------
    layer_depths : float
        depth of vel. model interface in km
    src_depth : array
        depth of source in km
    path_type : str ['direct' or 'reflected']
        determines if the path is an upgoing wave direct from source to receiver
        or a downgoing wave which is assumed to reflect off the bottom layer

    Returns
    -------
    h_k : array
        Vertical distance in km traveled by a ray in each layer
    """
    # Placeholder for retruned array
    h_k = np.zeros_like(layer_depths)
    # Calc. layer thicknesses - be carefull with indexing change
    finite_layer_thicknesses = np.diff(layer_depths)
    l = len(finite_layer_thicknesses)
    # Get index of layer containing the event "src_idx"
    src_idx = np.argwhere(layer_depths < src_depth)[-1][0]
    # Get distance to the top of layer containing the source
    src_layer_above = src_depth - layer_depths[src_idx]
    # Store thicknesses of all layers above the source layer
    h_k[0:src_idx] = finite_layer_thicknesses[0:src_idx]
    # Downgoing or upgoing rays
    if path_type=='direct':
        # Only include the portion of the source layer above the source 
        h_k[src_idx] =  src_layer_above
        # Exclude layers below the source
        h_k[src_idx+1:l+1] = 0
    elif path_type=='reflected':
        assert src_depth < layer_depths[-1], "Source located below deepest interface"
        # Include both the downgoing and upgoing ray paths
        # down - the portion of the source layer below the source
        #        plus the all the layers below
        # up   - all layers below plus the source layer
        h_k[src_idx] =  2 * finite_layer_thicknesses[src_idx] - src_layer_above
        h_k[src_idx + 1:l] = 2 * finite_layer_thicknesses[src_idx+1:l]
    return h_k


def calc_range_offset(q, h_k, lmd_k):
    """
    Calculate horizontal distance (aka range offset) of a ray 
    through a set of constant velocity layers 

    Parameters
    -------
    q : float
        non-dim. ray parameter
        (hTilde_max * q = distance traveled in layer with maximum velocity and
         p = q / (v_max sqrt(1 + q^2) is the traditional ray parameter
    h_k : array
        equivalent thickness of k-th layer in km
    lmd_k : array
        unitless normalized velocity ratio (i.e. v_k / v_max) of k-th layer

    Returns
    -------
    X : float
        Total range offset in km
    """
    numer = q * lmd_k * h_k
    denom = np.sqrt(1 + (1 - lmd_k**2) * q**2)
    return np.sum(numer / denom)


def calc_range_offset_first_deriv(q, h_k, lmd_k):
    """
    Calculate horizontal distance (aka range offset) of a ray 
    through a set of constant velocity layers 

    Parameters
    -------
    q : float
        non-dim. ray parameter
        (hTilde_max * q = distance traveled in layer with maximum velocity and
         p = q / (v_max sqrt(1 + q^2) is the traditional ray parameter
    h_k : array
        equivalent thickness of k-th layer in km
    lmd_k : array
        unitless normalized velocity ratio (i.e. v_k / v_max) of k-th layer

    Returns
    -------
    dX_dq : float
        First derivative of total range offset for ray with takeoff angle theta : dX/dq(q)
    """
    numer = lmd_k * h_k
    denom = np.sqrt(1 + (1 - lmd_k**2) * q**2)**3
    return np.sum(numer / denom)


def calc_range_offset_sec_deriv(q, h_k, lmd_k):
    """
    Calculate first derivative of horizontal distance (aka range offset) 
    of a ray through a set of constant velocity layers 

    Parameters
    -------
    q : float
        non-dim. ray parameter
        (hTilde_max * q = distance traveled in layer with maximum velocity and
         p = q / (v_max sqrt(1 + q^2) is the traditional ray parameter
    h_k : array
        equivalent thickness of k-th layer in km
    lmd_k : array
        unitless normalized velocity ratio (i.e. v_k / v_max) of k-th layer

    Returns
    -------
    ddX_dqdq : float
        Second derivative of total range offset for ray with takeoff angle theta : d^2X/dq^2(q)
    """
    numer = 3 * (lmd_k - 1) * lmd_k * (1 + lmd_k) * q
    denom = np.sqrt(1 + q**2 - lmd_k**2 * q**2)**5
    return np.sum(numer / denom)


def quadratic_newton_iteration(q_i, X_R, h_k, lmd_k):
    """
    Apply second-order Newton Iteration Step to solve for the ray takeoff angle

    Parameters
    -------
    q_i : float
        Current iteration of the approximate non-dim. ray parameter which arrives at X_R
    X_R : float
        Range offset or equivalently epicentral distance in km
    h_k : array
        equivalent thickness of k-th layer in km
    lmd_k : array
        unitless normalized velocity ratio (i.e. v_k / v_max) of k-th layer

    Returns
    -------
    q_new : float
        Updated q (i.e. q + Delta_{q}^{Newton})
    """
    X = calc_range_offset(q_i, h_k, lmd_k)
    A = 0.5 * calc_range_offset_sec_deriv(q_i, h_k, lmd_k)
    B = calc_range_offset_first_deriv(q_i, h_k, lmd_k)
    C = X - X_R
    sqrt_term = np.sqrt(B**2 - 4*A*C)
    plus_delta = (-B + sqrt_term) / (2*A)
    minus_delta = (-B - sqrt_term) / (2*A)
    q_list = q_i + np.array([plus_delta, minus_delta])
    X_list = np.array([calc_range_offset(q_new, h_k, lmd_k) 
                      for q_new in q_list])  
    idx_min = np.argmin(np.abs(X_list - X_R))
    return q_list[idx_min], X_list[idx_min]


def get_max_vel_layer_index(h_k, v_k): 
    max_vel = np.max(v_k[h_k > 0])
    return np.where(v_k==max_vel)[0][0]


def est_asymptotic_q(X_R, h_k, lmd_k, test_mode=False):
    """
    Fit straight line asymptotics as q->0 and q->INF of the form X = m * q + b
    """
    idx_max = np.where(lmd_k == 1.)[0][0]
    h_max = h_k[idx_max]
    m_0 = np.sum(lmd_k * h_k)  # b_0 = 0
    numer = np.delete(lmd_k * h_k, idx_max)
    denom = np.sqrt(1 - np.delete(lmd_k, idx_max)**2)
    m_inf = h_max
    b_inf = np.sum(numer / denom)
    assert not m_0 == m_inf, "problem here"
    X_asym_int = m_0 * b_inf / (m_0 - m_inf)
    if X_R < X_asym_int:
        q_asymptotic = X_R / m_0
    else:
        q_asymptotic = (X_R - b_inf) / m_inf
    if not test_mode:
        return q_asymptotic
    elif test_mode:
        return m_0, m_inf, b_inf, X_asym_int, q_asymptotic


def q_to_p(q, v_max):
    return q / (v_max * np.sqrt(1 + q**2))


def p_to_q(p, v_max):
    """
    Non.-dimensional ray parameter from horizontal ray parameter
    """
    return np.sqrt(p**2 / (v_max**-2 - p**2))


def calc_travel_time(vel_model, epi_dist, src_depth, tol=1.e-4, path_type='direct'):
    """
    Calculate travel times of direct waves using non-dim. 
    ray parameter following Fang and Chen 2019.


    Parameters
    ----------
    vel_model : N-by-2 numpy.array
        Velocity model is assumed to be a latteraly homogeneous with constant 
        velocity (1D).Each row denotes a layer with the first entry denoting
          the depth of the top of the layer in km
        and the second entry the corresponding velocity of the layer in km/s.     
    epi_dist : float
        Epicentral distance between source and receiver in km. 
    src_depth : float
        Depth of source in km
    tol : convergence tolerance
        Max abs. distance ray can arrive from receiver position in km
    path_type : str ['direct' or 'reflected']
        determines if the path is an upgoing wave direct from source to receiver
        or a downgoing wave which is assumed to reflect off the bottom layer

    Returns
    -------
    travel_time : float
        Amount of time it takes the ray to travel from source to receiver

    TODO: Generalize for linear velocity gradient layer, will pass instead
      velocity at top and bottom so that the input model is then N-by-3
    """

    # Hypocentral Distance
    hypo_dist = np.sqrt(epi_dist**2 + src_depth**2)
    
    # Interface depths and velocities (For code readability)
    layer_depths = vel_model[:, 0]
    layer_vels = vel_model[:, 1]

    # Check if in top layer for easy calc. 
    if src_depth < layer_depths[1]:
        # Trivial travel time and takeoff angle calc.
        travel_time = hypo_dist / layer_vels[0]
        return travel_time

    # Equivalent layer thickeness
    layer_thicks = get_equivalent_layer_thicknesses(layer_depths, src_depth, path_type)

    # Only keep layers traversered by the ray
    idx_nonzero = np.where(layer_thicks > 0.)[0]
    h_k, v_k = layer_thicks[idx_nonzero], layer_vels[idx_nonzero]

    # Normalize velocities by max. vel.
    v_max = np.max(v_k)
    # h_max = h_k[np.where(v_k==v_max)[0]][0]
    lmd_k = v_k / v_max     

    q_i = est_asymptotic_q(epi_dist, h_k, lmd_k)
    X_i = calc_range_offset(q_i, h_k, lmd_k)
    while np.abs(epi_dist - X_i) > tol:
        q_i, X_i = quadratic_newton_iteration(q_i, epi_dist, h_k, lmd_k)
    
    # TODO: Derive simple sum for q to get travel time faster
    p = q_to_p(q_i, v_max)
    xi, travel_time = prop_ray(p, h_k, v_k)
    return travel_time


def plot_test_asymptotic(epi_dist=100, src_depth=18, tol=1.e-4):
    vel_model = np.array([[0.0, 5.5], [5.0, 5.8], [10., 6.2], [12.5, 4.0], [15., 6.6], [17.5, 10.],
                        [22, 7.2], [32, 7.9], [42, 8.4]])

    
    # Interface depths and velocities (For code readability)
    layer_depths = vel_model[:, 0]
    layer_vels = vel_model[:, 1]

    # Equivalent layer thickeness
    layer_thicks = get_equivalent_layer_thicknesses(layer_depths, src_depth)

    # Only keep layers traversered by the ray
    idx_nonzero = np.where(layer_thicks > 0.)[0]
    h_k, v_k = layer_thicks[idx_nonzero], layer_vels[idx_nonzero]

    # Normalize velocities by max. vel.
    v_max = np.max(v_k)
    h_max = h_k[np.where(v_k==v_max)[0]][0]
    lmd_k = v_k / v_max   
    
    
    m_0, m_inf, b_inf, X_asym_int, q_est = est_asymptotic_q(epi_dist, h_k, lmd_k,
                                                            test_mode=True)
    
    q_est = est_asymptotic_q(epi_dist, h_k, lmd_k)
    qs = np.linspace(0, 1.5*q_est, 200)
    # Calculate range offset 
    Xs, Xs2 = [], []
    for q in qs:
        Xs.append(calc_range_offset(q, h_k, lmd_k))
    for q in qs:
        p = q_to_p(q, v_max)
        xi, ti = prop_ray(p, h_k, v_k)
        Xs2.append(xi)
    


    fig = plt.figure()
    plt.plot(qs, Xs)
    plt.plot(qs, Xs2, 'ok')
    plt.plot(qs, m_0 * qs)
    plt.plot(qs, m_inf * qs + b_inf)
    plt.axhline(X_asym_int)
    plt.axhline(epi_dist)
    plt.axhline(epi_dist, ls='--', color='k')
    plt.axvline(q_est)
    plt.ylim([0, max([2*X_asym_int, 1.25* epi_dist])])
    plt.savefig('asy_initial_estimate.png', dpi=300)
    plt.close()

    
    q_list = [q_est]
    q_i = q_est
    X_i = calc_range_offset(q_est, h_k, lmd_k)
    while np.abs(epi_dist - X_i) > tol:
        q_i, X_i = quadratic_newton_iteration(q_i, epi_dist, h_k, lmd_k)
        q_list.append(q_i)


    plt.figure()
    for z in layer_depths:
        plt.axhline(z)
    plt.plot(0, src_depth, 'o')
    plt.plot(epi_dist, 0, 'v', ms=10)
    plt.gca().invert_yaxis()

    # iterate of all layers
    X_list = []
    for ii, q in enumerate(q_list):
        p = q_to_p(q, v_max)
        deltas = np.empty_like(v_k)
        for jj, (h, v) in enumerate(zip(h_k, v_k)):
            dx, dt = get_const_layer_dxdt(p, h, v)
            deltas[jj] = dx
        X_list.append(np.sum(deltas))
        xp = np.cumsum(np.flip(np.append(deltas, 0)))
        yp = src_depth - np.cumsum(np.flip(np.append(h_k, 0)))
        plt.plot(xp, yp, '-o', label=ii)
    plt.legend()
    plt.savefig('ray_paths.png', dpi=300)
    plt.close()
