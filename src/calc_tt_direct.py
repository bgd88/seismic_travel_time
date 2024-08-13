import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


epi_dist = 35
src_depth = 28
vel_model = np.array([[0.0, 5.5], [5.0, 5.8], [10., 6.2], [12., 4.2], [15., 6.6],
                      [22, 7.2], [32, 7.9], [42, 8.4]])


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


def calc_range_offset(theta, h_i, lmd_i):
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
    numer = lmd_i * h_i * sin_theta
    denom = np.sqrt(1 - (lmd_i * sin_theta)**2)
    return np.sum(numer / denom)


def calc_range_offset_first_deriv(theta, h_i, lmd_i):
    """
    Calculate first derivative of horizontal distance (aka range offset) 
    of a ray through a set of constant velocity layers 

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
    dX/dtheta : float
        First derivative of total range offset for ray with takeoff angle theta
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    numer = lmd_i * h_i * cos_theta
    denom = np.sqrt(1 - (lmd_i * sin_theta)**2)**3
    return np.sum(numer / denom)


def calc_range_offset_sec_deriv(theta, h_i, lmd_i):
    """
    Calculate first derivative of horizontal distance (aka range offset) 
    of a ray through a set of constant velocity layers 

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
    dX/dtheta : float
        First derivative of total range offset for ray with takeoff angle theta
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    numer1 = - lmd_i * sin_theta
    denom1 = np.sqrt(1 - (lmd_i * sin_theta)**2)**3
    summands1 = numer1 / denom1 
    numer2 = 3 * lmd_i**3 * cos_theta**2 * sin_theta
    denom2 = np.sqrt(1 - (lmd_i * sin_theta)**2)**5
    summands2 = numer2 / denom2 
    return np.sum(h_i * (summands1 + summands2))


def newton_update_takeoff(theta, X_R, h_i, lmd_i):
    """
    Apply Newton Iteration Step to solve for the ray takeoff angle

    Parameters
    -------
    theta : float
        takeoff angle in radians
    X_R : float
        Range offset or equivalently epicentral distance in km
    h_i : array
        layer thicknesses in km
    lmd_i : array
        layer velocity ratios(aka unitlesss normalized layer velocities v_i / v_source

    Returns
    -------
    theta' : float
        Updated takeoff angle (i.e. theta + Delta_{theta}^{Newton})
    """
    X = calc_range_offset(theta, h_i, lmd_i)
    A = 0.5 * calc_range_offset_sec_deriv(theta, h_i, lmd_i)
    B = calc_range_offset_first_deriv(theta, h_i, lmd_i)
    C = X - X_R
    sqrt_term = np.sqrt(B**2 - 4*A*C)
    plus_delta = (-B + sqrt_term) / (2*A)
    minus_delta = (-B - sqrt_term) / (2*A)
    theta_list = theta + np.array([plus_delta, minus_delta])
    new_X = np.array([calc_range_offset(new_theta, h_i, lmd_i) 
                      for new_theta in theta_list])  
    return theta_list[np.argmin(np.abs(new_X - X_R))]

def non_dim_ray_p(p, v_max):
    return np.sqrt(p**2 / (v_max**-2 - p**2))

def nondim_direct_wave_travel_time(vel_model, epi_dist, src_depth):
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

    Returns
    -------
    travel_time : float
        Amount of time it takes the ray to travel from source to receiver
    takeoff_angle : float
        Incidence angle of ray that arrives at the receiver in degrees

    TODO: Generalize for linear velocity gradient layer, will pass instead
      velocity at top and bottom
          so that the input model is then N-by-3
    TODO: Speed-up with quadratic perturbation for initial takeoff angle
    """
    # Convergence Tolerence
    tol = 1.e-3

    # Hypocentral Distance
    hypo_dist = np.sqrt(epi_dist**2 + src_depth**2)
    
    # Interface depths and velocities (For code readability)
    layer_depths = vel_model[:, 0]
    layer_vels = vel_model[:, 1]

    # Get index of layer containing the event "src_idx"
    src_idx = np.argwhere(layer_depths < src_depth)[-1][0]
    src_vel = layer_vels[src_idx]  # Velocity of layer containing the source
    surf_vel = layer_vels[0]  # Velocity of surface layer
    
    # Vel. model layer thicknesses
    layer_thicks = np.diff(layer_depths)
    # Get distance to the top of layer containing the source
    src_lyr_thickness = src_depth - layer_depths[src_idx]
    # Layer velocities and thicknesses traveled by the direct wave
    v_i = layer_vels[0:src_idx+1]  # plus 1 inclues source layer
    h_i = np.append(layer_thicks[0:src_idx], src_lyr_thickness)
    lmd_i = v_i / src_vel
    
    # Initial guess for source takeoff angle
    #
    # Undershoot - Est. source takeoff angle for a homogeneous medium
    takeoff_min = np.arctan2(epi_dist, src_depth)

    # Use Snells' law to estimate angle in each layer for initial 
    # source takeoff angle estimated assuming a homogeneous media  
    theta_i = np.arcsin(v_i * np.sin(takeoff_min)/ np.max(v_i))

    # Calc. weighted ave. velocity using the ray length of each layer 
    ray_length_i = h_i / np.cos(theta_i)
    v_ave = np.sum(ray_length_i * v_i) / np.sum(ray_length_i)

    # Ray parameter of homogeneous case
    ray_p_ave = np.sin(takeoff_min) / v_ave

    # Overshoot - Assume takeoff angle of the maximum layer
    # Check if the estimate works...
    if np.max(v_i) * ray_p_ave < 1.0:
        takeoff_max = np.arcsin( np.max(v_i) * ray_p_ave)
    else:
        takeoff_max = np.pi/2 - 0.01 
          
    # Initial Guess - EQ 11 of Kim and Baag (2002). 
    takeoff_center = (takeoff_min + takeoff_max) / 2  
    takeoff_list = np.array([takeoff_min, takeoff_center, takeoff_max])
    init_idx = np.argmin(np.abs(epi_dist - np.array([calc_range_offset(theta, h_i, lmd_i) 
                                              for theta in takeoff_list]))) 
    theta_init = theta = takeoff_list[init_idx]
    theta_list = [theta_init]

    while np.abs(epi_dist - calc_range_offset(theta, h_i, lmd_i)) > tol:
        theta = newton_update_takeoff(theta, epi_dist, h_i, lmd_i)
        theta_list.append(theta)

    if plot_rays:
    plt.figure()
    for z in layer_depths:
        plt.axhline(z)
    plt.plot(0, src_depth, 'o')
    plt.plot(epi_dist, 0, 'v', ms=10)
    plt.gca().invert_yaxis()

    # iterate of all layers
    X_list = []
    for ii, theta in enumerate(theta_list):
        p = np.sin(theta) / src_vel
        deltas = np.empty_like(v_i)
        for jj, (h, v) in enumerate(zip(h_i, v_i)):
            dx, dt = get_const_layer_dxdt(p, h, v)
            deltas[jj] = dx
        X_list.append(np.sum(deltas))
        xp = np.cumsum(np.flip(np.append(deltas, 0)))
        yp = src_depth - np.cumsum(np.flip(np.append(h_i, 0)))
        plt.plot(xp, yp, '-o', label=ii)
    
    plt.legend()
    plt.savefig('ray_paths.png',)


def direct_wave_travel_time(vel_model, epi_dist, src_depth):
    """
    Calculate travel times of direct waves using ray parameter following Kim and Baag 2002.

    Parameters
    ----------
    vel_model : N-by-2 numpy.array
        Velocity model is assumed to be a latteraly homogeneous with constant velocity (1D).
        Each row denotes a layer with the first entry denoting the depth of the top of 
        the layer in km and the second entry the corresponding velocity of the layer in km/s.     
    epi_dist : float
        Epicentral distance between source and receiver in km. 
    src_depth : float
        Depth of source in km

    Returns
    -------
    travel_time : float
        Amount of time it takes the ray to travel from source to receiver
    takeoff_angle : float
        Incidence angle of ray that arrives at the receiver in degrees

    TODO: Generalize for linear velocity gradient layer, will pass instead velocity
      at top and bottom so that the input model is then N-by-3
    TODO: Speed-up with quadratic perturbation for initial takeoff angle
    """
    # Convergence Tolerence
    tol = 1.e-4

    # Hypocentral Distance
    hypo_dist = np.sqrt(epi_dist**2 + src_depth**2)
    
    # Interface depths and velocities (For code readability)
    layer_depths = vel_model[:, 0]
    layer_vels = vel_model[:, 1]

    # Get index of layer containing the event "src_idx"
    src_idx = np.argwhere(layer_depths < src_depth)[-1][0]
    src_vel = layer_vels[src_idx]  # Velocity of layer containing the source
    surf_vel = layer_vels[0]  # Velocity of surface layer
    
    # Vel. model layer thicknesses
    layer_thicks = np.diff(layer_depths)
    # Get distance to the top of layer containing the source
    src_lyr_thickness = src_depth - layer_depths[src_idx]
    # Layer velocities and thicknesses traveled by the direct wave
    v_i = layer_vels[0:src_idx+1]  # plus 1 inclues source layer
    h_i = np.append(layer_thicks[0:src_idx], src_lyr_thickness)
    
    # Initial guess for source takeoff angle
    #
    # Undershoot - Est. source takeoff angle for a homogeneous medium
    takeoff_min = np.arctan2(epi_dist, src_depth)
    # ray parameter for this takeoff angle
    ray_p_min = np.sin(takeoff_min) / src_vel

    # Use Snells' law to estimate angle in each layer for initial 
    # source takeoff angle estimated assuming a homogeneous media  
    theta_i = np.arcsin(v_i * np.sin(takeoff_min)/ np.max(v_i))

    # Calc. weighted ave. velocity using the ray length of each layer 
    ray_length_i = h_i / np.cos(theta_i)
    v_ave = np.sum(ray_length_i * v_i) / np.sum(ray_length_i)

    # Ray parameter of homogeneous case
    ray_p_ave = np.sin(takeoff_min) / v_ave

    # Check if the estimate works...
    if np.max(v_i) * ray_p_ave < 1.0:
        takeoff_max = np.arcsin( np.max(v_i) * ray_p_ave)
    else:
        takeoff_max = np.pi/2 - 0.1
    # Overshoot - Assume takeoff angle of the maximum layer
    
    ray_p_max = np.sin(takeoff_max) / src_vel

    # Initial Guess - EQ 11 of Kim and Baag (2002). 
    takeoff_0 = (takeoff_min + takeoff_max) / 2  
    ray_p_0 = np.sin(takeoff_0) / src_vel
    # X_c = prop_ray(ray_p_0, h_i, v_i)
    
    # # Improved estimate using dynamic Properties of the ray 
    # # See Figure 2 of Kim and Baag (2002)
    # X_R = epi_dist
    # delta_v = 2. * np.pi / 180  # 2 degree pertrubation
    # if X_c < X_R:
        
    # else:
    # ray_p_v = np.sin(takeoff_0 + delta_v) / src_vel
    # X_v = prop_ray(ray_p_v, h_i, v_i)

    # # Incidence angles of arriving center and "vicitinty" rays
    # i_c = np.arcsin(ray_p_0 * surf_vel)
    # i_v = np.arcsin(ray_p_v * surf_vel)
    # gamma_v = i_c - i_v

    # # "Horizontal" distance in center ray coordinates
    # q = (X_c - X_v) * np.cos(i_v) / np.cos(gamma_v)

    # # Radius of curvature of wavefront
    # R_A = q / np.tan(gamma_v)

    # # Distance to receiver from point A
    # D_fac = 2 * R_A * (X_c - X_R) * np.sin(i_c)
    # D = np.sqrt(R_A**2 + (X_c - X_R)**2 - D_fac)

    # # Distance between center and receiver ray
    # gamma_R = np.arcsin((X_c - X_R) * np.sin(np.pi/2 - gamma_v) / D)
    # n = R_A * np.tan(gamma_R)
    
    # # Radius of curvature of hypothetical "enlarged" wave
    # R_B = q / np.tan(delta_v)

    # # Approximate correction factor 
    # delta_R = np.arctan(n / R_B)
    # takeoff_init = takeoff_0 + delta_R
    # ray_p_init = np.sin(takeoff_init) / src_vel

    plt.figure()
    for z in layer_depths:
        plt.axhline(z)
    plt.plot(0, src_depth, 'o')
    plt.plot(epi_dist, 0, 'v', ms=10)
    plt.gca().invert_yaxis()

    # iterate of all layers
    X_list = []
    for ii, p in enumerate([ray_p_min, ray_p_max, ray_p_0]):
        deltas = np.empty_like(v_i)
        for jj, (h, v) in enumerate(zip(h_i, v_i)):
            dx, dt = get_const_layer_dxdt(p, h, v)
            deltas[jj] = dx
        X_list.append(np.sum(deltas))
        xp = np.cumsum(np.flip(np.append(deltas, 0)))
        yp = src_depth - np.cumsum(np.flip(np.append(h_i, 0)))
        plt.plot(xp, yp, '-o', label=ii)
    
    plt.legend()
    plt.show()


    # # Get bounds on sin theta of ray parameter 
    # max_vel_idx = np.argmax(direct_layer_vels)
    # max_vel = direct_layer_vels[max_vel_idx]
    # max_vel_thickness = direct_layer_thicks[max_vel_idx]
    # min_dist = np.sqrt(epi_dist**2 + max_vel_thickness**2)
    # # lower
    # sint_lower = (src_vel / max_vel) * (epi_dist / hypo_dist)
    # tant_lower = sint_lower / np.sqrt(1. - sint_lower**2)
    # above_sin_lower = (layer_vels[0:src_idx] / src_vel) * sint_lower
    # above_Tan_Lower = above_sin_lower / np.sqrt(1. - above_sin_lower**2)
    # deltaCalLower =  deplayerTop*lTan + sum(thickness(1:evtL-1).*aboveTanLower);
    # deltaDiffLower =  deltaCalLower-delta;
    # sint_upper = src_vel / max_vel * hypo_dist / min_dist




######
    # Check if in top layer for easy calc. 
    if src_idx == 0:
        # Trivial travel time and takeoff angle calc.
        travel_time = hypo_dist / layer_vels[src_idx]
        takeoff_angle = 180 - np.rad2deg(np.arcsin(epi_dist/hypo_dist))
        return travel_time, takeoff_angle
