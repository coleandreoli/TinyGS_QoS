import numpy as np
import pandas as pd
import joblib


def base64_to_bits(b64_string: str) -> int:
    """Returns the number of bits represented by a Base64-encoded string."""
    # Remove padding
    stripped = b64_string.rstrip("=")
    num_chars = len(stripped)
    num_bits = num_chars * 6
    return num_bits


def StandardFilter(df, drop_alt=True):
    df = df[df["freq"] > 350]
    df = df[df["freq"] < 500]
    df = df[df["mode"] == "LoRa"]
    df["raw_size"] = df["raw"].apply(base64_to_bits)
    if drop_alt:
        df = df[df["satPosAlt"] > 400]
        df = df[~df["satellite"].str.startswith("Surveillance-")]
    return df


def numpy_to_features(arr, column_names=None):
    """
    Adapter function to convert numpy array, add features via FeatureColumns, and return numpy array.

    Parameters:
    -----------
    arr : numpy.ndarray
        2D array where columns represent features. Expected order:
        [lat, lon, alt, ...other features...]
    column_names : list, optional
        List of column names for the input array. If None, uses default names.
        First 3 columns must be lat, lon, alt in that order.

    Returns:
    --------
    numpy.ndarray
        2D array with original columns plus 'az', 'el', 'distance_to_station'
    """

    if column_names is None:
        # Default column names
        n_cols = arr.shape[1]
        column_names = ["satPosLat", "satPosLng", "satPosAlt"]
        if n_cols > 3:
            column_names.extend([f"feature_{i}" for i in range(3, n_cols)])
    else:
        # Ensure first 3 columns have the required names
        column_names = column_names.copy()
        column_names[0] = "satPosLat"
        column_names[1] = "satPosLng"
        column_names[2] = "satPosAlt"

    # Convert to DataFrame
    df = pd.DataFrame(arr, columns=column_names)

    return FeatureColumns(df)


def FeatureColumns(df):
    import pymap3d as pm
    from scipy.spatial import KDTree

    sf = pd.read_parquet("data/station_locations.parquet")

    # Build KDTree
    tree = joblib.load("data/kdtree_stations.joblib")

    # Pre-allocate output
    n_rows = len(df)
    az_result = np.empty(n_rows, dtype=np.float64)
    el_result = np.empty(n_rows, dtype=np.float64)
    dist_result = np.empty(n_rows, dtype=np.float64)

    # Process in batches
    batch_size = 10000
    for i in range(0, n_rows, batch_size):
        end_idx = min(i + batch_size, n_rows)
        batch_slice = slice(i, end_idx)

        sat_lats = df["satPosLat"].iloc[batch_slice].values
        sat_lons = df["satPosLng"].iloc[batch_slice].values
        sat_alts = df["satPosAlt"].iloc[batch_slice].values

        # Convert satellites to ECEF
        sat_x, sat_y, sat_z = pm.geodetic2ecef(sat_lats, sat_lons, sat_alts * 1000)
        sat_coords_ecef = np.column_stack([sat_x, sat_y, sat_z])

        # Find nearest station (MUCH faster than computing all distances)
        distances_ecef, min_idx = tree.query(sat_coords_ecef)

        # Get closest station coordinates
        closest_lats = sf["lat"].values[min_idx]
        closest_lons = sf["lon"].values[min_idx]

        # Compute az, el, distance only for closest stations
        az, el, distance = pm.geodetic2aer(
            sat_lats, sat_lons, sat_alts * 1000, closest_lats, closest_lons, 0
        )

        az_result[batch_slice] = az
        el_result[batch_slice] = el
        dist_result[batch_slice] = distance / 1000

    df["az"] = az_result
    df["el"] = el_result
    df["distance_to_station"] = dist_result
    return df


def TestSample(
    n_samples,
    rand_lat: bool = False,
    sf: list = None,
    bw: list = None,
    cr: list = None,
    gain: list = None,
    alt: float = 600.0,
):
    import joblib

    # Lat, Lon, Alt
    if rand_lat:
        lats = np.random.uniform(-89.999999999999, 89.999999999999, n_samples)
        lngs = np.random.uniform(-179.999999999999, 179.999999999999, n_samples)
    else:
        # Check if perfect square and adjust if needed
        grid_size = int(np.sqrt(n_samples))
        if grid_size * grid_size != n_samples:
            # Adjust to nearest perfect square
            n_samples = grid_size * grid_size
            print(f"Adjusted n_samples to {n_samples} ({grid_size}x{grid_size} grid)")

        lats_1d = np.linspace(-89.999999999999, 89.999999999999, grid_size)
        lngs_1d = np.linspace(-179.999999999999, 179.999999999999, grid_size)
        lats_grid, lngs_grid = np.meshgrid(lats_1d, lngs_1d)
        lats = lats_grid.flatten()
        lngs = lngs_grid.flatten()

    model_alt = joblib.load("data/kde_satPosAlt.joblib")
    alt_samples = np.full(n_samples, alt)
    alt_samples = model_alt.sample(n_samples)[:, 0]

    # Define approved (bw, sf) pairs
    if bw is not None and sf is not None:
        # User provided specific pairs - create all combinations
        approved_pairs = [(b, s) for b in bw for s in sf]
        # Uniform probabilities when user specifies custom pairs
        pair_probabilities = None
    else:
        # Default approved pairs based on actual data distribution
        approved_pairs = [
            (125.0, 7),
            (62.5, 8),
            (125.0, 8),
            (125.0, 9),
            (500.0, 9),
            (125.0, 10),
            (250.0, 10),
            (125.0, 11),
        ]
        # Probabilities based on actual data frequencies
        pair_probabilities = np.array(
            [
                0.001554,  # (125.0, 7)
                0.100817,  # (62.5, 8)
                0.002164,  # (125.0, 8)
                0.012203,  # (125.0, 9)
                0.349668,  # (500.0, 9)
                0.500641,  # (125.0, 10)
                0.020523,  # (250.0, 10)
                0.012430,  # (125.0, 11)
            ]
        )

    # Sample from approved pairs (vectorized)
    approved_pairs_array = np.array(approved_pairs)
    sampled_indices = np.random.choice(
        len(approved_pairs), size=n_samples, p=pair_probabilities
    )
    bw_samples = approved_pairs_array[sampled_indices, 0]
    sf_samples = approved_pairs_array[sampled_indices, 1]

    # cr_categories = np.array([5, 6, 8])
    # cr_probabilities = np.array([0.87976141, 0.10780799, 0.0124306])
    # cr_samples = np.random.choice(cr_categories, size=n_samples, p=cr_probabilities)
    cr_samples = np.full(n_samples, cr)

    X_random = np.vstack([lats, lngs, alt_samples, sf_samples, bw_samples, cr_samples])

    X_random = numpy_to_features(
        X_random.T,
        column_names=["satPosLat", "satPosLng", "satPosAlt", "sf", "bw", "cr"],
    )

    # Load min_gain data

    if gain:
        X_random["min_gain"] = np.random.choice(gain, size=n_samples)
    else:
        min_gain = pd.read_parquet("data/satellite_min_gain.parquet")
        grouped = min_gain.groupby(["sf", "bw"])["min_gain"].apply(np.array).to_dict()

        # Vectorized assignment using pandas map + numpy random choice
        def fast_assign(X_random, grouped):
            # Get group arrays for each row
            keys = pd.Series(list(zip(X_random["sf"], X_random["bw"])))
            arrays = keys.map(grouped)

            # Random choice from each array
            return np.array(
                [np.random.choice(arr) if arr is not None else np.nan for arr in arrays]
            )

        X_random["min_gain"] = fast_assign(X_random, grouped)

    return X_random


def SSOOrbitTestSample(
    altitude: float,
    sf: int,
    bw: float,
    gain: float,
    raan: float = 0.0,
    time_step: float = 1.0,
):
    """
    Generate test samples for a complete Sun-Synchronous Orbit.

    Parameters:
    -----------
    altitude : float
        Orbital altitude in km
    sf : int
        LoRa spreading factor
    bw : float
        LoRa bandwidth in kHz
    gain : float
        Antenna gain in dB
    raan : float, optional
        Right Ascension of Ascending Node in degrees (default: 0.0)
    time_step : float, optional
        Time step in seconds (default: 1.0)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with orbital positions and LoRa parameters including
        az, el, distance_to_station features
    """
    # Earth parameters
    R_earth = 6371.0  # km
    mu = 398600.4418  # km^3/s^2 (Earth's gravitational parameter)
    omega_earth = 7.2921159e-5  # rad/s (Earth's rotation rate)
    J2 = 0.00108263  # Earth's oblateness coefficient

    # Orbital parameters
    r = R_earth + altitude  # orbital radius

    # Calculate orbital period using Kepler's third law
    orbital_period = 2 * np.pi * np.sqrt(r**3 / mu)  # seconds

    # Calculate SSO inclination using the standard formula
    # i = arccos(-dot_Omega * 2 * a^(7/2) / (3 * J2 * R_earth^2 * sqrt(mu)))
    # where dot_Omega = 0.9856 deg/day for sun-synchronous
    mean_motion = np.sqrt(mu / r**3)  # rad/s

    # SSO nodal precession rate (0.9856 deg/day)
    dot_omega_sso = 0.9856 * (np.pi / 180) / 86400  # rad/s

    # Calculate cos(i) for SSO
    cos_i = -dot_omega_sso * 2 * (r**3.5) / (3 * J2 * (R_earth**2) * np.sqrt(mu))

    # Clamp to valid range
    cos_i = np.clip(cos_i, -1.0, 1.0)
    inclination = np.arccos(cos_i)

    raan_rad = np.radians(raan)
    arg_perigee = 0  # Circular orbit

    # Generate time points for one complete orbit
    n_samples = int(np.ceil(orbital_period / time_step))
    time_array = np.arange(0, n_samples) * time_step

    # Calculate mean motion
    n = np.sqrt(mu / r**3)  # rad/s

    # Initialize arrays
    lats = []
    lons = []
    alts = []

    for t in time_array:
        # Mean anomaly (for circular orbit, true anomaly ≈ mean anomaly)
        # Negative sign to make longitude increase (eastward ground track)
        M = -n * t

        # For circular orbit, true anomaly = mean anomaly
        nu = M  # true anomaly

        # Position in orbital plane (perifocal coordinates)
        x_peri = r * np.cos(nu)
        y_peri = r * np.sin(nu)
        z_peri = 0

        # Rotation matrices for orbital elements
        # 1. Rotate by argument of perigee
        cos_omega = np.cos(arg_perigee)
        sin_omega = np.sin(arg_perigee)

        # 2. Rotate by inclination
        cos_i = np.cos(inclination)
        sin_i = np.sin(inclination)

        # 3. Rotate by RAAN
        cos_raan = np.cos(raan_rad)
        sin_raan = np.sin(raan_rad)

        # Transform to Earth-Centered Inertial (ECI) coordinates
        x_eci = (cos_raan * cos_omega - sin_raan * sin_omega * cos_i) * x_peri + (
            -cos_raan * sin_omega - sin_raan * cos_omega * cos_i
        ) * y_peri

        y_eci = (sin_raan * cos_omega + cos_raan * sin_omega * cos_i) * x_peri + (
            -sin_raan * sin_omega + cos_raan * cos_omega * cos_i
        ) * y_peri

        z_eci = (sin_omega * sin_i) * x_peri + (cos_omega * sin_i) * y_peri

        # Account for Earth's rotation (convert ECI to ECEF)
        theta = omega_earth * t  # Earth rotation angle
        x_ecef = np.cos(theta) * x_eci + np.sin(theta) * y_eci
        y_ecef = -np.sin(theta) * x_eci + np.cos(theta) * y_eci
        z_ecef = z_eci

        # Convert ECEF to lat/lon/alt
        lon = np.arctan2(y_ecef, x_ecef)
        lat = np.arctan2(z_ecef, np.sqrt(x_ecef**2 + y_ecef**2))
        alt = np.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2) - R_earth

        # Convert to degrees and append
        lats.append(np.degrees(lat))
        lons.append(np.degrees(lon))
        alts.append(alt)

    # Create arrays for LoRa parameters
    sf_samples = np.full(n_samples, sf)
    bw_samples = np.full(n_samples, bw)

    # Use coding rate distribution from TestSample
    cr_categories = np.array([5, 6, 8])
    cr_probabilities = np.array([0.87976141, 0.10780799, 0.0124306])
    cr_samples = np.random.choice(cr_categories, size=n_samples, p=cr_probabilities)

    gain_samples = np.full(n_samples, gain)

    # Create feature array
    X_orbit = np.column_stack([lats, lons, alts, sf_samples, bw_samples, cr_samples])

    # Add features using existing function
    X_orbit_df = numpy_to_features(
        X_orbit,
        column_names=["satPosLat", "satPosLng", "satPosAlt", "sf", "bw", "cr"],
    )

    X_orbit_df["min_gain"] = gain_samples

    return X_orbit_df


def pu_f1_modified(y_true, y_pred):
    from sklearn.metrics import recall_score

    """
    Modified F1 score for PU learning
    F1_mod = recall^2 / Pr(y_hat = 1)

    Parameters:
    -----------
    y_true : array-like
        True labels (1 = positive/labeled, 0 = unlabeled)
    y_pred : array-like
        Predicted labels (1 = positive, 0 = negative)

    Returns:
    --------
    float : Modified F1 score
    """
    # Calculate recall on the labeled positive examples
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Calculate Pr(y_hat = 1) - proportion of predictions that are positive
    pr_y_hat_1 = y_pred.sum() / len(y_pred)

    # Avoid division by zero
    if pr_y_hat_1 == 0:
        return 0.0

    # Calculate modified F1
    f1_mod = (recall**2) / pr_y_hat_1

    return f1_mod


def fspl(d_km, f_mhz):
    d_m = d_km * 1000
    f_hz = f_mhz * 1e6
    c = 3e8
    return 20 * np.log10(4 * np.pi * d_m * f_hz / c)
