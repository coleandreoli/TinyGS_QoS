import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import make_scorer


from utils.tiny_utils import pu_f1_modified
from utils.tiny_utils import TestSample

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="TinyGS QoS", page_icon="🛰️", layout="wide", initial_sidebar_state="auto"
)

pu_f1_scorer = make_scorer(pu_f1_modified)
st.title("TinyGS QoS Predictor")


class TransmissionPredictor:
    """Class for predicting satellite transmission probabilities."""

    def __init__(
        self,
        model_path="data/PU_optuna_SDG_log_loss_12_7_v2.joblib",
        data_path="data/packet_features.parquet",
    ):
        """
        Initialize the predictor by loading the model and data.

        Parameters:
        -----------
        model_path : str
            Path to the trained model joblib file
        data_path : str
            Path to the packet features parquet file
        """
        self.pipeline = joblib.load(model_path)
        self.packet_data = pd.read_parquet(data_path)

        self.X_grid = None

    def predict(
        self,
        sat_alt: float,
        sf: int,
        bw: float,
        cr: int,
        min_gain: float,
        el: float,
        distance_to_station: float,
    ) -> float:
        """
        Predict transmission probability for a single satellite position and configuration.

        Parameters:
        -----------
        sat_alt : float
            Satellite altitude in km
        sf : int
            LoRa spreading factor (7-12)
        bw : float
            LoRa bandwidth in kHz (e.g., 62.5, 125.0, 250.0, 500.0)
        cr : int
            LoRa coding rate (5, 6, or 8)
        min_gain : float
            Minimum antenna gain in dB
        el : float
            Elevation angle in degrees
        distance_to_station : float
            Distance to nearest station in km

        Returns:
        --------
        float
            Predicted probability of transmission (0-1)
        """
        features = np.array([[sat_alt, sf, bw, cr, el, distance_to_station, min_gain]])

        # Return probability of positive class
        return self.pipeline.predict_proba(features)[0, 1]

    def predict_batch(self, df):
        """
        Predict transmission probabilities for a batch of rows.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing features

        Returns:
        --------
        np.ndarray
            Array of predicted probabilities
        """
        # Define the expected column order for the model
        expected_cols = [
            "satPosAlt",
            "sf",
            "bw",
            "cr",
            "el",
            "distance_to_station",
            "min_gain",
        ]

        # Reorder columns to match model expectations
        X = df[expected_cols]

        return self.pipeline.predict_proba(X.values)[:, 1]

    def score(self, X, y):
        from utils.tiny_utils import pu_f1_modified
        from sklearn.metrics import recall_score

        y_pred = self.predict_batch(X)
        recall = recall_score(y, y_pred, zero_division=0)
        f1_mod = pu_f1_modified(y, y_pred)
        return {"recall": recall, "f1_mod": f1_mod}

    def plot_transmission_probability(
        self, sf: int, bw: float, cr: int, gain: float, alt: float
    ) -> plt.Figure:
        """
        Generate a transmission probability heatmap.

        Parameters:
        -----------
        sf : int
            Spreading factor
        bw : float
            Bandwidth in kHz
        cr: int
            Coding rate
        gain : float
            Antenna gain in dB
        alt : float
            Satellite altitude in km

        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the probability heatmap
        """
        # Generate grid samples
        X_grid = TestSample(
            10000, sf=[sf], bw=[bw], cr=[cr], gain=[gain], alt=alt, rand_lat=False
        )

        # Predict probabilities for the grid
        X_grid["probability"] = self.predict_batch(X_grid)
        self.X_grid = X_grid

        # Create plot
        test_size = int(np.sqrt(len(X_grid)))
        fig, ax = plt.subplots(figsize=(14, 7))
        plt.rcParams.update({"font.size": 14})

        im = ax.contourf(
            X_grid["satPosLng"].values.reshape(test_size, test_size),
            X_grid["satPosLat"].values.reshape(test_size, test_size),
            X_grid["probability"].values.reshape(test_size, test_size),
            levels=np.linspace(0, 1, 20),
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
        )
        cbar = plt.colorbar(im, ax=ax, label="Probability of Transmission")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_title(
            f"Transmission Probability: SF={sf}, BW={bw}, Gain={gain}, Alt={alt} km",
            fontsize=14,
        )

        # Overlay actual transmission data
        tf = self.packet_data[
            (self.packet_data["sf"] == sf) & (self.packet_data["bw"] == bw)
        ]

        if len(tf) > 20000:
            tf = tf.sample(20000)
        ax.scatter(
            tf["satPosLng"],
            tf["satPosLat"],
            alpha=0.2,
            s=1,
            color="black",
            label="True Transmissions",
        )
        ax.legend()
        plt.tight_layout()
        return fig


@st.cache_resource
def load_predictor():
    """Load and cache the predictor instance."""
    return TransmissionPredictor()


def main():
    # Load predictor (cached)
    predictor = load_predictor()

    # Define valid (bw, sf) pairs based on your data
    valid_pairs = {
        (125.0, 7),
        (62.5, 8),
        (125.0, 8),
        (125.0, 9),
        (500.0, 9),
        (125.0, 10),
        (250.0, 10),
        (125.0, 11),
    }

    # Input section
    st.header("Input Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        sf = st.number_input(
            "Spreading Factor (SF)",
            min_value=7,
            max_value=11,
            value=10,
            step=1,
            help="LoRa spreading factor (7-11)",
        )

    with col2:
        bw = st.selectbox(
            "Select Bandwidth (kHz)",
            options=[62.5, 125.0, 250.0, 500.0],
            index=1,
            help="LoRa bandwidth in kHz",
        )
    with col3:
        cr = st.selectbox(
            "Select Coding Rate",
            options=[5, 6, 8],
            index=0,
            help="LoRa coding rate",
        )

    col4, col5 = st.columns(2)
    with col4:
        gain = st.number_input(
            "Antenna Gain (dB)",
            min_value=-10.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Minimum antenna gain in dB",
        )

    with col5:
        alt = st.number_input(
            "Satellite Altitude (km)",
            min_value=400.0,
            max_value=1100.0,
            value=600.0,
            step=50.0,
        )

    warnings = []
    errors = []

    # Check if (bw, sf) pair is valid
    if (bw, int(sf)) not in valid_pairs:
        warnings.append(
            f"Warning: (BW={bw}, SF={int(sf)}) is not a commonly used combination. "
            f"Valid pairs are: {sorted(valid_pairs)}"
        )

    # Display warnings and errors
    for warning in warnings:
        st.warning(warning)

    for error in errors:
        st.error(error)

    # Initialize session state for storing figure and data
    if "fig" not in st.session_state:
        st.session_state.fig = None
    if "X_grid" not in st.session_state:
        st.session_state.X_grid = None

    # Generate plot
    if st.button("Generate Coverage Map", disabled=len(errors) > 0):
        with st.spinner("Generating predictions..."):
            try:
                fig = predictor.plot_transmission_probability(
                    int(sf), bw, cr, gain, alt
                )
                # Store in session state
                st.session_state.fig = fig
                st.session_state.X_grid = predictor.X_grid
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.exception(e)

    if st.session_state.X_grid is not None:
        csv = st.session_state.X_grid.to_csv(index=False).encode("utf-8")
    else:
        csv = pd.DataFrame().to_csv(index=False).encode("utf-8")
        
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="transmission_predictions.csv",
        mime="text/csv",
        disabled=st.session_state.X_grid is None,
    )
    # Display the map if it exists in session state
    if st.session_state.fig is not None:
        st.pyplot(st.session_state.fig)


if __name__ == "__main__":
    main()
