import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tinygs_qos.utils.tiny_utils import TestSample


class TransmissionPredictor:
    """Class for predicting satellite transmission probabilities."""

    def __init__(
        self,
        model_path="data/PU_optuna_SDG_log_loss_v4.joblib",
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

    def predict(
        self,
        sat_alt: float,
        sf: int,
        bw: float,
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
        features = np.array([[sat_alt, sf, bw, el, distance_to_station, min_gain]])

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
            "el",
            "distance_to_station",
            "min_gain",
        ]

        # Reorder columns to match model expectations
        X = df[expected_cols]

        return self.pipeline.predict_proba(X.values)[:, 1]

    def score(self, X, y):
        from tinygs_qos.utils.tiny_utils import pu_f1_modified
        from sklearn.metrics import recall_score

        y_pred = self.predict_batch(X)
        recall = recall_score(y, y_pred, zero_division=0)
        f1_mod = pu_f1_modified(y, y_pred)
        return {"recall": recall, "f1_mod": f1_mod}

    def plot_transmission_probability(
        self, sf: int, bw: float, gain: float, alt: float
    ):
        """
        Generate a transmission probability heatmap.

        Parameters:
        -----------
        sf : int
            Spreading factor
        bw : float
            Bandwidth in kHz
        gain : float
            Antenna gain in dB
        alt : float
            Satellite altitude in km

        Returns:
        --------
        tuple
            (matplotlib.figure.Figure, pd.DataFrame) - Figure and X_grid data
        """
        # Generate grid samples
        X_grid = TestSample(
            10000, sf=[sf], bw=[bw], gain=[gain], alt=alt, rand_lat=False
        )
        # Predict probabilities for the grid
        X_grid["probability"] = self.predict_batch(X_grid)

        # Create plot
        test_size = int(np.sqrt(len(X_grid)))
        fig, ax = plt.subplots(figsize=(14, 7))

        im = ax.contourf(
            X_grid["satPosLng"].values.reshape(test_size, test_size),
            X_grid["satPosLat"].values.reshape(test_size, test_size),
            X_grid["probability"].values.reshape(test_size, test_size),
            levels=np.linspace(0, 1, 20),
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
        )
        cbar = fig.colorbar(im, ax=ax, label="Probability of Transmission")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_title(
            f"Transmission Probability: SF={sf}, BW={bw:.2f}, Gain={gain:.2f}, Alt={alt:.2f} km",
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
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig, X_grid
