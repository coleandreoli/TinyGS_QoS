import streamlit as st
import pandas as pd

from tinygs_qos.predictor import TransmissionPredictor

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="TinyGS QoS", page_icon="🛰️", layout="wide", initial_sidebar_state="auto"
)

st.title("TinyGS QoS Predictor")

# Valid (bw, sf) pairs based on data
VALID_PAIRS = {
    (125.0, 7),
    (62.5, 8),
    (125.0, 8),
    (125.0, 9),
    (500.0, 9),
    (125.0, 10),
    (250.0, 10),
    (125.0, 11),
}


@st.cache_resource
def load_predictor():
    """Load and cache the predictor instance."""
    return TransmissionPredictor()


def main():
    # Load predictor (cached)
    predictor = load_predictor()

    # Input section
    st.header("Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        sf = st.number_input(
            "Spreading Factor (SF)",
            min_value=7,
            max_value=11,
            value=10,
            step=1,
            help="LoRa spreading factor (7-11)",
        )
        sf = int(sf)

    with col2:
        bw = st.selectbox(
            "Select Bandwidth (kHz)",
            options=[62.5, 125.0, 250.0, 500.0],
            index=1,
            help="LoRa bandwidth in kHz",
        )

    col4, col5 = st.columns(2)
    with col4:
        gain = st.number_input(
            "Antenna Gain (dB)",
            min_value=-10.0,
            max_value=20.0,
            value=5.15,
            step=0.5,
            help="Minimum antenna gain in dB",
        )

    with col5:
        alt = st.number_input(
            "Satellite Altitude (km)",
            min_value=400.0,
            max_value=1100.0,
            value=500.0,
            step=50.0,
        )

    warnings = []
    errors = []

    # Check if (bw, sf) pair is valid
    if (bw, sf) not in VALID_PAIRS:
        warnings.append(
            f"Warning: (BW={bw}, SF={sf}) is not a commonly used combination. "
            f"Valid pairs are: {sorted(VALID_PAIRS)}"
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
                fig, X_grid = predictor.plot_transmission_probability(sf, bw, gain, alt)
                # Store in session state
                st.session_state.fig = fig
                st.session_state.X_grid = X_grid
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
