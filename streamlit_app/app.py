import streamlit as st
import requests
import os
import socket  # For hostname and IP address

# Set the page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = os.getenv("API_URL", "http://model:8000").rstrip("/")

# Escala oficial de King County, de peor a mejor.
CONDITION_ORDER = ["Poor", "Fair", "Average", "Good", "Very Good"]

FACTOR_LABELS = {
    "sqft": "Square Footage",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "year_built": "Year Built",
    "bed_bath_ratio": "Bedroom/Bathroom Ratio",
    "location": "Location (zipcode)",
    "condition": "Condition",
}

@st.cache_data(ttl=300, show_spinner=False)
def get_model_info():
    """Metrics of the served model and the values its inputs accept."""
    response = requests.get(f"{API_URL}/model-info", timeout=5)
    response.raise_for_status()
    return response.json()

def api_error_detail(error):
    """Readable message for a failed API call (validation detail if there is one)."""
    response = getattr(error, "response", None)
    if response is not None:
        try:
            return response.json().get("detail", response.text)
        except ValueError:
            return response.text
    return str(error)

try:
    model_info = get_model_info()
except requests.exceptions.RequestException as e:
    st.title("House Price Prediction")
    st.error(f"The prediction API is not available right now: {api_error_detail(e)}")
    st.stop()

metrics = model_info["metrics"]

# Add title and description
st.title("House Price Prediction")
st.markdown(
    f"""
    <p style="font-size: 18px; color: gray;">
        Estimated sale price for a house in King County, Washington (Seattle area).
        Trained on {model_info["train_rows"] + model_info["test_rows"]:,} real home sales from 2014&ndash;2015,
        so prices are in 2014&ndash;2015 dollars.
    </p>
    """,
    unsafe_allow_html=True,
)

# Create a two-column layout
col1, col2 = st.columns(2, gap="large")

# Input form
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Square Footage slider
    sqft = st.slider("Square Footage (living area, sq ft)", 400, 6000, 1900, 50, key="sqft")

    # Bedrooms and Bathrooms in two columns
    bed_col, bath_col = st.columns(2)
    with bed_col:
        bedrooms = st.selectbox("Bedrooms", options=list(range(1, 9)), index=2)

    with bath_col:
        bathroom_options = [x / 4 for x in range(3, 21)]  # 0.75 ... 5.0, como en King County
        bathrooms = st.selectbox("Bathrooms", options=bathroom_options, index=bathroom_options.index(2.0))

    # Location dropdown (zipcodes seen in training)
    zipcodes = model_info["categories"]["location"]
    location = st.selectbox(
        "Location (zipcode)",
        options=zipcodes,
        index=zipcodes.index("98103") if "98103" in zipcodes else 0,
    )

    # Condition dropdown
    conditions = [c for c in CONDITION_ORDER if c in model_info["categories"]["condition"]]
    condition = st.selectbox("Condition", options=conditions, index=conditions.index("Average"))

    # Year Built slider
    year_built = st.slider("Year Built", 1900, 2015, 1975, 1, key="year")

    # Predict button
    predict_button = st.button("Predict Price", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Results section
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)

    # If button is clicked, show prediction
    if predict_button:
        # Show loading spinner
        with st.spinner("Calculating prediction..."):
            # Prepare data for API call
            api_data = {
                "sqft": sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "location": location,
                "year_built": year_built,
                "condition": condition
            }

            try:
                response = requests.post(f"{API_URL}/predict", json=api_data, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                st.session_state.prediction = response.json()
            except requests.exceptions.RequestException as e:
                # Sin datos de ejemplo: si la API falla, no se muestra un precio.
                st.session_state.pop("prediction", None)
                st.error(f"Could not get a prediction: {api_error_detail(e)}")

    # Display prediction if available
    if "prediction" in st.session_state:
        pred = st.session_state.prediction

        # Format the predicted price
        formatted_price = "${:,.0f}".format(pred["predicted_price"])
        st.markdown(f'<div class="prediction-value">{formatted_price}</div>', unsafe_allow_html=True)

        # Display test-set accuracy and model used
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Typical Error (test set)</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="info-value">±{metrics["mape"]:.0%} &middot; ${metrics["mae"]:,.0f}</p>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Model Used</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="info-value">{model_info["algorithm"]} &middot; R² {metrics["r2"]:.2f}</p>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Display price range and prediction time
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            coverage = model_info["interval"]["coverage"]
            st.markdown(f'<p class="info-label">Price Range ({coverage:.0%} of test homes)</p>', unsafe_allow_html=True)
            lower = "${:,.0f}".format(pred["confidence_interval"][0])
            upper = "${:,.0f}".format(pred["confidence_interval"][1])
            st.markdown(f'<p class="info-value">{lower} - {upper}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Prediction Time</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="info-value">{pred["latency_ms"]:.0f} ms</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Top factors
        st.markdown('<div class="top-factors">', unsafe_allow_html=True)
        st.markdown("<p><strong>Top Factors Affecting Price (whole model):</strong></p>", unsafe_allow_html=True)
        top_factors = list(pred.get("features_importance", {}).items())[:3]
        items = "".join(
            f"<li>{FACTOR_LABELS.get(name, name)} ({weight:.0%})</li>"
            for name, weight in top_factors
        )
        st.markdown(f"<ul>{items}</ul>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display placeholder message
        st.markdown("""
        <div style="display: flex; height: 300px; align-items: center; justify-content: center; color: #6b7280; text-align: center;">
            Fill out the form and click "Predict Price" to see the estimated house price.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Fetch version, hostname, and IP address
version = os.getenv("APP_VERSION", "1.0.0")  # Default version if not set in environment
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
st.markdown(
    f"""
    <div style="text-align: center; color: gray; margin-top: 20px;">
        <p><strong>Built for MLOps Bootcamp</strong></p>
        <p>by <a href="https://www.schoolofdevops.com" target="_blank">School of Devops</a></p>
        <p><strong>Version:</strong> {version}</p>
        <p><strong>Hostname:</strong> {hostname}</p>
        <p><strong>IP Address:</strong> {ip_address}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
