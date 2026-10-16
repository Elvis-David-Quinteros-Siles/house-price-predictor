import streamlit as st
import requests
import os
import socket  # For hostname and IP address
import folium
from streamlit_folium import st_folium

# Set the page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = os.getenv("API_URL", "http://model:8000").rstrip("/")

st.markdown(
    """
    <style>
    .prediction-value { font-size: 2.6rem; font-weight: 700; line-height: 1.2; margin: 0.25rem 0 1rem; }
    .info-label { margin-bottom: 0.1rem; opacity: 0.7; font-size: 0.9rem; }
    .info-value { font-size: 1.15rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ubicación inicial del marcador: Wallingford, Seattle.
DEFAULT_LOCATION = (47.6615, -122.3341)

# ── Textos de la interfaz (español / inglés)
LANGUAGES = {"es": "Español", "en": "English"}

TEXT = {
    "es": {
        "title": "Predicción del precio de una casa",
        "subtitle": (
            "Precio de venta estimado para una casa en King County, Washington (zona de Seattle). "
            "Entrenado con {rows:,} ventas reales de 2014&ndash;2015, así que los precios están en "
            "dólares de 2014&ndash;2015."
        ),
        "api_down": "La API de predicción no está disponible en este momento: {detail}",
        "location": "Ubicación",
        "location_help": "Haz clic en el mapa para colocar la casa.",
        "location_selected": "Coordenadas: {lat:.4f}, {long:.4f}",
        "sqft": "Superficie habitable (sq ft)",
        "sqft_lot": "Superficie del terreno (sq ft)",
        "bedrooms": "Dormitorios",
        "bathrooms": "Baños",
        "year_built": "Año de construcción",
        "condition": "Estado de conservación",
        "grade": "Calidad de construcción (grade)",
        "view": "Vista",
        "waterfront": "Frente al agua",
        "predict": "Calcular precio",
        "calculating": "Calculando la predicción...",
        "results": "Resultado",
        "error": "No se pudo obtener una predicción: {detail}",
        "location_too_far": (
            "Esa ubicación está a más de {km:g} km de cualquier casa vendida en King County. "
            "Elige un punto en tierra dentro del condado."
        ),
        "typical_error": "Error típico (test)",
        "model_used": "Modelo",
        "price_range": "Rango de precio ({coverage:.0%} de las casas de test)",
        "prediction_time": "Tiempo de predicción",
        "top_factors": "Factores que más pesan en el precio (modelo completo):",
        "placeholder": "Elige la ubicación en el mapa, completa el formulario y pulsa \"Calcular precio\".",
        "footer_built": "Creado para el MLOps Bootcamp",
        "footer_by": "de",
        "version": "Versión",
        "hostname": "Hostname",
        "ip": "Dirección IP",
        "factors": {
            "sqft": "Superficie habitable",
            "sqft_lot": "Superficie del terreno",
            "bedrooms": "Dormitorios",
            "bathrooms": "Baños",
            "year_built": "Año de construcción",
            "bed_bath_ratio": "Relación dormitorios/baños",
            "grade": "Calidad de construcción",
            "view": "Vista",
            "waterfront": "Frente al agua",
            "location": "Ubicación",
            "condition": "Estado de conservación",
        },
        "conditions": {"Poor": "Malo", "Fair": "Regular", "Average": "Normal", "Good": "Bueno", "Very Good": "Muy bueno"},
        "views": {0: "Sin vista", 1: "Regular", 2: "Normal", 3: "Buena", 4: "Excelente"},
        "grades": {
            3: "Por debajo del estándar",
            4: "Antigua, baja calidad",
            5: "Construcción económica",
            6: "Mínimo que cumple el código",
            7: "Promedio",
            8: "Algo superior al promedio",
            9: "Mejor diseño arquitectónico",
            10: "Alta calidad",
            11: "Diseño a medida, acabados superiores",
            12: "A medida, constructores excelentes",
            13: "Mansión de diseño exclusivo",
        },
    },
    "en": {
        "title": "House Price Prediction",
        "subtitle": (
            "Estimated sale price for a house in King County, Washington (Seattle area). "
            "Trained on {rows:,} real home sales from 2014&ndash;2015, so prices are in "
            "2014&ndash;2015 dollars."
        ),
        "api_down": "The prediction API is not available right now: {detail}",
        "location": "Location",
        "location_help": "Click on the map to place the house.",
        "location_selected": "Coordinates: {lat:.4f}, {long:.4f}",
        "sqft": "Living area (sq ft)",
        "sqft_lot": "Lot size (sq ft)",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "year_built": "Year built",
        "condition": "Condition",
        "grade": "Construction grade",
        "view": "View",
        "waterfront": "Waterfront",
        "predict": "Predict Price",
        "calculating": "Calculating prediction...",
        "results": "Prediction Results",
        "error": "Could not get a prediction: {detail}",
        "location_too_far": (
            "That location is more than {km:g} km from any house sold in King County. "
            "Pick a point on land inside the county."
        ),
        "typical_error": "Typical Error (test set)",
        "model_used": "Model Used",
        "price_range": "Price Range ({coverage:.0%} of test homes)",
        "prediction_time": "Prediction Time",
        "top_factors": "Top Factors Affecting Price (whole model):",
        "placeholder": "Pick the location on the map, fill out the form and click \"Predict Price\".",
        "footer_built": "Built for MLOps Bootcamp",
        "footer_by": "by",
        "version": "Version",
        "hostname": "Hostname",
        "ip": "IP Address",
        "factors": {
            "sqft": "Living Area",
            "sqft_lot": "Lot Size",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "year_built": "Year Built",
            "bed_bath_ratio": "Bedroom/Bathroom Ratio",
            "grade": "Construction Grade",
            "view": "View",
            "waterfront": "Waterfront",
            "location": "Location",
            "condition": "Condition",
        },
        "conditions": {"Poor": "Poor", "Fair": "Fair", "Average": "Average", "Good": "Good", "Very Good": "Very Good"},
        "views": {0: "None", 1: "Fair", 2: "Average", 3: "Good", 4: "Excellent"},
        "grades": {
            3: "Below building standards",
            4: "Older, low quality",
            5: "Low construction costs",
            6: "Lowest grade meeting code",
            7: "Average",
            8: "Just above average",
            9: "Better architectural design",
            10: "High quality",
            11: "Custom design, higher-quality finish",
            12: "Custom design, excellent builders",
            13: "Custom-designed mansion",
        },
    },
}

# Escala oficial de King County, de peor a mejor.
CONDITION_ORDER = ["Poor", "Fair", "Average", "Good", "Very Good"]

@st.cache_data(ttl=300, show_spinner=False)
def get_model_info():
    """Metrics of the served model and the values its inputs accept."""
    response = requests.get(f"{API_URL}/model-info", timeout=5)
    response.raise_for_status()
    return response.json()

def api_error_detail(error, t, max_distance_km):
    """Readable, translated message for a failed API call."""
    response = getattr(error, "response", None)
    if response is None:
        return str(error)
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text
    if isinstance(detail, dict):
        if detail.get("code") == "location_too_far":
            return t["location_too_far"].format(km=max_distance_km)
        return detail.get("message", str(detail))
    return str(detail)

# ── Idioma: selector arriba a la derecha; ?lang=en en la URL lo fija al entrar.
if "lang" not in st.session_state:
    st.session_state.lang = st.query_params.get("lang", "es") if st.query_params.get("lang") in LANGUAGES else "es"

_, lang_col = st.columns([5, 1])
with lang_col:
    lang = st.radio(
        "Idioma / Language",
        options=list(LANGUAGES),
        format_func=LANGUAGES.get,
        horizontal=True,
        label_visibility="collapsed",
        key="lang",
    )
st.query_params["lang"] = lang
t = TEXT[lang]

try:
    model_info = get_model_info()
except requests.exceptions.RequestException as e:
    st.title(t["title"])
    st.error(t["api_down"].format(detail=e))
    st.stop()

metrics = model_info["metrics"]
map_area = model_info["map_area"]

# Add title and description
st.title(t["title"])
st.markdown(
    f"""
    <p style="font-size: 18px; color: gray;">
        {t["subtitle"].format(rows=model_info["train_rows"] + model_info["test_rows"])}
    </p>
    """,
    unsafe_allow_html=True,
)

# Create a two-column layout
col1, col2 = st.columns(2, gap="large")

# Input form
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Location map: un clic mueve el marcador
    if "location" not in st.session_state:
        st.session_state.location = DEFAULT_LOCATION
    lat, long = st.session_state.location

    st.markdown(f"**{t['location']}** &middot; {t['location_help']}")
    house_map = folium.Map(location=[lat, long], zoom_start=11, tiles="OpenStreetMap")
    folium.Marker([lat, long]).add_to(house_map)
    map_state = st_folium(
        house_map,
        height=360,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="map",
    )
    clicked = (map_state or {}).get("last_clicked")
    if clicked:
        new_location = (round(clicked["lat"], 5), round(clicked["lng"], 5))
        if new_location != st.session_state.location:
            st.session_state.location = new_location
            st.session_state.pop("prediction", None)
            st.session_state.pop("prediction_error", None)
            st.rerun()
    st.caption(t["location_selected"].format(lat=lat, long=long))

    # Square Footage sliders
    size_col, lot_col = st.columns(2)
    with size_col:
        sqft = st.slider(t["sqft"], 400, 6000, 1900, 50, key="sqft")
    with lot_col:
        sqft_lot = st.number_input(t["sqft_lot"], min_value=500, max_value=200000, value=7600, step=100, key="sqft_lot")

    # Bedrooms and Bathrooms in two columns
    bed_col, bath_col = st.columns(2)
    with bed_col:
        bedrooms = st.selectbox(t["bedrooms"], options=list(range(1, 9)), index=2, key="bedrooms")

    with bath_col:
        bathroom_options = [x / 4 for x in range(3, 21)]  # 0.75 ... 5.0, como en King County
        bathrooms = st.selectbox(t["bathrooms"], options=bathroom_options, index=bathroom_options.index(2.0), key="bathrooms")

    # Construction grade and condition
    grade_col, cond_col = st.columns(2)
    with grade_col:
        grade = st.selectbox(
            t["grade"],
            options=list(t["grades"]),
            index=list(t["grades"]).index(7),
            format_func=lambda g: f"{g} · {t['grades'][g]}",
            key="grade",
        )
    with cond_col:
        conditions = [c for c in CONDITION_ORDER if c in model_info["categories"]["condition"]]
        condition = st.selectbox(
            t["condition"],
            options=conditions,
            index=conditions.index("Average"),
            format_func=t["conditions"].get,
            key="condition",
        )

    # View and waterfront
    view_col, water_col = st.columns(2)
    with view_col:
        view = st.selectbox(t["view"], options=list(t["views"]), format_func=t["views"].get, key="view")
    with water_col:
        st.write("")
        waterfront = st.checkbox(t["waterfront"], key="waterfront")

    # Year Built slider
    year_built = st.slider(t["year_built"], 1900, 2015, 1975, 1, key="year")

    # Predict button
    predict_button = st.button(t["predict"], use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Results section
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h2>{t['results']}</h2>", unsafe_allow_html=True)

    # If button is clicked, show prediction
    if predict_button:
        # Show loading spinner
        with st.spinner(t["calculating"]):
            # Prepare data for API call
            api_data = {
                "sqft": sqft,
                "sqft_lot": sqft_lot,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "year_built": year_built,
                "condition": condition,
                "grade": grade,
                "view": view,
                "waterfront": waterfront,
                "lat": lat,
                "long": long,
            }

            try:
                response = requests.post(f"{API_URL}/predict", json=api_data, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                st.session_state.prediction = response.json()
                st.session_state.pop("prediction_error", None)
            except requests.exceptions.RequestException as e:
                # Sin datos de ejemplo: si la API falla, no se muestra un precio.
                st.session_state.pop("prediction", None)
                st.session_state.prediction_error = e

    if "prediction_error" in st.session_state:
        detail = api_error_detail(st.session_state.prediction_error, t, map_area["max_distance_km"])
        st.error(t["error"].format(detail=detail))

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
            st.markdown(f'<p class="info-label">{t["typical_error"]}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="info-value">±{metrics["mape"]:.0%} &middot; ${metrics["mae"]:,.0f}</p>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="info-label">{t["model_used"]}</p>', unsafe_allow_html=True)
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
            st.markdown(f'<p class="info-label">{t["price_range"].format(coverage=coverage)}</p>', unsafe_allow_html=True)
            lower = "${:,.0f}".format(pred["confidence_interval"][0])
            upper = "${:,.0f}".format(pred["confidence_interval"][1])
            st.markdown(f'<p class="info-value">{lower} - {upper}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="info-label">{t["prediction_time"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="info-value">{pred["latency_ms"]:.0f} ms</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Top factors
        st.markdown('<div class="top-factors">', unsafe_allow_html=True)
        st.markdown(f"<p><strong>{t['top_factors']}</strong></p>", unsafe_allow_html=True)
        top_factors = list(pred.get("features_importance", {}).items())[:3]
        items = "".join(
            f"<li>{t['factors'].get(name, name)} ({weight:.0%})</li>"
            for name, weight in top_factors
        )
        st.markdown(f"<ul>{items}</ul>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif "prediction_error" not in st.session_state:
        # Display placeholder message
        st.markdown(f"""
        <div style="display: flex; height: 300px; align-items: center; justify-content: center; color: #6b7280; text-align: center;">
            {t["placeholder"]}
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
        <p><strong>{t["footer_built"]}</strong></p>
        <p>{t["footer_by"]} <a href="https://www.schoolofdevops.com" target="_blank">School of Devops</a></p>
        <p><strong>{t["version"]}:</strong> {version}</p>
        <p><strong>{t["hostname"]}:</strong> {hostname}</p>
        <p><strong>{t["ip"]}:</strong> {ip_address}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
