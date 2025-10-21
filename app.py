import os
import time
import hashlib
import re
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import altair as alt

# ==============================
# Constants & helpers
# ==============================
BER_ORDER = [
    "A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","E1","E2","F","G","Exempt","Unknown"
]

NUMERIC_COLS = [
    "Price (€)","Bedrooms_int","Bathrooms","lat","lon",
    "nearest_park_km","nearest_beach_km","nearest_gym_km","nearest_supermarket_km",
    "nearest_bus_stop_km","nearest_rail_station_km","nearest_tram_stop_km",
    "distance_to_city_centre_km","price_per_bedroom","energy_monthly_estimate",
    "effective_monthly_cost","min_transit_km",
    "pred_price","delta_pct","Fairness_Delta",
]

PRIMARY_COLS = [
    "Address","Property Type","Bedrooms","Bathrooms","BER Rating",
    "Price (€)","pred_price","Fairness_Delta","delta_pct",
    "effective_monthly_cost","price_per_bedroom",
    "distance_to_city_centre_km","min_transit_km","within_500m_transit",
    "energy_estimate_available","URL"
]

# ---- Value badge thresholds & confidence defaults ----
CV_RMSE_EUR = 822  # used for display only

B_FAIR  = 4
B_SLIGHT = 7
B_OVER  = 10
B_VERY  = 15

BADGE_LABELS = {
    "VERY_UNDER":   "Very underpriced",
    "UNDER":        "Underpriced",
    "SLIGHT_UNDER": "Slightly underpriced",
    "FAIR":         "Fair",
    "SLIGHT_OVER":  "Slightly overpriced",
    "OVER":         "Overpriced",
    "VERY_OVER":    "Very overpriced",
}
BADGE_ICONS = {
    "VERY_UNDER":   "🔥🔥",
    "UNDER":        "🔥",
    "SLIGHT_UNDER": "✅",
    "FAIR":         "🟢",
    "SLIGHT_OVER":  "🟠",
    "OVER":         "💰",
    "VERY_OVER":    "💰💰",
}
BADGE_ORDER = {
    "VERY_UNDER": 0,
    "UNDER": 1,
    "SLIGHT_UNDER": 2,
    "FAIR": 3,
    "SLIGHT_OVER": 4,
    "OVER": 5,
    "VERY_OVER": 6,
}

BED_OPTIONS  = list(range(1,10))
BATH_OPTIONS = list(range(1,10))

def file_digest(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:10]
    except Exception:
        return "n/a"

def normalise_ber(val: str) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "Unknown"
    s = str(val).strip()
    if s == "": return "Unknown"
    s_low = s.lower()
    if "exempt" in s_low: return "Exempt"
    s_compact = re.sub(r"\s+", "", s.upper()).replace("-", "")
    if s_compact in BER_ORDER: return s_compact
    m = re.match(r"^([A-G])(\d)$", s_compact)
    if m:
        guess = m.group(1) + m.group(2)
        if guess in BER_ORDER: return guess
    if s_low in {"n/a","na","none","unknown","-"}: return "Unknown"
    return "Unknown"

@st.cache_data(show_spinner=False)
def load_full_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # keep only rows with valid coordinates
    if {"lat","lon"}.issubset(df.columns):
        df = df.dropna(subset=["lat","lon"]).copy()
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df[df["lat"].between(-90,90) & df["lon"].between(-180,180)]

    # coerce numeric cols
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # engineered helpers
    if "Bedrooms_int" in df.columns:
        df["Bedrooms_int"] = pd.to_numeric(df["Bedrooms_int"], errors="coerce").astype("Int64")

    if "Bathrooms" in df.columns:
        df["Bathrooms_num"] = pd.to_numeric(df["Bathrooms"], errors="coerce")
        df["Bathrooms_int_approx"] = df["Bathrooms_num"].round().astype("Int64")

    if "Property Type" in df.columns:
        df["Property_Type"] = df["Property Type"]

    if "Price (€)" in df.columns:
        df["price_fmt"] = df["Price (€)"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")

    if "distance_to_city_centre_km" in df.columns:
        df["dist_centre"] = df["distance_to_city_centre_km"].round(2)

    # BER normalization
    df["BER_norm"] = df.get("BER Rating", "Unknown")
    df["BER_norm"] = df["BER_norm"].apply(normalise_ber)
    cat_order = pd.CategoricalDtype(categories=BER_ORDER, ordered=True)
    df["BER_norm"] = df["BER_norm"].astype(cat_order)

    # ensure delta_pct if possible
    if "delta_pct" not in df.columns and {"Price (€)", "pred_price"}.issubset(df.columns):
        df["delta_pct"] = np.where(
            (df["pred_price"] > 0) & pd.notna(df["pred_price"]),
            100.0 * (df["Price (€)"] - df["pred_price"]) / df["pred_price"],
            np.nan,
        )

    # confidence (display only)
    if "pred_price" in df.columns:
        df["conf_eur"] = CV_RMSE_EUR
        df["conf_pct"] = (100.0 * df["conf_eur"].where(df["pred_price"] > 0, np.nan) / df["pred_price"]).clip(0, 50)
    else:
        df["conf_eur"] = np.nan
        df["conf_pct"] = np.nan

    return df

def _badge_from_delta_only(delta_pct):
    if pd.isna(delta_pct):
        return "FAIR"
    a = abs(float(delta_pct))
    if a <= B_FAIR:
        return "FAIR"
    elif a <= B_SLIGHT:
        return "SLIGHT_UNDER" if delta_pct < 0 else "SLIGHT_OVER"
    elif a <= B_OVER:
        return "UNDER" if delta_pct < 0 else "OVER"
    elif a <= B_VERY:
        return "UNDER" if delta_pct < 0 else "OVER"
    else:
        return "VERY_UNDER" if delta_pct < 0 else "VERY_OVER"

# ==============================
# Load + top diagnostics
# ==============================
# Option A: fixed file (your current default)
# FULL_FILE = "cleaned_data_enriched_with_fairness.csv"

# Option B (recommended): selectable source in sidebar
DATA_FILES = {
    "Enriched + fairness": "cleaned_data_enriched_with_fairness.csv",
    "Enriched (no fairness)": "cleaned_data_enriched.csv",
}
with st.sidebar:
    st.divider()
    st.subheader("Data source")
    data_choice = st.radio("CSV file", list(DATA_FILES.keys()), index=0, key="csv_choice")
    if st.button("Reload data"):
        st.cache_data.clear()
        st.rerun()

FULL_FILE = DATA_FILES[st.session_state.get("csv_choice", "Enriched + fairness")]

try:
    df = load_full_dataset(FULL_FILE)
except Exception as e:
    st.error(f"Couldn't load {FULL_FILE}. Ensure it's in the repo root.\nDetails: {e}")
    st.stop()

# No images in this version. If any lingering column exists, drop it.
if "image_url" in df.columns:
    df = df.drop(columns=["image_url"])

# Value badge columns (deterministic from delta_pct)
if "delta_pct" in df.columns:
    df["value_code"]  = df["delta_pct"].apply(_badge_from_delta_only)
    df["value_label"] = df["value_code"].map(BADGE_LABELS)
    df["value_emoji"] = df["value_code"].map(BADGE_ICONS)
    df["value_badge"] = df["value_label"]
df["conf_eur_fmt"] = df["conf_eur"].apply(lambda x: f"{int(round(x)):,}" if pd.notna(x) else "—")

# Required core check
required_core = {"Address","Price (€)","lat","lon"}
missing = [c for c in required_core if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(FULL_FILE))) if os.path.exists(FULL_FILE) else 'n/a'

with st.expander("ℹ️ About this app"):
    st.markdown("""
**Daft.ie Rental Intelligence (MVP+)**

- **Goal:** Estimate *fair rent* and surface **value**.
- **Data:** Dublin rental listings (cleaned, geocoded, enriched with amenities).
- **Model:** RandomForestRegressor baseline.
- **Confidence:** Fair rent shown with a simple ± band (CV RMSE).
- **Value badge:** Based on percentage delta vs fair rent.
- **Limitations:** Missing floor area in many listings; geocoding accuracy; no causal claims.

*Built by Lee Gallagher* — feedback welcome.
""")

# ==============================
# Filter defaults + reset helper
# ==============================
def _set_default_filters(df):
    st.session_state["search_q"] = ""
    st.session_state["match_mode"] = "contains"
    st.session_state["quick_pick"] = []

    pmin = int(np.nanmin(df["Price (€)"]))
    pmax = int(np.nanmax(df["Price (€)"]))
    st.session_state["price_range"] = (pmin, pmax)

    st.session_state["sel_beds"]  = BED_OPTIONS[:]
    st.session_state["sel_baths"] = BATH_OPTIONS[:]

    st.session_state["sel_types"] = (
        sorted(df["Property Type"].dropna().unique().tolist())
        if "Property Type" in df.columns else []
    )

    st.session_state["include_exempt"] = True
    st.session_state["sel_ber"] = BER_ORDER[:]

    st.session_state["energy_only"] = False
    st.session_state["within_500m"] = False

    if "distance_to_city_centre_km" in df.columns:
        dmax = float(np.nanmax(df["distance_to_city_centre_km"]))
        st.session_state["dist_centre"] = float(np.ceil(dmax))
    else:
        st.session_state["dist_centre"] = None

    if "min_transit_km" in df.columns:
        tmax = float(np.nanmax(df["min_transit_km"]))
        st.session_state["max_transit"] = float(np.ceil(tmax))
    else:
        st.session_state["max_transit"] = None

if "filters_init" not in st.session_state:
    _set_default_filters(df)
    st.session_state["filters_init"] = True

# ==============================
# Sidebar: Filters
# ==============================
with st.sidebar:
    st.divider()
    st.header("Filters")

    search_q = st.text_input(
        "Search area (Address / Eircode)",
        placeholder="e.g., Raheny or D05 (comma-separate for multiple)",
        key="search_q"
    )
    match_mode = st.selectbox("Match mode", ["contains","starts with"], key="match_mode")

    areas = []
    if "Address" in df.columns:
        candidates = df["Address"].dropna().astype(str).map(lambda s: s.split(",")[-1].strip())
        areas = sorted(pd.Series(candidates).unique().tolist())
    quick_pick = st.multiselect("Quick-pick area", areas, key="quick_pick")

    with st.expander("Price & size", expanded=True):
        pmin = int(np.nanmin(df["Price (€)"]))
        pmax = int(np.nanmax(df["Price (€)"]))
        price_range = st.slider(
            "Price (€)", min_value=pmin, max_value=pmax,
            value=st.session_state["price_range"], step=50, key="price_range"
        )

        sel_beds = st.multiselect(
            "Bedrooms", BED_OPTIONS, default=st.session_state["sel_beds"], key="sel_beds"
        )
        sel_baths = st.multiselect(
            "Bathrooms (≈ rounded)", BATH_OPTIONS, default=st.session_state["sel_baths"], key="sel_baths"
        )

    with st.expander("Type & energy", expanded=False):
        types = sorted(df["Property Type"].dropna().unique()) if "Property Type" in df.columns else []
        sel_types = st.multiselect(
            "Property Type", options=types, default=st.session_state["sel_types"], key="sel_types"
        )

        exempt_count = int((df["BER_norm"].astype(str) == "Exempt").sum())
        st.caption(f"BER Exempt in data: {exempt_count} listings")
        include_exempt = st.checkbox("Include BER Exempt", value=st.session_state["include_exempt"], key="include_exempt")
        default_ber = BER_ORDER[:] if include_exempt else [b for b in BER_ORDER if b != "Exempt"]
        sel_ber = st.multiselect("BER Rating", options=BER_ORDER, default=default_ber, key="sel_ber")

        energy_only = st.checkbox("Energy estimate available", value=st.session_state["energy_only"], key="energy_only")

    with st.expander("Location & transit", expanded=False):
        if "distance_to_city_centre_km" in df.columns:
            dmax = float(np.nanmax(df["distance_to_city_centre_km"]))
            dist_centre = st.slider(
                "Max distance to city centre (km)",
                min_value=0.0, max_value=float(np.ceil(dmax)),
                value=st.session_state["dist_centre"], step=0.5, key="dist_centre"
            )
        else:
            dist_centre = None

        within_500m = st.checkbox(
            "Within 500m of any transit",
            value=st.session_state["within_500m"], key="within_500m"
        )

        if "min_transit_km" in df.columns:
            tmax = float(np.nanmax(df["min_transit_km"]))
            max_transit = st.slider(
                "Max distance to nearest transit (km)",
                min_value=0.0, max_value=float(np.ceil(tmax)),
                value=st.session_state["max_transit"], step=0.1, key="max_transit"
            )
        else:
            max_transit = None

    # Reset button + legend
    if st.button("Reset filters"):
        for k in [
            "search_q", "match_mode", "quick_pick",
            "price_range", "sel_beds", "sel_baths",
            "sel_types", "include_exempt", "sel_ber",
            "energy_only", "within_500m",
            "dist_centre", "max_transit",
        ]:
            st.session_state.pop(k, None)
        st.session_state.pop("filters_init", None)
        st.rerun()

    st.markdown("**Price bands (map colors)**")
    st.markdown("""
- 🟤 ≤ €1,499  
- 🟢 €1,500–2,000  
- 🔴 €2,001–2,500  
- 🔵 €2,501–3,000  
- 🟣 €3,001–4,000  
- ⚫ €4,001+
""")

# ==============================
# Apply filters
# ==============================
f = df.copy()
debug_counts = {"start": len(f)}

# Price
f = f[(f["Price (€)"] >= price_range[0]) & (f["Price (€)"] <= price_range[1])]
debug_counts["price"] = len(f)

# Bedrooms
if "Bedrooms_int" in f.columns and f["Bedrooms_int"].notna().any():
    f = f[f["Bedrooms_int"].astype("Int64").isin(sel_beds)]
else:
    if "Bedrooms" in f.columns:
        parsed = f["Bedrooms"].astype(str).str.extract(r"(\d+)").astype(float)[0].astype("Int64")
        f = f[parsed.isin(sel_beds)]
debug_counts["beds"] = len(f)

# Bathrooms (rounded)
if "Bathrooms_int_approx" in f.columns:
    f = f[f["Bathrooms_int_approx"].isin(sel_baths)]
elif "Bathrooms" in f.columns:
    # already engineered in load, but keep guard
    f["Bathrooms_num"] = pd.to_numeric(f["Bathrooms"], errors="coerce")
    f["Bathrooms_int_approx"] = f["Bathrooms_num"].round().astype("Int64")
    f = f[f["Bathrooms_int_approx"].isin(sel_baths)]
debug_counts["baths"] = len(f)

# Type
if "Property Type" in f.columns and len(sel_types):
    f = f[f["Property Type"].isin(sel_types)]
debug_counts["type"] = len(f)

# BER (normalized)
if len(sel_ber) and "BER_norm" in f.columns:
    f = f[f["BER_norm"].astype(str).isin(sel_ber)]
debug_counts["ber"] = len(f)

# Centre / transit / energy
if 'distance_to_city_centre_km' in f.columns and 'dist_centre' in locals() and dist_centre is not None:
    f = f[f["distance_to_city_centre_km"] <= dist_centre]
debug_counts["dist_centre"] = len(f)

if within_500m and "within_500m_transit" in f.columns:
    f = f[f["within_500m_transit"] == True]
debug_counts["within_500m"] = len(f)

if 'min_transit_km' in f.columns and 'max_transit' in locals() and max_transit is not None:
    f = f[f["min_transit_km"] <= max_transit]
debug_counts["transit"] = len(f)

if energy_only and "energy_estimate_available" in f.columns:
    f = f[f["energy_estimate_available"] == True]
debug_counts["energy_only"] = len(f)

# Search / eircode (robust contains)
if search_q and search_q.strip():
    terms = [t.strip().lower() for t in search_q.split(",") if t.strip()]
    def norm(x): 
        s = (x or "").lower()
        s = re.sub(r"\s+", " ", s)
        return s
    def any_term_in(s):
        s = norm(s)
        return any(term in s for term in terms)
    mask = None
    if "Address" in f.columns:
        mask = f["Address"].astype(str).apply(any_term_in)
    if "Eircode" in f.columns:
        m2 = f["Eircode"].astype(str).apply(any_term_in)
        mask = m2 if mask is None else (mask | m2)
    if mask is not None:
        f = f[mask]
debug_counts["search"] = len(f)

# Quick pick by trailing area
if len(quick_pick) and "Address" in f.columns:
    f = f[f["Address"].astype(str).str.strip().str.split(",").str[-1].str.strip().isin(quick_pick)]
debug_counts["quick_pick"] = len(f)

# ==============================
# KPIs
# ==============================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Listings", len(f))
with col2:
    st.metric("Avg price (€)", f["Price (€)"].mean().round(0) if len(f) else 0)
with col3:
    if "effective_monthly_cost" in f.columns and len(f):
        med = pd.to_numeric(f["effective_monthly_cost"], errors="coerce").median(skipna=True)
        st.metric("Median effective cost (€)", int(med) if pd.notna(med) else "—")
    else:
        st.metric("Median effective cost (€)", "—")
with col4:
    if "distance_to_city_centre_km" in f.columns:
        st.metric("Avg km to city centre", round(f["distance_to_city_centre_km"].mean(), 2) if len(f) else 0)
    else:
        st.metric("Avg km to city centre", "—")

st.divider()

# ==============================
# Map
# ==============================
def price_to_color(price):
    if price <= 1499: return [139, 69, 19]  # brown
    if price <= 2000: return [0, 255, 0]
    if price <= 2500: return [255, 0, 0]
    if price <= 3000: return [0, 0, 255]
    if price <= 4000: return [128, 0, 128]
    return [0, 0, 0]

if "Price (€)" in f.columns:
    f["color"] = f["Price (€)"].apply(price_to_color)

g = f.copy()
for col in ["lat","lon"]:
    if col in g.columns:
        g[col] = pd.to_numeric(g[col], errors="coerce")
g = g.dropna(subset=["lat","lon"])
g = g[g["lat"].between(-90,90) & g["lon"].between(-180,180)]

# Build a separate, formatted copy for tooltip only
gt = g.copy()
if "pred_price" in gt.columns:
    gt["pred_price"] = gt["pred_price"].apply(lambda x: f"{round(x):,}" if pd.notna(x) else "—")
if "delta_pct" in gt.columns:
    gt["delta_pct"] = pd.to_numeric(gt["delta_pct"], errors="coerce")
    gt["delta_pct"] = gt["delta_pct"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
if "conf_eur" in gt.columns:
    gt["conf_eur_fmt"] = gt["conf_eur"].apply(lambda x: f"{int(round(x)):,}" if pd.notna(x) else "—")
if "value_badge" in gt.columns and "value_emoji" in gt.columns:
    gt["value_label"] = gt["value_emoji"].fillna("") + " " + gt["value_badge"].fillna("")

DUBLIN_LAT, DUBLIN_LON, DUBLIN_ZOOM = 53.3498, -6.2603, 11.0
view = pdk.ViewState(latitude=DUBLIN_LAT, longitude=DUBLIN_LON, zoom=DUBLIN_ZOOM)

if len(gt) == 0:
    st.info("No mappable rows after filters. Showing Dublin.")
    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view))
else:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=gt,
        get_position='[lon, lat]',
        get_radius=50,
        get_fill_color='color' if "color" in gt.columns else [0, 122, 255],
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": """
            <div style='font-size:12px'>
              <b>{Address}</b><br/>
              Actual: €{price_fmt} • {Bedrooms} bed • {Property_Type}<br/>
              BER: {BER Rating} • {dist_centre} km to city centre<br/>
              <span style='display:inline-block;margin-top:4px;'>
                Fair rent: <b>€{pred_price}</b> &nbsp;± €{conf_eur_fmt}
                &nbsp;(<b>{delta_pct}</b>%)
              </span><br/>
              <span><b>Value:</b> {value_label}</span>
            </div>
        """,
        "style": {"backgroundColor": "#111", "color": "#fff"}
    }

    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view, layers=[layer], tooltip=tooltip))

st.divider()

# ==============================
# Charts
# ==============================
chart_cols = st.columns(2)

with chart_cols[0]:
    if "Price (€)" in f.columns and len(f):
        chart_df = f[["Price (€)","Property Type"]].dropna().rename(columns={"Price (€)": "PriceEUR"})
        price_hist = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("PriceEUR:Q", bin=alt.Bin(maxbins=30), title="Price (€)"),
            y=alt.Y("count()", title="Listings"),
            tooltip=["count()"]
        ).properties(height=300)
        st.altair_chart(price_hist, use_container_width=True)

with chart_cols[1]:
    if {"price_per_bedroom","Property Type"}.issubset(f.columns) and len(f):
        grp = f.groupby("Property Type", dropna=True)["price_per_bedroom"].mean().reset_index()
        grp = grp.sort_values("price_per_bedroom", ascending=False)
        bar = alt.Chart(grp).mark_bar().encode(
            x=alt.X("price_per_bedroom:Q", title="Avg € per bedroom"),
            y=alt.Y("Property Type:N", sort="-x"),
            tooltip=[alt.Tooltip("price_per_bedroom:Q", format=",.0f"), "Property Type"]
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)

st.divider()

# ==============================
# Explainability: Feature importances
# ==============================
st.subheader("What drives price (model view)")
imp_path = "feature_importances.csv"  # two cols: feature,importance
try:
    imps = pd.read_csv(imp_path)
    imps = imps.rename(columns={imps.columns[0]: "feature", imps.columns[1]: "importance"})
    imps = imps.sort_values("importance", ascending=True)

    n = len(imps)
    h = int(min(max(26 * n, 260), 600))

    chart = (
        alt.Chart(imps)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature", axis=alt.Axis(labelLimit=320, labelPadding=6)),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")]
        )
        .properties(height=h)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Random Forest feature importances (all {n} features).")
except Exception:
    st.caption("Feature importances file not found yet.")

st.divider()

# ==============================
# Matching listings (richer + selectable, no images)
# ==============================
# Base columns
show_cols = [c for c in PRIMARY_COLS if c in f.columns]

# Common value/confidence fields
for extra in ["value_emoji", "value_badge", "pred_price", "Fairness_Delta", "delta_pct", "conf_eur", "conf_pct"]:
    if extra in f.columns and extra not in show_cols:
        show_cols.append(extra)

# Optional extras (only include if present)
OPTIONAL_COLS = [
    "price_per_bedroom", "effective_monthly_cost",
    "distance_to_city_centre_km", "min_transit_km", "within_500m_transit",
    "nearest_park_km","nearest_beach_km","nearest_gym_km","nearest_supermarket_km",
    "nearest_bus_stop_km","nearest_rail_station_km","nearest_tram_stop_km",
    "nearest_garda_km","nearest_hospital_km",
    "BER_norm","Bedrooms_int","Bathrooms_int_approx"
]
optional_cols = [c for c in OPTIONAL_COLS if c in f.columns and c not in show_cols]

labels = {
    "URL": "Daft Listing",
    "Price (€)": "Price (€)",
    "pred_price": "Fair rent (€)",
    "Fairness_Delta": "Δ vs fair (€)",
    "delta_pct": "Δ vs fair (%)",
    "conf_eur": "± € (conf)",
    "conf_pct": "± % (conf)",
    "price_per_bedroom": "€ per bedroom",
    "effective_monthly_cost": "Effective € / mo",
    "distance_to_city_centre_km": "Km to centre",
    "min_transit_km": "Km to transit",
    "within_500m_transit": "≤500m transit?",
    "nearest_park_km": "Park_Dist (km)",
    "nearest_beach_km": "Beach_Dist (km)",
    "nearest_gym_km": "Gym_Dist (km)",
    "nearest_supermarket_km": "Supermarket_Dist (km)",
    "nearest_bus_stop_km": "BusStop_Dist (km)",
    "nearest_rail_station_km": "Train_Dist (km)",
    "nearest_tram_stop_km": "Luas_Dist (km)",
    "nearest_garda_km": "Garda_Dist (km)",
    "nearest_hospital_km": "Hospital_Dist (km)",
    "BER_norm": "BER",
    "Bedrooms_int": "Bedrooms (int)",
    "Bathrooms_int_approx": "Bathrooms (≈ rounded)",
    "value_badge": "Value",
    "value_emoji": " "
}

col_cfg = {}
if ("URL" in show_cols) or ("URL" in f.columns):
    col_cfg["URL"] = st.column_config.LinkColumn(label=labels["URL"], display_text="Open")
if "Price (€)" in show_cols:
    col_cfg["Price (€)"] = st.column_config.NumberColumn(label=labels["Price (€)"], format="%.0f")
if "pred_price" in show_cols:
    col_cfg["pred_price"] = st.column_config.NumberColumn(label=labels["pred_price"], format="%.0f")
if "Fairness_Delta" in show_cols:
    col_cfg["Fairness_Delta"] = st.column_config.NumberColumn(label=labels["Fairness_Delta"], format="%.0f")
if "delta_pct" in show_cols:
    col_cfg["delta_pct"] = st.column_config.NumberColumn(label=labels["delta_pct"], format="%.0f%%")
if "conf_eur" in show_cols:
    col_cfg["conf_eur"] = st.column_config.NumberColumn(label=labels["conf_eur"], format="%.0f")
if "conf_pct" in show_cols:
    col_cfg["conf_pct"] = st.column_config.NumberColumn(label=labels["conf_pct"], format="%.1f%%")
if "price_per_bedroom" in f.columns:
    col_cfg["price_per_bedroom"] = st.column_config.NumberColumn(label=labels["price_per_bedroom"], format="%.0f")
if "effective_monthly_cost" in f.columns:
    col_cfg["effective_monthly_cost"] = st.column_config.NumberColumn(label=labels["effective_monthly_cost"], format="%.0f")

for k in [
    "distance_to_city_centre_km","min_transit_km",
    "nearest_park_km","nearest_beach_km","nearest_gym_km","nearest_supermarket_km",
    "nearest_bus_stop_km","nearest_rail_station_km","nearest_tram_stop_km",
    "nearest_garda_km","nearest_hospital_km"
]:
    if k in f.columns:
        col_cfg[k] = st.column_config.NumberColumn(label=labels.get(k, k), format="%.2f")

if "within_500m_transit" in f.columns:
    col_cfg["within_500m_transit"] = st.column_config.CheckboxColumn(label=labels["within_500m_transit"])
if "BER_norm" in f.columns:
    col_cfg["BER_norm"] = st.column_config.TextColumn(label=labels["BER_norm"])
if "Bedrooms_int" in f.columns:
    col_cfg["Bedrooms_int"] = st.column_config.NumberColumn(label=labels["Bedrooms_int"], format="%.0f")
if "Bathrooms_int_approx" in f.columns:
    col_cfg["Bathrooms_int_approx"] = st.column_config.NumberColumn(label=labels["Bathrooms_int_approx"], format="%.0f")
if "value_badge" in f.columns:
    col_cfg["value_badge"] = st.column_config.TextColumn(label=labels["value_badge"])
if "value_emoji" in f.columns:
    col_cfg["value_emoji"] = st.column_config.TextColumn(label=labels["value_emoji"])

with st.expander("Columns to show", expanded=False):
    selected_extras = st.multiselect(
        "Extra columns",
        options=optional_cols,
        default=[c for c in ["value_emoji","value_badge","pred_price","Fairness_Delta","delta_pct",
                             "conf_eur","price_per_bedroom","effective_monthly_cost",
                             "distance_to_city_centre_km","min_transit_km","within_500m_transit",
                             "nearest_supermarket_km","nearest_bus_stop_km"]
                 if c in optional_cols],
        format_func=lambda c: labels.get(c, c)
    )

for c in selected_extras:
    if c not in show_cols:
        show_cols.append(c)

if len(f):
    st.subheader("Matching listings")

    # Key column: prefer URL if present anywhere
    key_col = "URL" if (("URL" in f.columns) or ("URL" in show_cols)) else "Address"

    # Sort: best value first (underpriced → fair → overpriced)
    if "value_code" in f.columns and "delta_pct" in f.columns:
        delta_num = pd.to_numeric(f["delta_pct"], errors="coerce")
        f = (
            f.assign(
                _value_rank=f["value_code"].map(BADGE_ORDER).astype(float).fillna(99),
                _delta_abs=delta_num.abs()
            )
            .sort_values(by=["_value_rank","_delta_abs"], ascending=[True, True])
            .drop(columns=["_value_rank","_delta_abs"])
        )

    display = f[show_cols].copy().reset_index(drop=True)

    if "favs" not in st.session_state:
        st.session_state["favs"] = set()
    display["Favourite"] = display[key_col].apply(lambda x: x in st.session_state["favs"])

    edited = st.data_editor(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg | {
            "Favourite": st.column_config.CheckboxColumn(label="⭐ Favourite", default=False)
        },
        key="main_table"
    )

    st.session_state["favs"] = set(edited.loc[edited["Favourite"] == True, key_col].tolist())

    if st.button("Show details for selected (first ⭐)"):
        sel_rows = edited.loc[edited["Favourite"] == True]
        if len(sel_rows):
            st.session_state["selected_key"] = sel_rows.iloc[0][key_col]
            st.rerun()

    with st.expander("⭐ Saved listings"):
        saved = edited.loc[edited["Favourite"] == True, show_cols + ([key_col] if key_col not in show_cols else [])]
        if len(saved):
            st.dataframe(saved, use_container_width=True, hide_index=True, column_config=col_cfg)
        else:
            st.caption("No favourites yet — tick ⭐ in the table above.")

    csv_bytes = f[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="filtered_listings.csv", mime="text/csv")
else:
    st.warning("No listings match your filters. Try widening your criteria.")

# ==============================
# Details panel & Debug
# ==============================
st.markdown("### Quick details")
st.caption("Tick ⭐ on a row and click the button above to show details here.")

sel_key = st.session_state.get("selected_key")
if sel_key:
    key_col = "URL" if (("URL" in f.columns) or ("URL" in show_cols)) else "Address"
    if key_col in f.columns and sel_key in set(f[key_col]):
        row = f[f[key_col] == sel_key].iloc[0]
        lines = []
        def add_line(label, col):
            if col in f.columns and pd.notna(row[col]):
                lines.append(f"**{label}:** {row[col]}")
        add_line("Address", "Address")
        add_line("Price (€)", "Price (€)")
        add_line("Fair rent (€)", "pred_price")
        add_line("Δ vs fair (%)", "delta_pct")
        add_line("Bedrooms", "Bedrooms")
        add_line("Bathrooms", "Bathrooms")
        add_line("BER", "BER_norm")
        add_line("Distance to centre (km)", "distance_to_city_centre_km")
        add_line("Nearest transit (km)", "min_transit_km")
        add_line("Park (km)", "nearest_park_km")
        add_line("Gym (km)", "nearest_gym_km")
        add_line("Supermarket (km)", "nearest_supermarket_km")
        if "URL" in f.columns:
            lines.append(f"[Open listing]({row['URL']})")
        st.markdown("<br>".join(lines), unsafe_allow_html=True)
    else:
        st.caption("Selection not in current filter results.")

with st.expander("Debug & Schema"):
    st.write("Data file:", FULL_FILE, "| md5:", file_digest(FULL_FILE))
    st.write("Columns present:", list(df.columns))
    st.write("Rows (original → filtered):", len(df), "→", len(f))
    st.write("Filter step counts:", debug_counts)
    st.write("BER normalized counts:", df["BER_norm"].astype(str).value_counts().to_dict())
    if "Bedrooms_int" in df.columns:
        st.write("Bedrooms unique:", sorted([int(x) for x in df["Bedrooms_int"].dropna().unique()]))
    if "Price (€)" in df.columns:
        st.write("Price min/max:", int(df["Price (€)"].min()), int(df["Price (€)"].max()))

st.divider()
st.markdown(
    """
    <div style='text-align:center; font-size:14px; opacity:0.9; margin-top:10px; line-height:1.6;'>
        Built by <b><a href='https://www.linkedin.com/in/lee-gallagher-7ba1721a3/' target='_blank' style='text-decoration:none; color:inherit;'>Lee Gallagher</a></b><br>
        <a href='https://www.linkedin.com/in/lee-gallagher-7ba1721a3/' target='_blank'>🔗 LinkedIn</a> |
        <a href='https://github.com/LeeGallagher42' target='_blank'>💻 GitHub</a><br>
        <span style='font-size:12px; color:#888;'>Data scraped, cleaned, and modelled to analyse Dublin rental fairness.</span>
    </div>
    """,
    unsafe_allow_html=True
)
