import os
import time
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import altair as alt

# ------------------------------
# Constants & helpers
# ------------------------------
BER_ORDER = [
    "A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","E1","E2","F","G","Exempt","Unknown"
]

NUMERIC_COLS = [
    "Price (€)","Bedrooms_int","Bathrooms","lat","lon",
    "nearest_park_km","nearest_beach_km","nearest_gym_km","nearest_supermarket_km",
    "nearest_bus_stop_km","nearest_rail_station_km","nearest_tram_stop_km",
    "distance_to_city_centre_km","price_per_bedroom","energy_monthly_estimate",
    "effective_monthly_cost","min_transit_km",
]

PRIMARY_COLS = [
    "Address","Property Type","Bedrooms","Bathrooms","BER Rating",
    "Price (€)","effective_monthly_cost","price_per_bedroom",
    "distance_to_city_centre_km","min_transit_km","within_500m_transit",
    "energy_estimate_available","URL"
]

FULL_FILE = "cleaned_data_enriched.csv"  # ensure this file is in repo root
BED_OPTIONS  = list(range(1,10))         # 1..9 fixed choices
BATH_OPTIONS = list(range(1,10))         # 1..9 fixed choices

@st.cache_data(show_spinner=False)
def load_full_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # keep only rows with valid coordinates
    if {"lat","lon"}.issubset(df.columns):
        df = df.dropna(subset=["lat","lon"]).copy()
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]

    # numeric coercions used elsewhere
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # tidy helpers
    if "BER Rating" in df.columns:
        df["BER Rating"] = df["BER Rating"].fillna("Unknown").astype(str)
        df["BER Rating"] = pd.Categorical(df["BER Rating"], categories=BER_ORDER, ordered=True)
    if "Property Type" in df.columns:
        df["Property_Type"] = df["Property Type"]
    if "Price (€)" in df.columns:
        df["price_fmt"] = df["Price (€)"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
    if "distance_to_city_centre_km" in df.columns:
        df["dist_centre"] = df["distance_to_city_centre_km"].round(2)

    # create a rounded Bathrooms integer for filtering (handles 1.5 => 2, etc.)
    if "Bathrooms" in df.columns:
        df["Bathrooms_num"] = pd.to_numeric(df["Bathrooms"], errors="coerce")
        df["Bathrooms_int_approx"] = df["Bathrooms_num"].round().astype("Int64")

    # ensure Bedrooms_int is numeric if present
    if "Bedrooms_int" in df.columns:
        df["Bedrooms_int"] = pd.to_numeric(df["Bedrooms_int"], errors="coerce").astype("Int64")

    return df

def file_digest(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:10]
    except Exception:
        return "n/a"

# ------------------------------
# Load data
# ------------------------------
try:
    df = load_full_dataset(FULL_FILE)
except Exception as e:
    st.error(f"Couldn't load {FULL_FILE}. Make sure the file exists in the repo root.\nDetails: {e}")
    st.stop()

required_core = {"Address","Price (€)","lat","lon"}
missing = [c for c in required_core if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# Diagnostics (prove what the app is using)
mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(FULL_FILE))) if os.path.exists(FULL_FILE) else 'n/a'
st.caption(
    f"📄 Using file: **{FULL_FILE}** | rows: **{len(df)}** | md5: **{file_digest(FULL_FILE)}** | mtime: **{mtime}**"
)
if "Price (€)" in df.columns:
    pmin_actual = int(pd.to_numeric(df["Price (€)"], errors="coerce").min())
    pmax_actual = int(pd.to_numeric(df["Price (€)"], errors="coerce").max())
    st.caption(f"Data price range detected: €{pmin_actual:,} – €{pmax_actual:,}")

if st.button("🔄 Clear cache and reload"):
    st.cache_data.clear()
    st.rerun()

# ------------------------------
# Sidebar: Filters (collapsible groups)
# ------------------------------
with st.sidebar:
    st.divider()
    st.header("Filters")

    # --- Area search (text) ---
    search_q = st.text_input(
        "Search area (Address / Eircode)",
        placeholder="e.g., Raheny or D05 (comma-separate for multiple)"
    )
    match_mode = st.selectbox("Match mode", ["contains","starts with"], index=0)

    # --- Area quick-pick (derived from Address suffix) ---
    areas = []
    if "Address" in df.columns:
        candidates = df["Address"].dropna().astype(str).map(lambda s: s.split(",")[-1].strip())
        areas = sorted(pd.Series(candidates).unique().tolist())
    quick_pick = st.multiselect("Quick-pick area", areas, default=[])

    with st.expander("Price & size", expanded=True):
        # Price slider
        pmin = int(np.nanmin(df["Price (€)"]))
        pmax = int(np.nanmax(df["Price (€)"]))
        price_range = st.slider("Price (€)", min_value=pmin, max_value=pmax, value=(pmin, pmax), step=50)

        # --- Bedrooms: fixed 1..9, show all by default ---
        sel_beds = st.multiselect("Bedrooms", BED_OPTIONS, default=BED_OPTIONS)

        # --- Bathrooms (≈ nearest integer): fixed 1..9, show all by default ---
        sel_baths = st.multiselect("Bathrooms", BATH_OPTIONS, default=BATH_OPTIONS)

with st.expander("Type & energy", expanded=False):
    # Property type
    if "Property Type" in df.columns:
        types = sorted(df["Property Type"].dropna().unique())
        sel_types = st.multiselect("Property Type", types, default=types)
    else:
        sel_types = None

    # BER — include *all* ratings by default (including Exempt)
    if "BER Rating" in df.columns:
        ber_options = BER_ORDER[:]  # fixed full list, in order
        st.caption(f"BER Exempt in data: {(df['BER Rating'].astype(str) == 'Exempt').sum()} listings")
        include_exempt = st.checkbox("Include BER Exempt", value=True)
        default_ber = ber_options if include_exempt else [b for b in ber_options if b != "Exempt"]

        # Multiselect uses the full ordered list; defaults include/exclude Exempt via the checkbox
        sel_ber = st.multiselect("BER Rating", options=ber_options, default=default_ber)
    else:
        sel_ber = None

    energy_only = st.checkbox("Energy estimate available", value=False) if "energy_estimate_available" in df.columns else False

    with st.expander("Location & transit", expanded=False):
        if "distance_to_city_centre_km" in df.columns:
            dmax = float(np.nanmax(df["distance_to_city_centre_km"]))
            dist_centre = st.slider(
                "Max distance to city centre (km)",
                min_value=0.0, max_value=float(np.ceil(dmax)), value=float(np.ceil(dmax)), step=0.5
            )
        else:
            dist_centre = None

        within_500m = st.checkbox("Within 500m of any transit", value=False) if "within_500m_transit" in df.columns else False

        max_transit = None
        if "min_transit_km" in df.columns:
            tmax = float(np.nanmax(df["min_transit_km"]))
            max_transit = st.slider(
                "Max distance to nearest transit (km)",
                min_value=0.0, max_value=float(np.ceil(tmax)), value=float(np.ceil(tmax)), step=0.1
            )

    # Reset & legend
    if st.button("Reset filters"):
        st.session_state.clear()
        st.rerun()

    st.markdown("**Price bands (map colors)**")
    st.markdown("""
- 🟡 ≤ €1,499  
- 🟢 €1,500–2,000  
- 🔴 €2,001–2,500  
- 🔵 €2,501–3,000  
- 🟣 €3,001–4,000  
- ⚫ €4,001+
""")

# ------------------------------
# Apply filters (with debug counts)
# ------------------------------
f = df.copy()
debug_counts = {"start": len(f)}

# Price
f = f[(f["Price (€)"] >= price_range[0]) & (f["Price (€)"] <= price_range[1])]
debug_counts["price"] = len(f)

# Bedrooms (filter using Bedrooms_int if present; otherwise derive best-effort)
if "Bedrooms_int" in f.columns and f["Bedrooms_int"].notna().any():
    f = f[f["Bedrooms_int"].astype("Int64").isin(sel_beds)]
else:
    # fallback: try to parse from 'Bedrooms' text if needed
    if "Bedrooms" in f.columns:
        parsed = f["Bedrooms"].astype(str).str.extract(r"(\d+)").astype(float)[0].astype("Int64")
        f = f[parsed.isin(sel_beds)]
debug_counts["beds"] = len(f)

# Bathrooms (use rounded integer approx)
if "Bathrooms_int_approx" in f.columns:
    f = f[f["Bathrooms_int_approx"].isin(sel_baths)]
elif "Bathrooms" in f.columns:
    f["Bathrooms_num"] = pd.to_numeric(f["Bathrooms"], errors="coerce")
    f["Bathrooms_int_approx"] = f["Bathrooms_num"].round().astype("Int64")
    f = f[f["Bathrooms_int_approx"].isin(sel_baths)]
debug_counts["baths"] = len(f)

# Type / BER
if "Property Type" in f.columns and 'sel_types' in locals() and sel_types is not None:
    f = f[f["Property Type"].isin(sel_types)]
debug_counts["type"] = len(f)

if "BER Rating" in f.columns and 'sel_ber' in locals() and sel_ber is not None:
    f = f[f["BER Rating"].astype(str).isin(sel_ber)]
debug_counts["ber"] = len(f)

# Centre / transit / energy
if "distance_to_city_centre_km" in f.columns and 'dist_centre' in locals() and dist_centre is not None:
    f = f[f["distance_to_city_centre_km"] <= dist_centre]
debug_counts["dist_centre"] = len(f)

if "within_500m_transit" in f.columns and 'within_500m' in locals() and within_500m:
    f = f[f["within_500m_transit"] == True]
debug_counts["within_500m"] = len(f)

if "min_transit_km" in f.columns and 'max_transit' in locals() and max_transit is not None:
    f = f[f["min_transit_km"] <= max_transit]
debug_counts["transit"] = len(f)

if "energy_estimate_available" in f.columns and 'energy_only' in locals() and energy_only:
    f = f[f["energy_estimate_available"] == True]
debug_counts["energy_only"] = len(f)

# Area / Eircode text search
if search_q and search_q.strip():
    terms = [t.strip().lower() for t in search_q.split(",") if t.strip()]

    def _normalise(text):
        text = text.lower() if isinstance(text, str) else ""
        text = text.replace("dublin ", "d")  # dublin 5 ~ d5
        return text

    def _match(text: str) -> bool:
        t = _normalise(text)
        return any(term in t or t.replace("d", "dublin ") in term for term in terms)

    mask = None
    if "Address" in f.columns:
        mask = f["Address"].apply(_match)
    if "Eircode" in f.columns:
        m2 = f["Eircode"].apply(_match)
        mask = m2 if mask is None else (mask | m2)
    if mask is not None:
        f = f[mask]
debug_counts["search"] = len(f)

# Quick-pick area
if quick_pick and "Address" in f.columns:
    f = f[f["Address"].astype(str).str.strip().str.split(",").str[-1].str.strip().isin(quick_pick)]
debug_counts["quick_pick"] = len(f)

# ------------------------------
# KPIs
# ------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Listings", len(f))
with col2:
    st.metric("Avg price (€)", f["Price (€)"].mean().round(0) if len(f) else 0)
with col3:
    if "effective_monthly_cost" in f.columns:
        st.metric("Median effective cost (€)", int(f["effective_monthly_cost"].median()) if len(f) else 0)
    else:
        st.metric("Median effective cost (€)", "—")
with col4:
    if "distance_to_city_centre_km" in f.columns:
        st.metric("Avg km to city centre", round(f["distance_to_city_centre_km"].mean(), 2) if len(f) else 0)
    else:
        st.metric("Avg km to city centre", "—")

st.divider()

# ------------------------------
# Map (token-free, fixed Dublin view)
# ------------------------------
if "Price (€)" in f.columns:
    def price_to_color(price):
        if price <= 1499: return [255, 255, 0]
        if price <= 2000: return [0, 255, 0]
        if price <= 2500: return [255, 0, 0]
        if price <= 3000: return [0, 0, 255]
        if price <= 4000: return [128, 0, 128]
        return [0, 0, 0]
    f["color"] = f["Price (€)"].apply(price_to_color)

g = f.copy()
for col in ["lat", "lon"]:
    if col in g.columns:
        g[col] = pd.to_numeric(g[col], errors="coerce")
g = g.dropna(subset=["lat","lon"])
g = g[g["lat"].between(-90, 90) & g["lon"].between(-180, 180)]

DUBLIN_LAT, DUBLIN_LON, DUBLIN_ZOOM = 53.3498, -6.2603, 11.0
view = pdk.ViewState(latitude=DUBLIN_LAT, longitude=DUBLIN_LON, zoom=DUBLIN_ZOOM)

if len(g) == 0:
    st.info("No mappable rows after filters. Showing Dublin.")
    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view))
else:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=g,
        get_position='[lon, lat]',
        get_radius=50,
        get_fill_color='color' if "color" in g.columns else [0, 122, 255],
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": """
            <div style='font-size:12px'>
              <b>{Address}</b><br/>
              €{price_fmt} • {Bedrooms} bed • {Property_Type}<br/>
              BER: {BER Rating} • {dist_centre} km to centre<br/>
              Transit: {min_transit_km} km
            </div>
        """,
        "style": {"backgroundColor": "#111", "color": "#fff"}
    }
    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view, layers=[layer], tooltip=tooltip))

    st.markdown(
        """
        <div style='display:flex; justify-content:flex-end; margin-top:-10px;'>
          <div style='background: rgba(255,255,255,0.95); padding:8px 10px; border:1px solid #ddd; border-radius:8px; font-size:12px;'>
            <div style='font-weight:600; margin-bottom:4px;'>Price bands</div>
            <div><span style='display:inline-block;width:12px;height:12px;background:#ffff00;border:1px solid #999;margin-right:6px;'></span>≤ €1,499</div>
            <div><span style='display:inline-block;width:12px;height:12px;background:#00ff00;border:1px solid #999;margin-right:6px;'></span>€1,500–2,000</div>
            <div><span style='display:inline-block;width:12px;height:12px;background:#ff0000;border:1px solid #999;margin-right:6px;'></span>€2,001–2,500</div>
            <div><span style='display:inline-block;width:12px;height:12px;background:#0000ff;border:1px solid #999;margin-right:6px;'></span>€2,501–3,000</div>
            <div><span style='display:inline-block;width:12px;height:12px;background:#800080;border:1px solid #999;margin-right:6px;'></span>€3,001–4,000</div>
            <div><span style='display:inline-block;width:12px;height:12px;background:#000000;border:1px solid #999;margin-right:6px;'></span>€4,001+</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ------------------------------
# Charts
# ------------------------------
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

# ------------------------------
# Table with Link column + Favourites + Saved panel
# ------------------------------
show_cols = [c for c in PRIMARY_COLS if c in f.columns]
if len(f):
    st.subheader("Matching listings")

    col_cfg = {}
    if "URL" in show_cols:
        col_cfg["URL"] = st.column_config.LinkColumn(label="Daft Listing", display_text="Open")
    if "Price (€)" in show_cols:
        col_cfg["Price (€)"] = st.column_config.NumberColumn(label="Price (€)", format="%.0f")
    if "effective_monthly_cost" in show_cols:
        col_cfg["effective_monthly_cost"] = st.column_config.NumberColumn(label="Effective € / mo", format="%.0f")
    if "price_per_bedroom" in show_cols:
        col_cfg["price_per_bedroom"] = st.column_config.NumberColumn(label="€ per bedroom", format="%.0f")
    if "distance_to_city_centre_km" in show_cols:
        col_cfg["distance_to_city_centre_km"] = st.column_config.NumberColumn(label="Km to centre", format="%.2f")
    if "min_transit_km" in show_cols:
        col_cfg["min_transit_km"] = st.column_config.NumberColumn(label="Km to transit", format="%.2f")

    key_col = "URL" if "URL" in f.columns else "Address"
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

# ------------------------------
# Details panel
# ------------------------------
st.markdown("### Quick details")
st.caption("Tick ⭐ on a row and click the button above to show details here.")

sel_key = st.session_state.get("selected_key")
if sel_key:
    key_col = "URL" if "URL" in f.columns else "Address"
    if key_col in f.columns and sel_key in set(f[key_col]):
        row = f[f[key_col] == sel_key].iloc[0]
        lines = []
        def add_line(label, col):
            if col in f.columns and pd.notna(row[col]):
                lines.append(f"**{label}:** {row[col]}")
        add_line("Address", "Address")
        add_line("Price (€)", "Price (€)")
        add_line("Bedrooms", "Bedrooms")
        add_line("Bathrooms", "Bathrooms")
        add_line("BER", "BER Rating")
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

# ------------------------------
# Footer / Debug
# ------------------------------
with st.expander("Debug & Schema"):
    st.write("Columns present:", list(df.columns))
    st.write("Rows (original → filtered):", len(df), "→", len(f))
    st.write("Filter step counts:", debug_counts)
    if "Bathrooms" in df.columns:
        st.write("Bathrooms unique (rounded 1dp):", sorted(df['Bathrooms'].dropna().round(1).unique().tolist()))
    if "Bedrooms_int" in df.columns:
        st.write("Bedrooms unique (from data):", sorted([int(x) for x in df['Bedrooms_int'].dropna().unique()]))
    if "Price (€)" in df.columns:
        st.write("Price min/max:", int(df["Price (€)"].min()), int(df["Price (€)"].max()))
    if "URL" in f.columns and len(f):
        st.write("Example URL:", f["URL"].iloc[0])

st.caption("MVP. Add scoring, SHAP, and model predictions later. 🚀")
