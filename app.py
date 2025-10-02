import os
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

# ------------------------------
# Data picker (new cloud-friendly)
# ------------------------------
with st.sidebar:
    st.header("Data")
    st.caption("Pick one option below")
    data_mode = st.radio(
        "Source",
        ["Use bundled sample.csv", "Paste CSV URL", "Upload CSV"],
        index=0
    )

    csv_url = None
    uploaded = None
    if data_mode == "Paste CSV URL":
        csv_url = st.text_input("CSV URL (raw CSV)", placeholder="https://...")
    elif data_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

@st.cache_data(show_spinner=False)
def load_data_from_source(mode, url, uploaded_file):
    import io
    if mode == "Use bundled sample.csv":
        df = pd.read_csv("sample.csv")
    elif mode == "Paste CSV URL":
        if not url:
            st.stop()
        df = pd.read_csv(url)
    elif mode == "Upload CSV":
        if not uploaded_file:
            st.stop()
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

    # Coerce numerics etc. (your existing logic)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "BER Rating" in df.columns:
        df["BER Rating"] = df["BER Rating"].fillna("Unknown").astype(str)
        df["BER Rating"] = pd.Categorical(df["BER Rating"], categories=BER_ORDER, ordered=True)

    if "Property Type" in df.columns:
        df["Property_Type"] = df["Property Type"]
    if "Price (€)" in df.columns:
        df["price_fmt"] = df["Price (€)"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
    if "distance_to_city_centre_km" in df.columns:
        df["dist_centre"] = df["distance_to_city_centre_km"].round(2)

    return df

try:
    df = load_data_from_source(data_mode, csv_url, uploaded)
except Exception as e:
    st.error(f"Couldn't load data. Check your source.\nDetails: {e}")
    st.stop()


required_core = {"Address","Price (€)","lat","lon"}
missing = [c for c in required_core if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

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
        # naive area extraction: last comma-part of address
        candidates = df["Address"].dropna().astype(str).map(lambda s: s.split(",")[-1].strip())
        areas = sorted(pd.Series(candidates).unique().tolist())
    quick_pick = st.multiselect("Quick-pick area", areas, default=[])

    with st.expander("Price & size", expanded=True):
        pmin = int(np.nanmin(df["Price (€)"]))
        pmax = int(np.nanmax(df["Price (€)"]))
        price_range = st.slider("Price (€)", min_value=pmin, max_value=pmax, value=(pmin, pmax), step=50)

        if "Bedrooms_int" in df.columns:
            beds_vals = sorted([int(x) for x in df["Bedrooms_int"].dropna().unique()])
            sel_beds = st.multiselect("Bedrooms", beds_vals, default=beds_vals)
        else:
            sel_beds = None

        if "Bathrooms" in df.columns:
            bath_vals = sorted([int(x) for x in df["Bathrooms"].dropna().unique()])
            sel_baths = st.multiselect("Bathrooms", bath_vals, default=bath_vals)
        else:
            sel_baths = None

    with st.expander("Type & energy", expanded=False):
        if "Property Type" in df.columns:
            types = sorted(df["Property Type"].dropna().unique())
            sel_types = st.multiselect("Property Type", types, default=types)
        else:
            sel_types = None

        if "BER Rating" in df.columns:
            sel_ber = st.multiselect(
                "BER Rating", BER_ORDER,
                default=[b for b in BER_ORDER if b in set(df["BER Rating"].astype(str))]
            )
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
# Apply filters
# ------------------------------
f = df.copy()

# Price
f = f[(f["Price (€)"] >= price_range[0]) & (f["Price (€)"] <= price_range[1])]

# Beds / baths / type / BER
if sel_beds is not None and "Bedrooms_int" in f.columns:
    f = f[f["Bedrooms_int"].isin(sel_beds)]
if sel_baths is not None and "Bathrooms" in f.columns:
    f = f[f["Bathrooms"].isin(sel_baths)]
if sel_types is not None and "Property Type" in f.columns:
    f = f[f["Property Type"].isin(sel_types)]
if sel_ber is not None and "BER Rating" in f.columns:
    f = f[f["BER Rating"].astype(str).isin(sel_ber)]

# Centre / transit / energy
if dist_centre is not None and "distance_to_city_centre_km" in f.columns:
    f = f[f["distance_to_city_centre_km"] <= dist_centre]
if within_500m and "within_500m_transit" in f.columns:
    f = f[f["within_500m_transit"] == True]
if max_transit is not None and "min_transit_km" in f.columns:
    f = f[f["min_transit_km"] <= max_transit]
if energy_only and "energy_estimate_available" in f.columns:
    f = f[f["energy_estimate_available"] == True]

# Area / Eircode text search
if search_q and search_q.strip():
    terms = [t.strip().lower() for t in search_q.split(",") if t.strip()]

    def _normalise(text):
        text = text.lower() if isinstance(text, str) else ""
        # Simple Dublin postcode normalisation: d05 -> dublin 5
        text = text.replace("dublin ", "d")
        return text

    def _match(text: str) -> bool:
        t = _normalise(text)
        return any(
            term in t or t.replace("d", "dublin ") in term
            for term in terms
        )

    mask = None
    if "Address" in f.columns:
        mask = f["Address"].apply(_match)
    if "Eircode" in f.columns:
        m2 = f["Eircode"].apply(_match)
        mask = m2 if mask is None else (mask | m2)

    if mask is not None:
        f = f[mask]


# Quick-pick area
if quick_pick and "Address" in f.columns:
    f = f[f["Address"].astype(str).str.strip().str.split(",").str[-1].str.strip().isin(quick_pick)]

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
# Map (robust + token-free)
# ------------------------------
import pydeck as pdk

# Color by price (same logic as before)
if "Price (€)" in f.columns:
    def price_to_color(price):
        if price <= 1499: return [255, 255, 0]
        if price <= 2000: return [0, 255, 0]
        if price <= 2500: return [255, 0, 0]
        if price <= 3000: return [0, 0, 255]
        if price <= 4000: return [128, 0, 128]
        return [0, 0, 0]
    f["color"] = f["Price (€)"].apply(price_to_color)

# Coerce & validate coords
g = f.copy()
for col in ["lat", "lon"]:
    if col in g.columns:
        g[col] = pd.to_numeric(g[col], errors="coerce")
g = g.dropna(subset=["lat","lon"])
g = g[g["lat"].between(-90, 90) & g["lon"].between(-180, 180)]

if len(g) == 0:
    st.info("No mappable rows after filters. Showing Dublin as a placeholder.")
    view = pdk.ViewState(latitude=53.3498, longitude=-6.2603, zoom=10.5)
    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view))
else:
    view = pdk.ViewState(latitude=float(g["lat"].mean()),
                         longitude=float(g["lon"].mean()),
                         zoom=10.5)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=g,
        get_position='[lon, lat]',     # NOTE: lon first, then lat
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

    st.pydeck_chart(pdk.Deck(
        map_style=None,                # <— important: no Mapbox token needed
        initial_view_state=view,
        layers=[layer],
        tooltip=tooltip
    ))

    # Legend (unchanged)
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

    # Column configs (price formatting + link)
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

    # Stable key for rows
    key_col = "URL" if "URL" in f.columns else "Address"

    display = f[show_cols].copy().reset_index(drop=True)

    # Favourites state
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

    # Update favourites
    st.session_state["favs"] = set(edited.loc[edited["Favourite"] == True, key_col].tolist())

    # Connect to details panel via first favourite
    if st.button("Show details for selected (first ⭐)"):
        sel_rows = edited.loc[edited["Favourite"] == True]
        if len(sel_rows):
            st.session_state["selected_key"] = sel_rows.iloc[0][key_col]
            st.rerun()

    # Saved list
    with st.expander("⭐ Saved listings"):
        saved = edited.loc[edited["Favourite"] == True, show_cols + ([key_col] if key_col not in show_cols else [])]
        if len(saved):
            st.dataframe(saved, use_container_width=True, hide_index=True, column_config=col_cfg)
        else:
            st.caption("No favourites yet — tick ⭐ in the table above.")

    # Download
    csv_bytes = f[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="filtered_listings.csv", mime="text/csv")
else:
    st.warning("No listings match your filters. Try widening your criteria.")

# ------------------------------
# Details panel (populated when a row is chosen)
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
# Footer
# ------------------------------
with st.expander("Debug & Schema"):
    st.write("Columns present:", list(df.columns))
    st.write("Rows (original → filtered):", len(df), "→", len(f))
    if "URL" in f.columns and len(f):
        st.write("Example URL:", f["URL"].iloc[0])

st.caption("MVP. Add scoring, SHAP, and model predictions later.")
