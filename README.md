# Dublin Rental Intelligence App 🏠📊

An interactive data app built to explore fairness in Dublin’s rental market.  
It helps users find good-value listings by comparing asking rent to estimated sale value and quality-of-life factors.

[🔗 Live app](https://lee-rentals-dashboard.streamlit.app/) • [💻 Source code](https://github.com/LeeGallagher42/RentersInsightProject)

---

## 🎯 What This App Does

- **Scrapes real Dublin rental listings** from Daft.ie
- **Enriches data** with BER scores, proximity to parks, transit, gyms, supermarkets, etc.
- **Estimates sale value** using area-level benchmarks (via Property Price Register)
- **Flags listings** as ✅ Fair, 🔥 Underpriced, or 💰 Overpriced based on a rent-to-value ratio
- **Lets users filter** by price, BER rating, number of bedrooms, distance to city centre, and more
- **Displays maps, filters, dashboards and ranking logic** — all in a polished Streamlit interface

---

## 🧠 How Value Is Judged

We estimate what a listing *might sell for* based on recent sale data in its area (postcode-level).  
Then we calculate a **rental yield**, and flag listings using a transparent set of thresholds:

- 🔥 **Underpriced** → rent much lower than estimated value
- ✅ **Fair** → rent within ±5% of expected
- 💰 **Overpriced** → rent far above typical value for that area

This isn’t a prediction model yet — but the value flagging logic is honest, transparent, and highlights anomalies that renters can act on.

---

## ⚙️ Tools & Stack

- **Python**: pandas, numpy, altair, pydeck
- **Streamlit**: full front-end, user filtering, KPIs, map layers, and download features
- **Geospatial enrichment**: location-based features like proximity to gyms, supermarkets, parks
- **Custom feature engineering**: `price_per_bedroom`, `effective_monthly_cost`, energy estimate proxies, transit access flags, etc.
- **Clean design**: mobile-usable, intuitive, and ready to demo

---

## 👨‍💻 Built by

**Lee Gallagher** — data analyst with a finance + computer science background.  

[🔗 LinkedIn](https://www.linkedin.com/in/lee-gallagher-7ba1721a3/)  
[💻 GitHub](https://github.com/LeeGallagher42)

---

## Link to app

https://lee-rentals-dashboard.streamlit.app/
