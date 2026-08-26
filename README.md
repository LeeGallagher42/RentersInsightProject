# Dublin Rental Intelligence App 🏠📊

A deployed Python/Streamlit data product for exploring value and trade-offs in Dublin rental listings. It combines scraped rental data, feature engineering, geospatial enrichment and transparent scoring in an interactive app.

[🔗 Live app](https://lee-rentals-dashboard.streamlit.app/) • [💻 Source code](https://github.com/LeeGallagher42/RentersInsightProject)

---

## Why I built it

Rental listings are easy to browse but hard to compare. Asking rent alone does not tell a renter whether a property is relatively good value once bedrooms, location, BER, transport and nearby amenities are considered.

The project turns that messy decision into a structured data workflow: collect listings, normalize them, enrich them with external/contextual features, calculate interpretable value indicators and expose the result through filters, maps and dashboards.

---

## What the app does

- Scrapes Dublin rental listings through the project scraper notebook.
- Enriches listings with BER and proximity features for parks, beaches, gyms, supermarkets and public transport.
- Uses engineered fields such as price per bedroom, effective monthly cost, distance to the city centre and minimum transit distance.
- Includes predicted-price / fairness fields and transparent value badges from **Very underpriced** through **Very overpriced**.
- Lets users filter by price, bedrooms, bathrooms, BER, property type, distance and other practical criteria.
- Displays interactive map layers, KPIs, ranking logic and listing-level detail in Streamlit.

---

## Architecture

```text
Rental listing collection
        │
        ▼
Raw / scraped listing data
        │
        ▼
Cleaning + type normalisation
        │
        ├── BER normalisation
        ├── coordinate validation
        └── numeric coercion
        │
        ▼
Feature engineering + enrichment
        │
        ├── amenity / transit distance
        ├── city-centre distance
        ├── price-per-bedroom
        ├── effective monthly cost
        └── predicted-value / fairness fields
        │
        ▼
Curated CSV snapshot
        │
        ▼
Streamlit application
        ├── filters
        ├── KPIs
        ├── charts
        ├── map layers
        └── listing-level comparisons
```

This separation matters: the app is not responsible for inventing business logic at runtime. The data is prepared and validated upstream, while the UI focuses on exploration and explanation.

---

## Reliability / defensive data handling

The application contains explicit defensive handling rather than assuming the dataset is clean:

- Coordinates are coerced to numeric values and rejected when outside valid latitude/longitude ranges.
- Numeric fields are normalised with `errors="coerce"` so malformed values become observable missing data rather than silently corrupting calculations.
- BER values are normalised into an ordered controlled set with an `Unknown` fallback.
- Amenity columns are normalised when enrichment produces suffixed field names.
- External image data is attached through a normalised URL key, with a many-to-one merge validation.
- Data loading is cached and app startup stops with an explicit error if the required dataset cannot be loaded.
- Value categories are implemented through explicit thresholds rather than an opaque LLM decision.

Those choices are intentionally boring. They are also the same kind of choices that make automation and data products maintainable in production: validate inputs, make failure visible, keep deterministic rules deterministic and document the hand-off between stages.

---

## Tools & stack

- **Python** — pandas, numpy and application logic
- **Streamlit** — interactive UI, filtering and deployment
- **PyDeck** — geospatial map layers
- **Altair** — charts / visual exploration
- **Jupyter notebooks** — scraper and feature-engineering stages
- **CSV snapshots** — simple, inspectable hand-off between data preparation and the app
- **Git / GitHub** — source control and public delivery

Repository structure includes:

- `app.py` — deployed Streamlit application
- `Scripts/Scraper.ipynb` — listing collection workflow
- `Scripts/feature_engineering.ipynb` — feature preparation / enrichment
- `cleaned_data_enriched_with_fairness.csv` — application-ready data
- `feature_importances.csv` and `predictions_snapshot.csv` — modelling artefacts

---

## What this project demonstrates

This is useful as more than a dashboard demo. It demonstrates how I approach ambiguous business/data problems:

1. **Turn an unclear question into explicit decision variables.**
2. **Separate collection, transformation and presentation.**
3. **Engineer features that map to the real decision rather than simply displaying raw data.**
4. **Use deterministic rules where transparency matters.**
5. **Handle dirty inputs and edge cases explicitly.**
6. **Ship the result as something another person can actually use.**

That same workflow — process mapping → data contract → transformation / automation → controls → usable interface — is the foundation I use for AI and business-automation projects as well.

---

## Limitations

- Rental-market data changes quickly, so any static snapshot becomes stale.
- Fairness/value indicators are decision-support signals, not financial advice or guaranteed market valuations.
- Amenity proximity does not capture every qualitative factor a renter may care about.
- Model-derived fields should be interpreted alongside the underlying listing features, not treated as ground truth.

Being explicit about those limitations is part of the project: a useful analytical tool should communicate what it knows and what it does not.

---

## Built by

**Lee Gallagher** — automation-focused business analyst/developer with a finance and computer-science background.

[LinkedIn](https://www.linkedin.com/in/lee-gallagher-7ba1721a3/) • [GitHub](https://github.com/LeeGallagher42)
