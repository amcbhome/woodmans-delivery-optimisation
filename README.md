# Woodman's Delivery Optimisation

A Streamlit decision-support app implementing a reduced 5-customer version of the Woodman's Grocery Delivery heterogeneous fleet vehicle routing problem with time windows (HVRPTW).

## Stack

- Python
- PuLP / CBC
- Streamlit
- Pandas

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

The app translates the LP/MIP structure into Python/PuLP and provides interactive controls for:

- Woodman's vehicle availability
- Rideshare vehicle availability
- Penalty applied to rejected cartons

The app reports the objective value, demand served, rejected customers and the resulting routes.

**Important:** this is a reduced educational instance. The customer coordinates and distances are illustrative and are not the original 30-customer NEOS data.

## Source

[NEOS Guide — Woodman's Delivery](https://neos-guide.org/case-studies/tra/woodmans-delivery/)
