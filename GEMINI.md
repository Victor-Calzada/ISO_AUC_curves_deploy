# Project Context
Building interactive data apps using Python.

# Tech Stack
- **Marimo:** For the notebook interface (files are .py).
- **Polars:** For data handling.
- **Plotly:** For plotting (preferred over Matplotlib for Marimo).

# Coding Standards
- **Reactivity:** Ensure UI elements (sliders) directly filter dataframes.
- **Format:** Output code as a valid marimo `.py` file (using `@app.cell` decorators).
- **Style:** Use `mo.vstack` or `mo.hstack` to organize the layout cleanly.
