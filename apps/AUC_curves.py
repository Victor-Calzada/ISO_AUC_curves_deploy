import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _(mo, np, pl):

    Cu2 = np.arange(0, 0.5, 0.033) 
    Ni2 = np.arange(0, 1.8, 0.133) 


    XCu, YNi = np.meshgrid(Cu2, Ni2)



    Fl = np.array([i * 1e18 for i in range(1, int(1e20 / 1e18) + 1)])*1e4  # n/m2

    # -----------------------------
    file = mo.notebook_location() / "public" / "df_plotter.csv"
    df_plotter = pl.read_csv(str(file))
    return Cu2, Fl, Ni2, df_plotter


@app.cell
def _():
    # Tslide = mo.ui.slider(start=df_plotter["Temperature_Celsius"].min(), stop=df_plotter["Temperature_Celsius"].max(), step=df_plotter["Temperature_Celsius"].std()/3, value=df_plotter["Temperature_Celsius"].mean(), label="Temperature_Celsius")
    # Mnslide = mo.ui.slider(start=df_plotter["Mn"].min(), stop=df_plotter["Mn"].max(), step=df_plotter["Mn"].std()/3, value=df_plotter["Mn"].mean(), label="Mn")
    # Pslide = mo.ui.slider(start=df_plotter["P"].min(), stop=df_plotter["P"].max(), step=df_plotter["P"].std()/3, value=df_plotter["P"].mean(), label="P")
    return


@app.cell
def _(df_plotter, mo, np):
    Tslide = mo.ui.slider(steps=np.arange(df_plotter["Temperature_Celsius"].min(), df_plotter["Temperature_Celsius"].max(), df_plotter["Temperature_Celsius"].std()/3), label="")
    Mnslide = mo.ui.slider(steps=np.arange(df_plotter["Mn"].min(), df_plotter["Mn"].max(), df_plotter["Mn"].std()/3), label="")
    Pslide = mo.ui.slider(steps=np.arange(df_plotter["P"].min(), df_plotter["P"].max(), df_plotter["P"].std()/3), label="")
    return Mnslide, Pslide, Tslide


@app.cell
def _(mo):
    pfdrop = mo.ui.dropdown(["F", "P", "W"], label="Product Form", value="W")
    return (pfdrop,)


@app.cell
def _(Mnslide, Pslide, Tslide, pfdrop):
    T = Tslide.value
    P = Pslide.value
    Mn = Mnslide.value
    pf = pfdrop.value
    return Mn, P, T, pf


@app.cell
def _(
    Cu2,
    Fl,
    Mn,
    Ni2,
    P,
    T,
    TTS_eval,
    df_and_dist,
    df_plotter,
    go,
    mo,
    np,
    pf,
    simpson,
):
    ZT = np.zeros((len(Ni2), len(Cu2)))
    c=np.arange(0,4000,250)

    for nii in range(len(Ni2)):
        for cui in range(len(Cu2)):
            ZT[nii, cui] = simpson(y=TTS_eval(pf, Cu2[cui], Ni2[nii], Mn, P, T, Fl), x=Fl*1e-23)


    fig = go.Figure()
    _max = int(np.max(ZT))
    fig.add_trace(go.Contour(z = ZT,x=Cu2, y=Ni2,  contours_coloring='lines',colorscale="Sunsetdark", contours=dict(showlabels=True, start=0, end=_max, size=200),showscale=False, hoverinfo="none"))
    fig.add_trace(go.Scatter(x=df_plotter["Cu"], y=df_plotter["Ni"], mode="markers", opacity=0.3, 
                                 marker=dict(color="grey",colorscale='Sunsetdark', # <--- Define la paleta de color para AUC
                colorbar=dict(
                    title='AUC', # Título de la barra de color
                    # Puedes ajustar los ticks si lo necesitas
                    # tickvals=[0, 100, 200, 300, 400]
                ), showscale=True,line=dict(width=0.3, color='DarkSlateGrey'),cmin=np.min(ZT),cmax=np.max(ZT)),hoverinfo="none", showlegend=False))
    aux_df, dist = df_and_dist(df_plotter, P, Mn, T, pf)
    if aux_df.shape[0] != 0:

        AUC = simpson(y = np.array([TTS_eval(pf, cu=aux_df["Cu"].to_numpy(), ni=aux_df["Ni"].to_numpy(), mn=Mn, p=P, t=T, fl=Fli) for Fli in Fl]), x=Fl*1e-23, axis=0)
        fig.add_trace(go.Scatter(x=aux_df["Cu"], y=aux_df["Ni"], mode="markers",  
                                     marker=dict(color=AUC,colorscale='Sunsetdark', opacity=dist,
                    colorbar=dict(
                        title='AUC', # Título de la barra de color
                        # Puedes ajustar los ticks si lo necesitas
                        # tickvals=[0, 100, 200, 300, 400]
                    ), showscale=True,
                                                 line=dict(width=0.3, color='DarkSlateGrey'), cmin=np.min(ZT),cmax=np.max(ZT)),
                                     customdata=AUC,
                                     hovertemplate=(
                    "<b>X:</b> %{x:.2f}<br>" +  # Muestra el valor X, formateado a 2 decimales
                    "<b>Y:</b> %{y:.2f}<br>" +  # Muestra el valor Y, formateado a 2 decimales
                    "<b>AUC:</b> %{customdata:.2f}<extra></extra>" ), 
                                 showlegend=False))
    fig.update_layout(title=f"Cu-Ni [{pf}] -- AUC", xaxis_title="Cu", yaxis_title="Ni")
    fig.update_layout(xaxis=dict(range=[0,np.max(Cu2)]), yaxis=dict(range=[0,np.max(Ni2)]))
    fig.update_layout(width=800, height=800)
    mofig = mo.ui.plotly(fig)
    return (mofig,)


@app.cell
def _(Mn, Mnslide, P, Pslide, T, Tslide, df_plotter, mo, pfdrop):
    commands = mo.vstack([mo.hstack([Tslide, mo.md(f"{T=:.2f} +- {df_plotter['Temperature_Celsius'].std()/2:.2f}")], justify="center"), 
               mo.hstack([Mnslide, mo.md(f"{Mn=:.3f} +- {df_plotter['Mn'].std()/2:.3f}")], justify="center"), 
               mo.hstack([Pslide, mo.md(f"{P=:.3f} +- {df_plotter['P'].std()/2:.3f}")], justify="center"), 
               mo.hstack([pfdrop], justify="center")])
    return (commands,)


@app.cell
def _(commands, mo, mofig):
    mo.hstack([commands, mofig], justify="center", align="center")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from scipy.integrate import simpson
    from itertools import product

    return np, pl, simpson


@app.cell
def _():
    import plotly.graph_objects as go
    return (go,)


@app.cell
def _(np):
    def minimo (a,b):
        return (0.5 * ( a+b - abs(a-b)))

    def maximo (a,b):
        return (0.5 * ( a+b + abs(a-b)))

    def TTS_eval(pf,cu,ni,mn,p,t,fl): 
        #fl un np.array
        if pf == 'F':
            A=1.011
            B=0.738
        elif (pf == 'P') or (pf == 'SRM'):
            A=1.080
            B=0.819
        elif pf == 'W':
            A=0.919
            B=0.968
        else:
            raise ValueError("Product Form no admisible")

        TTS1 = A*5/9*1.8943e-12*fl**(0.5695)*((1.8*t+32)/550)**(-5.47)*(0.09+p/0.012)**(0.216)*(1.66+ni**(8.54)/0.63)**(0.39)*(mn/1.36)**(0.3)

        M = B*maximo(minimo(113.87*(np.log(fl)-np.log(4.5e20)),612.6),0)*((1.8*t+32)/550)**(-5.45)*(0.1+p/0.012)**(-0.098)*(0.168+ni**(0.58)/0.63)**(0.73)

        TTS2 = 5/9*M*maximo((minimo(cu,0.28)-0.053),0)

        return TTS1+TTS2
    return (TTS_eval,)


@app.cell
def _(np, pl):
    def df_and_dist(df: pl.DataFrame, P: float, Mn: float, T: float, pf: str):
        """
        Filters a Polars DataFrame based on specified parameters (P, Mn, T, pf)
        and calculates a normalized Euclidean distance for the filtered data.

        Args:
            df (pl.DataFrame): The input Polars DataFrame.
            P (float): Reference value for 'P' column.
            Mn (float): Reference value for 'Mn' column.
            T (float): Reference value for 'Temperature_Celsius' column.
            pf (str): Reference value for 'Product_Form' column.

        Returns:
            tuple[pl.DataFrame, np.ndarray]: A tuple containing:
                - aux_df (pl.DataFrame): The filtered DataFrame.
                - dist (np.ndarray): A NumPy array of normalized distances.
        """
        # Calculate half standard deviation for each numerical column
        # Polars Series have a .std() method similar to Pandas
        stdP = df["P"].std() / 2
        stdMn = df["Mn"].std() / 2
        stdT = df["Temperature_Celsius"].std() / 2

        # Filter the DataFrame using Polars' .filter() method and pl.col() for expressions.
        # Using .is_between() for cleaner range checks.
        aux_df = df.filter(
            (pl.col("Product_Form") == pf) &
            (pl.col("P").is_between(P - stdP, P + stdP)) &
            (pl.col("Mn").is_between(Mn - stdMn, Mn + stdMn)) &
            (pl.col("Temperature_Celsius").is_between(T - stdT, T + stdT))
        )

        # Check if the filtered DataFrame is empty
        # In Polars, df.height gives the number of rows
        if aux_df.height == 0:
            return aux_df, np.array([0.0]) # Return a NumPy array for consistency

        # Extract columns as NumPy arrays for distance calculation
        # Polars Series also have a .to_numpy() method
        px = aux_df["P"].to_numpy()
        mny = aux_df["Mn"].to_numpy()
        Tz = aux_df["Temperature_Celsius"].to_numpy()

        # Calculate squared differences
        dpx = (px - P)**2
        dmny = (mny - Mn)**2
        dTz = (Tz - T)**2

        # Calculate Euclidean distance
        dist = np.sqrt(dpx + dmny + dTz)

        # Normalize the distance to a range (0.5 to 1.0)
        min_orig = np.min(dist)
        max_orig = np.max(dist)

        if max_orig == min_orig:
            # If all distances are the same, set them to 1.0 (or 0.5 depending on desired behavior)
            # The original code set to np.ones(len(dist)), which would be 1.0.
            # Given the normalization to 0.5 + ..., setting to 0.5 or 1.0 might be more consistent.
            # Sticking to the original's np.ones for now, which implies a normalized value of 1.
            dist = np.ones(len(dist))
        else:
            normalized_to_0_1 = (dist - min_orig) / (max_orig - min_orig)
            # The original code had (1 - 0.5) which simplifies to 0.5.
            # This normalizes to the range [0.5, 1.0] if normalized_to_0_1 is [0, 1].
            dist = 0.5 + (normalized_to_0_1 * 0.5)

        return aux_df, dist
    return (df_and_dist,)


if __name__ == "__main__":
    app.run()
