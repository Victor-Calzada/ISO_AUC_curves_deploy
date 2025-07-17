import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _(mo, np, pl):
    file = mo.notebook_location() / "public" / "df_plotter.csv"
    file2 = mo.notebook_location() / "public" / "df_plotter_astm_auc.csv"
    df_plotter = pl.read_csv(str(file))
    df_plotter_auc = pl.read_csv(str(file2))

    pf = df_plotter["Product_Form"].unique().to_list()
    # --- Initialize Fl ---
    # This part remains largely the same as it's a NumPy array creation.
    Fl = np.array([i * 1e18 for i in range(1, int(1e20 / 1e18) + 1)]) * 1e4 # n/m2

    df_plotter = df_plotter.join(df_plotter_auc, on=["Product_Form", "Cu", "Ni", "Mn", "P", "Temperature_Celsius"], how="left")
    # # --- Calculate ranges for figures using Polars expressions ---
    # # Polars provides direct min() and max() methods on Series.
    rangex_fig1 = [df_plotter["Cu"].min() - 0.025, df_plotter["Cu"].max() + 0.025]
    rangey_fig1 = [df_plotter["Ni"].min() - 0.075, df_plotter["Ni"].max() + 0.075]
    rangex_fig2 = [df_plotter["Fluence_1E19_n_cm2"].min() - 1, df_plotter["Fluence_1E19_n_cm2"].max() + 1]
    rangey_fig2 = [df_plotter["DT41J_Celsius"].min() - 10, df_plotter["DT41J_Celsius"].max() + 10]

    return df_plotter, pf, rangex_fig1, rangex_fig2, rangey_fig1, rangey_fig2


@app.cell
def _(mo):

    slstep = mo.ui.slider(start=10, stop=200, step=20, show_value=False, label="AUC Gap")

    return (slstep,)


@app.cell
def _(df_plotter, mo, np, slstep):
    step = slstep.value
    umbralAUC = np.arange(df_plotter['AUC'].min(), df_plotter['AUC'].max()+step,step)
    tope = len(umbralAUC)-2

    slindex = mo.ui.slider(start=0, stop=tope, step=1, show_value=False, label="AUC Gap Value")
    return slindex, umbralAUC


@app.cell
def _(mo):
    switch = mo.ui.switch(label="ASTM curves", value=False)
    return (switch,)


@app.cell
def _(mo, slindex, slstep, switch):
    vstack_cntrols = mo.vstack([slstep,slindex, switch])
    return (vstack_cntrols,)


@app.cell
def _(slindex):

    indx = slindex.value
    return (indx,)


@app.cell
def _(df_plotter, go, indx, mo, umbralAUC):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=[umbralAUC[indx], umbralAUC[indx + 1]], y=[0,0], mode="markers+lines", marker=dict(size=20)))
    _fig.update_xaxes(showgrid=False, range=(df_plotter["AUC"].min() - 0.1, df_plotter["AUC"].max() + 0.1),)
    _fig.update_yaxes(showgrid=False, 
                     zeroline=True, zerolinecolor='black', zerolinewidth=3,
                     showticklabels=False)
    _fig.update_layout(height=200, plot_bgcolor='white')
    auc_fig = mo.ui.plotly(_fig)
    return (auc_fig,)


@app.cell
def _(
    TTS_eval,
    df_plotter,
    go,
    indx,
    make_subplots,
    mo,
    np,
    pf,
    pl,
    rangex_fig1,
    rangex_fig2,
    rangey_fig1,
    rangey_fig2,
    switch,
    umbralAUC,
):
    _aux_df_b = df_plotter.filter((pl.col("AUC") > umbralAUC[indx]) & (pl.col("AUC") <= umbralAUC[indx + 1]))

    _aux_df_max = _aux_df_b.filter(pl.col("AUC") == pl.col("AUC").max())
    _aux_df_min = _aux_df_b.filter(pl.col("AUC") == pl.col("AUC").min())



    col_shape = {"W":{"color":'#1f77b4', "shape":"circle"}, "P":{"color":'#1f77b4', "shape":"square"}, "F":{"color":'#1f77b4', "shape":"cross"}}
    fig = make_subplots(rows=1, cols=2)
    for _p in pf:
        _aux_df = _aux_df_b.filter(pl.col("Product_Form") == _p)
        fig.add_trace(go.Scatter(x=_aux_df["Cu"], y=_aux_df["Ni"], 
                                 mode='markers', 
                                 marker=dict(symbol=col_shape[_p]["shape"], color=col_shape[_p]["color"],line=dict(width=0.5, color="DarkSlateGrey")), 
                                 name=f"{_p}", legendgroup=f"{_p}"),row=1, col=1)
        fig.add_trace(go.Scatter(x=_aux_df["Fluence_1E19_n_cm2"], y=_aux_df["DT41J_Celsius"], 
                                 mode='markers', 
                                 marker=dict(symbol=col_shape[_p]["shape"], 
                                             color=_aux_df["Cu"], colorscale='Sunsetdark', showscale=True, 
                                             cmin = rangex_fig1[0], cmax = rangex_fig1[1],
                                             line=dict(width=0.5, color="DarkSlateGrey")), 
                                 name=f"{_p}",legendgroup=f"{_p}", showlegend=False), row=1, col=2)
    if _aux_df_max.shape[0] !=0 and switch.value:
        _Fl = np.linspace(1e22, 5*1e24, 500)
        tts_max = TTS_eval(pf=_aux_df_max["Product_Form"].to_numpy()[0],cu=_aux_df_max["Cu"].to_numpy()[0], ni=_aux_df_max["Ni"].to_numpy()[0], mn=_aux_df_max["Mn"].to_numpy()[0], p=_aux_df_max["P"].to_numpy()[0], t=_aux_df_max["Temperature_Celsius"].to_numpy()[0], fl=_Fl)

        tts_min = TTS_eval(pf=_aux_df_min["Product_Form"].to_numpy()[0],cu=_aux_df_min["Cu"].to_numpy()[0], ni=_aux_df_min["Ni"].to_numpy()[0], mn=_aux_df_min["Mn"].to_numpy()[0], p=_aux_df_min["P"].to_numpy()[0], t=_aux_df_min["Temperature_Celsius"].to_numpy()[0], fl=_Fl)
        fig.add_trace(go.Scatter(x=_Fl*1e-23, y=tts_max,line=dict(color="indigo"), fill="tonexty",showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=_Fl*1e-23, y=tts_min, line=dict(color="indigo"),fill="tonexty",showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Cu", row=1, col=1, range=rangex_fig1)
    fig.update_yaxes(title_text="Ni", row=1, col=1, range=rangey_fig1)
    fig.update_xaxes(title_text="Fluence (1E19 n/cm2)", row=1, col=2, range=rangex_fig2)
    fig.update_yaxes(title_text="DT41J (Celsius)", row=1, col=2, range=rangey_fig2)
    fig.update_layout(width=1200, height=600)
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="reversed",
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=12,
                color="black"
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        )
    )
    mo_fig = mo.ui.plotly(fig)
    return (mo_fig,)


@app.cell
def _(auc_fig, mo, mo_fig):
    figs_ui = mo.vstack([auc_fig, mo_fig],gap=0, align="center")
    return (figs_ui,)


@app.cell
def _(figs_ui, mo, vstack_cntrols):
    mo.hstack([vstack_cntrols, figs_ui], align="center", gap=0)
    return


@app.cell
def _(mo):
    mo.md(r"""----------""")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np

    from scipy.integrate import simpson
    return mo, np, pl


@app.cell
def _():
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots


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


if __name__ == "__main__":
    app.run()
