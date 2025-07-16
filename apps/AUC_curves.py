import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _(mo, np, pd):

    Cu2 = np.arange(0, 0.5, 0.033) 
    Ni2 = np.arange(0, 1.8, 0.133) 


    XCu, YNi = np.meshgrid(Cu2, Ni2)



    Fl = np.array([i * 1e18 for i in range(1, int(1e20 / 1e18) + 1)])*1e4  # n/m2

    # -----------------------------
    file = mo.notebook_location() / "public" / "df_plotter.csv"
    df_plotter = pd.read_csv(str(file))
    df_plotter.reset_index(drop=True, inplace=True)
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
            ZT[nii, cui] = simpson(TTS_eval(pf, Cu2[cui], Ni2[nii], Mn, P, T, Fl), Fl*1e-23)

    AUC = simpson(y = np.array([TTS_eval(pf, cu=df_plotter["Cu"].to_numpy(), ni=df_plotter["Ni"].to_numpy(), mn=Mn, p=P, t=T, fl=Fli) for Fli in Fl]), x=Fl*1e-23, axis=0)
    fig = go.Figure()
    _max = int(np.max(ZT))
    fig.add_trace(go.Contour(z = ZT,x=Cu2, y=Ni2,  contours_coloring='lines',colorscale="Sunsetdark", contours=dict(showlabels=True, start=0, end=_max, size=200),showscale=False, hoverinfo="none"))
    fig.add_trace(go.Scatter(x=df_plotter["Cu"], y=df_plotter["Ni"], mode="markers", opacity=0.3, 
                                 marker=dict(color="grey",colorscale='Sunsetdark', # <--- Define la paleta de color para AUC
                colorbar=dict(
                    title='AUC', # Título de la barra de color
                    # Puedes ajustar los ticks si lo necesitas
                    # tickvals=[0, 100, 200, 300, 400]
                ), showscale=True,line=dict(width=0.3, color='DarkSlateGrey'),cmin=np.min(ZT),cmax=np.max(ZT)),customdata=AUC,hoverinfo="none", showlegend=False))
    aux_df, dist = df_and_dist(df_plotter, P, Mn, T, pf)
    if aux_df.shape[0] != 0:


        fig.add_trace(go.Scatter(x=aux_df["Cu"], y=aux_df["Ni"], mode="markers",  
                                     marker=dict(color=AUC[aux_df.index],colorscale='Sunsetdark', opacity=dist,
                    colorbar=dict(
                        title='AUC', # Título de la barra de color
                        # Puedes ajustar los ticks si lo necesitas
                        # tickvals=[0, 100, 200, 300, 400]
                    ), showscale=True,
                                                 line=dict(width=0.3, color='DarkSlateGrey'), cmin=np.min(ZT),cmax=np.max(ZT)),
                                     customdata=AUC[aux_df.index],
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
    commands = mo.vstack([mo.hstack([Tslide, mo.md(f"{T=:.2f}$\pm$ {df_plotter['Temperature_Celsius'].std()/2:.2f}")], justify="center"), 
               mo.hstack([Mnslide, mo.md(f"{Mn=:.3f}$\pm$ {df_plotter['Mn'].std()/2:.3f}")], justify="center"), 
               mo.hstack([Pslide, mo.md(f"{P=:.3f}$\pm$ {df_plotter['P'].std()/2:.3f}")], justify="center"), 
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
    import pandas as pd
    from scipy.integrate import simpson
    from itertools import product

    return np, pd, simpson


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
def _(np):
    def df_and_dist(df, P, Mn, T, pf):
        stdP = df["P"].std()/2
        stdMn = df["Mn"].std()/2
        stdT = df["Temperature_Celsius"].std()/2

        aux_df = df[(df["Product_Form"] == pf) & ((df["P"] > P-stdP) & ((df["P"] < P+stdP))) & 
            ((df["Mn"] > Mn-stdMn) & ((df["Mn"] < Mn+stdMn))) & 
            ((df["Temperature_Celsius"] > T-stdT) & ((df["Temperature_Celsius"] < T+stdT)))]
        if aux_df.shape[0]==0:
            return aux_df, [0]
        px = aux_df["P"].to_numpy()
        mny = aux_df["Mn"].to_numpy()
        Tz = aux_df["Temperature_Celsius"].to_numpy()

        dpx = (px - P)**2
        dmny = (mny - Mn)**2
        dTz = (Tz - T)**2

        dist = np.sqrt(dpx + dmny + dTz)
        min_orig = np.min(dist)
        max_orig = np.max(dist)
        if max_orig == min_orig:
            dist=np.ones(len(dist))
        else:
            normalized_to_0_1 = (dist - min_orig) / (max_orig - min_orig)
            dist = 0.5 + (normalized_to_0_1 * (1 - 0.5))
        return aux_df, dist
    return (df_and_dist,)


if __name__ == "__main__":
    app.run()
