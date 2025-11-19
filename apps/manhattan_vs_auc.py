import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy.integrate import simpson
    return mo, np, pl, simpson


@app.cell
def _(mo, np, pl):
    file = mo.notebook_location() / "public" / "df_plotter.csv"
    # file2 = mo.notebook_location() / "public" / "df_plotter_astm_auc.csv"
    file3 = mo.notebook_location() / "public" / "df_plotter_astm_auc_var.csv"

    df_plotter_big = pl.read_csv(str(file))
    # df_plotter_auc_big = pl.read_csv(str(file2))
    df_plotter_var_big = pl.read_csv(str(file3))

    d_c_prov = pl.DataFrame({"Product_Form": ["F", "F", "F"],
                             "Cu": [0.14, 0.14, 0.14], "Ni": [0.11, 0.11, 0.11], "Mn": [1.3, 1.3, 1.3], "P": [0.017, 0.017, 0.017],
                             "Temperature_Celsius": [279.7778, 279.7778, 279.7778], "Fluence_1E19_n_cm2": [0.609, 1.310, 3.960],
                             "DT41J_Celsius": [19.555556, 25.333333, 68.333333]})

    rangex_Cu = [df_plotter_big["Cu"].min() - 0.025,
                 df_plotter_big["Cu"].max() + 0.025]
    rangey_Ni = [df_plotter_big["Ni"].min() - 0.075,
                 df_plotter_big["Ni"].max() + 0.075]

    rangex_Fl = [df_plotter_big["Fluence_1E19_n_cm2"].min(
    ) - 1, df_plotter_big["Fluence_1E19_n_cm2"].max() + 1]
    rangey_41j = [df_plotter_big["DT41J_Celsius"].min(
    ) - 10, df_plotter_big["DT41J_Celsius"].max() + 10]
    rangex_Fl_log = [np.log10(df_plotter_big["Fluence_1E19_n_cm2"].min(
    ) + 0.1), np.log10(df_plotter_big["Fluence_1E19_n_cm2"].max() + 1)]
    cols = ["Fluence_1E19_n_cm2"]
    objective = "DT41J_Celsius"

    MAX_K = 800
    return (
        MAX_K,
        d_c_prov,
        df_plotter_big,
        df_plotter_var_big,
        rangex_Cu,
        rangex_Fl,
        rangey_41j,
        rangey_Ni,
    )


@app.cell
def _(df_plotter_var_big, mo):
    drop_col_auc = mo.ui.dropdown(
        options=df_plotter_var_big.columns[7:], value=df_plotter_var_big.columns[-1])


    return (drop_col_auc,)


@app.cell
def _(drop_col_auc, np):
    Fl = np.linspace(1e22, float(drop_col_auc.value.split("AUC")[1]), 100)
    return (Fl,)


@app.cell
def _(MAX_K, mo):
    k_n = mo.ui.slider(label="Number of Neighbors (k)", start=1, stop=MAX_K, step=1, value=10, full_width=True)

    return (k_n,)


@app.cell
def _(np):
    def manhattan_distance(curve1, curve2):
        return np.sum(np.abs(curve1 - curve2))
    return (manhattan_distance,)


@app.cell
def _(df_plotter_var_big, drop_col_auc):
    use_cols = df_plotter_var_big.columns[1:7]
    use_cols.append(drop_col_auc.value)
    return (use_cols,)


@app.cell
def _(df_plotter_var_big, use_cols):
    df_plotter_auc_big = df_plotter_var_big[use_cols]
    df_plotter_auc_big = df_plotter_auc_big.rename(
        {df_plotter_auc_big.columns[-1]: "AUC"})
    return (df_plotter_auc_big,)


@app.cell
def _(mo):

    drop_case = mo.ui.dropdown(options={"Outside Case": -1,
                                        "Case 1": 6,
                                        "Case 2": 9,
                                        "Case 3": 11,
                                        "Case 4": 16,
                                        "Case 5": 20,
                                        "Case 6": 520,
                                        "Case 7": 316,
                                        "Case 8": 797,
                                        "Case 9": 777,
                                        "Case A": 453,
                                        "Case B": 132,
                                        "Case C": 563,
                                        "Case D": 683}, label="Select Case", value="Outside Case")



    return (drop_case,)


@app.cell
def _(drop_case, drop_col_auc, k_n, mo):
    mo.hstack([drop_col_auc, k_n, drop_case], justify="start", gap=2, widths=[0.2, 0.6, 0.6])
    return


@app.cell
def _(
    d_c_prov,
    df_plotter_auc_big,
    df_plotter_big,
    drop_case,
    polars_remove_existing_rows,
    select_mat_temp_conf,
):
    if drop_case.value == -1:
        d_c = d_c_prov
        df_plotter = df_plotter_big
        df_plotter_auc = df_plotter_auc_big
    else:
        d_c = select_mat_temp_conf(df_plotter_big,df_plotter_auc_big[drop_case.value])
        df_plotter = polars_remove_existing_rows(df=df_plotter_big, selected_df=d_c)
        df_plotter_auc = polars_remove_existing_rows(df=df_plotter_auc_big, selected_df=df_plotter_auc_big[drop_case.value])
    return d_c, df_plotter, df_plotter_auc


@app.cell
def _(
    Fl,
    calculate_AUC_astm,
    d_c,
    df_plotter_auc,
    get_k_nearest_auc_neighbors_from_auc,
    k_n,
):
    auc = calculate_AUC_astm(d_c, Fl=Fl)["AUC"].to_numpy()[0]
    neig_auc = get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k_n.value)
    return (neig_auc,)


@app.cell
def _(Fl, TTS_eval, d_c, df_plotter_auc, k_n, manhattan_distance, np, pl):
    curva_sel = TTS_eval(pf=d_c["Product_Form"].to_numpy()[0],
            cu=d_c["Cu"].to_numpy()[0],
            ni=d_c["Ni"].to_numpy()[0],
            mn=d_c["Mn"].to_numpy()[0],
            p=d_c["P"].to_numpy()[0],
            t=d_c["Temperature_Celsius"].to_numpy()[0],
            fl=Fl
                        )
    dist = np.zeros(df_plotter_auc.height)
    for i in range(df_plotter_auc.height):
        curva_vec = TTS_eval(pf=df_plotter_auc["Product_Form"].to_numpy()[i],
            cu=df_plotter_auc["Cu"].to_numpy()[i],
            ni=df_plotter_auc["Ni"].to_numpy()[i],
            mn=df_plotter_auc["Mn"].to_numpy()[i],
            p=df_plotter_auc["P"].to_numpy()[i],
            t=df_plotter_auc["Temperature_Celsius"].to_numpy()[i],
            fl=Fl
                        )
        dist[i] = manhattan_distance(curva_sel, curva_vec)
    df_plotter_auc_dist = df_plotter_auc.with_columns(pl.Series("Distance", dist))
    neig_dist = df_plotter_auc_dist.sort("Distance").head(k_n.value)
    
    return (neig_dist,)


@app.cell
def _(neig_auc, neig_dist):
    com_cols = neig_auc.columns
    neig_com = neig_auc.join(neig_dist, on=com_cols, how="inner")
    anti_auc = neig_auc.join(neig_com, on=com_cols, how="anti")
    anti_dist = neig_dist.join(neig_com, on=com_cols, how="anti")
    return anti_auc, anti_dist, neig_com


@app.cell
def _(anti_auc, anti_dist, d_c, neig_com, plot_all):
    plot_all(d_c, neig_com, anti_auc, anti_dist)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---------------
    """)
    return


@app.cell
def _(
    df_plotter,
    rangex_Cu,
    rangex_Fl,
    rangey_41j,
    rangey_Ni,
    select_mat_temp_conf,
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    def plot_all(d_c, neig_com, anti_auc, anti_dist):
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}],
                   [{"colspan": 2}, None]],
            subplot_titles=("","", ""), column_widths=[0.4, 0.6], row_heights=[0.8, 0.2])
        # primer plot
    
        fig.add_trace(go.Scatter(
            x=neig_com["Cu"].to_numpy(),
            y=neig_com["Ni"].to_numpy(),
            mode='markers',
            marker=dict(color='#C77320', size=8, line=dict(color="white", width=0.5)),
            name='Common Neig',
            legendgroup='Common Neighbors',
            showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=anti_auc["Cu"].to_numpy(),
            y=anti_auc["Ni"].to_numpy(),
            mode='markers',
            marker=dict(color='#3BA2FF', size=8, line=dict(color="white", width=0.5)),
            name='AUC Neig',
            legendgroup='AUC Only Neighbors',
            showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=anti_dist["Cu"].to_numpy(),
            y=anti_dist["Ni"].to_numpy(),
            mode='markers',
            marker=dict(color='#1EC767', size=8, line=dict(color="white", width=0.5)),
            name='Distance Neig',
            legendgroup='Distance Only Neighbors',
            showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=d_c["Cu"].to_numpy(),
            y=d_c["Ni"].to_numpy(),
            mode='markers',
            marker=dict(color='#C72020', size=12, line=dict(color="white", width=0.5)),
            name='Selected Case',
            legendgroup='Selected Case',
            showlegend=True
        ), row=1, col=1)
        fig.update_xaxes(title_text="Cu (%)", range=rangex_Cu, row=1, col=1)
        fig.update_yaxes(title_text="Ni (%)", range=rangey_Ni, row=1, col=1)
        # segundo plot
        neig_com_complet = select_mat_temp_conf(df_plotter, neig_com)
        anti_auc_complet = select_mat_temp_conf(df_plotter, anti_auc)
        anti_dist_complet = select_mat_temp_conf(df_plotter, anti_dist)
    
        fig.add_trace(go.Scatter(
            x=neig_com_complet["Fluence_1E19_n_cm2"].to_numpy(),
            y=neig_com_complet["DT41J_Celsius"].to_numpy(),
            mode='markers',
            marker=dict(color='#C77320', size=8, line=dict(color="white", width=0.5)),
            name='Common Neighbors',
            legendgroup='Common Neighbors',
            showlegend=False # Hide legend for this trace
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=anti_auc_complet["Fluence_1E19_n_cm2"].to_numpy(),
            y=anti_auc_complet["DT41J_Celsius"].to_numpy(),
            mode='markers',
            marker=dict(color='#3BA2FF', size=8, line=dict(color="white", width=0.5)),
            name='AUC Only Neighbors',
            legendgroup='AUC Only Neighbors',
            showlegend=False # Hide legend for this trace
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=anti_dist_complet["Fluence_1E19_n_cm2"].to_numpy(),
            y=anti_dist_complet["DT41J_Celsius"].to_numpy(),
            mode='markers',
            marker=dict(color='#1EC767', size=8, line=dict(color="white", width=0.5)),
            name='Distance Only Neighbors',
            legendgroup='Distance Only Neighbors',
            showlegend=False # Hide legend for this trace
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=d_c["Fluence_1E19_n_cm2"].to_numpy(),
            y=d_c["DT41J_Celsius"].to_numpy(),
            mode='markers',
            marker=dict(color='#C72020', size=12, line=dict(color="white", width=0.5)),
            name='Selected Case',
            legendgroup='Selected Case',
            showlegend=False # Hide legend for this trace as it's grouped with the first 'Selected Case'
        ), row=1, col=2)
        fig.update_xaxes(title_text="Fluence (1E19 n/cm²)", range=rangex_Fl, row=1, col=2)
        fig.update_yaxes(title_text="DT41J (°C)", range=rangey_41j, row=1, col=2)
    
        # tercer plot
        fig.add_trace(go.Bar(x=["Common Neighbors", "AUC Only Neighbors", "Distance Only Neighbors", "Selected Case"],
                             y=[neig_com.height, anti_auc.height, anti_dist.height, 1],marker_color=["#C77320", "#3BA2FF", "#1EC767", "#C72020"], showlegend=False),  row=2, col=1)
        fig.update_layout( height=900, title_text="Neighbors Analysis")
        return fig

    return (plot_all,)


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
def _():
    return


@app.cell
def _(TTS_eval, np, pl, simpson):
    def calculate_AUC_astm(df: pl.DataFrame, Fl = None) -> pl.DataFrame:
        # 1. Preparar el array Fl, que es independiente del DataFrame.
        if Fl is None:    
            Fl = np.array([i * 1e18 for i in range(1, int(1e20 / 1e18) + 1)]) * 1e4 # n/m^2
        dfs=[]
        for i in df.group_by("Product_Form"):
            p = i[0][0]
            aux_df = i[1]
            val = simpson(y=[TTS_eval(        pf=p,
                cu=aux_df["Cu"].to_numpy(),
                ni=aux_df["Ni"].to_numpy(),
                mn=aux_df["Mn"].to_numpy(),
                p=aux_df["P"].to_numpy(),
                t=aux_df["Temperature_Celsius"].to_numpy(),
                fl=Fli
            ) for Fli in Fl], x=Fl * 1e-23, axis=0)
            aux_df = aux_df.with_columns(pl.Series("AUC", val))
            dfs.append(aux_df)
        df_sum = pl.concat(items=dfs)
        return df_sum


    def select_mat_temp_conf(df: pl.DataFrame, df_k: pl.DataFrame) -> pl.DataFrame:
        """
        A partir de los valores de las columnas Product_Form, Cu, Ni, Mn, P y Temperature_Celsius
        del dataframe de vecinos df_k, selecciona las filas del dataframe df que coincidan con esos valores.
        Retorna un dataframe con las filas seleccionadas.
        """
        required_cols = {'Product_Form', 'Cu', 'Ni', 'Mn', 'P', 'Temperature_Celsius'}
        if not required_cols.issubset(df_k.columns):
            raise ValueError(f"El DataFrame df_k debe contener las columnas: {', '.join(required_cols)}.")

        # Convertimos ambas DataFrames a un tipo de dato que permita el join
        df_k_unique = df_k.select(list(required_cols)).unique()

        # Realizamos un inner join sobre las columnas comunes.
        # Esto es mucho más rápido que iterar sobre las filas.
        result_df = df.join(df_k_unique, on=list(required_cols), how="inner")

        return result_df

    def get_k_nearest_auc_neighbors_from_auc(df: pl.DataFrame, auc: float, k: int) -> pl.DataFrame:
        """
        Returns the k nearest neighbors to the specified 'AUC' value.

        Args:
            df (pl.DataFrame): The DataFrame that contains the 'AUC' column.
            auc (float): The AUC value for which neighbors will be searched.
            k (int): The number of nearest neighbors to return.

        Returns:
            pl.DataFrame: A DataFrame with the k nearest neighbors to the specified AUC value.
        """
        if 'AUC' not in df.columns:
            raise ValueError("The DataFrame must contain a column named 'AUC'.")

        if k < 0:
            raise ValueError("The value of k must be a non-negative integer.")

        return (
            df.sort(pl.col("AUC").sub(auc).abs())
            .head(k)
        )
    return (
        calculate_AUC_astm,
        get_k_nearest_auc_neighbors_from_auc,
        select_mat_temp_conf,
    )


if __name__ == "__main__":
    app.run()
