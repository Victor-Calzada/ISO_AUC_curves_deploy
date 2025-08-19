import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _(mo, pl):
    file = mo.notebook_location() / "public" / "df_plotter.csv"
    file2 = mo.notebook_location() / "public" / "df_plotter_astm_auc.csv"


    df_plotter = pl.read_csv(str(file))
    df_plotter_auc = pl.read_csv(str(file2))

    d_c = pl.DataFrame({"Product_Form": ["F", "F", "F"],
     "Cu":[0.14,0.14,0.14], "Ni":[0.11,0.11,0.11], "Mn":[1.3, 1.3, 1.3], "P":[0.017,0.017,0.017], 
     "Temperature_Celsius":[279.7778, 279.7778, 279.7778], "Fluence_1E19_n_cm2":[0.609,1.310, 3.960],
     "DT41J_Celsius":[19.555556, 25.333333, 68.333333]})


    cols = ["Fluence_1E19_n_cm2"]
    objective= "DT41J_Celsius"
    return d_c, df_plotter, df_plotter_auc


@app.cell
def _(mo):
    k_n = mo.ui.slider(start=1, stop=50, label="K neighbors")
    return (k_n,)


@app.cell
def _(k_n):
    k_n
    return


@app.cell
def _(d_c, df_plotter, df_plotter_auc, k_explore, k_n, mo, plt_conf_k):
    loss_k, coef = k_explore(d_c, df_plotter_auc, df_plotter)
    fig = plt_conf_k(d_c, df_plotter, df_plotter_auc, k_n.value, coef, loss_k)
    mo_fig = mo.ui.plotly(fig)
    return loss_k, mo_fig


@app.cell
def _(k_n, loss_k, loss_k_plot, mo):
    _fig_k = loss_k_plot(loss_k, k_n.value)
    mo_fig_k = mo.ui.plotly(_fig_k)
    return (mo_fig_k,)


@app.cell
def _(d_c, df_plotter, df_plotter_auc, k_n, mo, plot_cu_ni):
    _fig = plot_cu_ni(d_c, df_plotter, df_plotter_auc, k_n.value)
    fig_cu_ni = mo.ui.plotly(_fig)
    return (fig_cu_ni,)


@app.cell
def _(fig_cu_ni, mo, mo_fig, mo_fig_k):
    tab = mo.ui.tabs({
        "K neighbors": mo_fig_k,
        "Cu Ni": fig_cu_ni
    })
    mo.hstack([tab, mo_fig], gap=0, widths=[1,1])
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    import plotly.graph_objects as go
    return go, mo, np, pl


@app.cell
def _():
    from scipy.integrate import simpson
    from scipy.optimize import curve_fit
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import mean_squared_error
    return curve_fit, mean_squared_error, simpson


@app.cell
def _(np):
    # Ajuste
    def func(x, a, b, c, alpha):
        return (a*x**alpha)+(b*(1-np.exp(c*x)))


    def jacobian(x, popt, func, epsilon=1e-7):
        jac = np.zeros((len(x), len(popt)))
        for i in range(len(popt)):
            p_plus = np.array(popt)
            p_minus = np.array(popt)
            p_plus[i] += epsilon
            p_minus[i] -= epsilon
            jac[:, i] = (func(x, *p_plus) - func(x, *p_minus)) / (2 * epsilon)
        return jac
    return (func,)


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

    def TTS_eval_sd(pf, TTS):
        if pf == "F":
            C=6.972
            D=0.199
        elif (pf == 'P') or (pf == 'SRM'):
            C=6.593
            D=0.163
        elif pf == 'W':
            C=7.681
            D=0.181

        return C * TTS**D
    return (TTS_eval,)


@app.cell
def _(TTS_eval, np, pl, simpson):
    def calculate_AUC_astm(df: pl.DataFrame) -> pl.DataFrame:
        # 1. Preparar el array Fl, que es independiente del DataFrame.
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


@app.cell
def _(
    calculate_AUC_astm,
    curve_fit,
    d_c,
    func,
    get_k_nearest_auc_neighbors_from_auc,
    mean_squared_error,
    np,
    select_mat_temp_conf,
):
    def k_explore(df, df_plotter_auc, df_plotter, cols='Fluence_1E19_n_cm2',objective='DT41J_Celsius'):
        k_loss_mse=[]
        d_c_auc = calculate_AUC_astm(d_c)
        auc = d_c_auc["AUC"].to_numpy()[0]
        X = df[cols].to_numpy()
        y = df[objective].to_numpy()
        loss_mse=[]
        coef = []
        for k in range(1, 51):
            aux_df_k = select_mat_temp_conf(df_plotter,get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))
            X_k = aux_df_k[cols].to_numpy()
            y_k = aux_df_k[objective].to_numpy()

            X_train = np.concatenate([X, X_k])
            y_train = np.concatenate([y, y_k])
            try:
                popt, pcov =curve_fit(func, X_train.flatten(), y_train, maxfev=5000)
                loss_mse.append(np.sqrt(mean_squared_error(y, func(X.flatten(), *popt))))
                coef.append((popt,pcov))
            except RuntimeError as e:
                loss_mse.append(np.nan)
                coef.append((np.nan,np.nan))
            except TypeError as t:
                loss_mse.append(np.nan)
                coef.append((np.nan,np.nan))
        return loss_mse, coef
    return (k_explore,)


@app.cell
def _(
    calcular_bandas_de_ajuste,
    calculate_AUC_astm,
    func,
    get_k_nearest_auc_neighbors_from_auc,
    go,
    np,
    pl,
    select_mat_temp_conf,
):
    def plt_conf_k(df, df_plotter, df_plotter_auc, k, coef, loss_k):
        fig = go.Figure()
        if ~np.isnan(loss_k[k-1]):
            popt, pcov = coef[k-1]
            auc = calculate_AUC_astm(df)["AUC"].to_numpy()[0]
            near_main_df = select_mat_temp_conf(df_plotter,get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))
            aux_concat_df = pl.concat([df, near_main_df])
            y, x = aux_concat_df["DT41J_Celsius"].to_numpy(),  aux_concat_df["Fluence_1E19_n_cm2"].to_numpy()
            y_fit, lower_conf, upper_conf, lower_pred, upper_pred = calcular_bandas_de_ajuste(func, x_data=x, y_data=y, popt=popt, pcov=pcov)
            indx = np.argsort(x)



            fig.add_trace(go.Scatter(x=x[indx], y=lower_pred[indx], mode="lines", line=dict(color='#C4C4C4', dash='dot'),opacity=0.7,name="pred_bound", legendgroup="pred", hoverinfo="none"))
            fig.add_trace(go.Scatter(x=x[indx], y=upper_pred[indx], mode="lines", line=dict(color='#C4C4C4', dash='dot'), fill="tonexty", opacity=0.7,name="pred_bound", showlegend=False, legendgroup="pred", hoverinfo="none"))

            fig.add_trace(go.Scatter(x=x[indx], y=lower_conf[indx], mode="lines",line=dict(color="#2ca02c", dash='dash'), name="conf_bound", legendgroup="conf", hoverinfo="none"))
            fig.add_trace(go.Scatter(x=x[indx], y=upper_conf[indx], mode="lines",line=dict(color="#2ca02c", dash='dash'),fill="tonexty", name="conf_bound", showlegend=False, legendgroup="conf", hoverinfo="none"))

            fig.add_trace(go.Scatter(x=x[indx], y=y_fit[indx], mode="lines",line=dict(color="black"), name="Prediction", hoverinfo="none"))

            fig.add_trace(go.Scatter(x=near_main_df["Fluence_1E19_n_cm2"], y=near_main_df["DT41J_Celsius"], mode="markers",marker=dict(color="#C77320",line=dict(color="#8C5217", width=1)), name="Neighbor"))
            fig.add_trace(go.Scatter(x=df["Fluence_1E19_n_cm2"], y=df["DT41J_Celsius"], mode="markers", marker=dict(size=10, color= "#C72020", line=dict(color="#651010", width=1)), name="Family"))
        fig.update_layout(width=800, height=700)
        fig.add_annotation(text=f"RMSE = {loss_k[k-1]:.2f}", showarrow=False, font=dict(size=18), xref="paper", yref="paper", x=0.03, y=0.95)
        fig.add_annotation(text=f"K = {k}", showarrow=False, font=dict(size=18), xref="paper", yref="paper", x=0.03, y=0.90)
        fig.update_xaxes(title="Fluence_1E19_n_cm2", range=[-1,9])
        fig.update_yaxes(title="DT41J_Celsius", range=[-40, 145])
        return fig
    return (plt_conf_k,)


@app.cell
def _(np):
    from scipy.stats import t
    def calcular_bandas_de_ajuste(func, x_data, y_data, popt, pcov, alpha_confidence=0.95):
        """
        Calcula las bandas de confianza y de predicción para un ajuste de curva.

        Args:
            func (callable): La función utilizada para el ajuste.
            x_data (np.ndarray): Los valores de la variable independiente.
            y_data (np.ndarray): Los valores de la variable dependiente.
            popt (np.ndarray): Los parámetros óptimos del ajuste (de curve_fit).
            pcov (np.ndarray): La matriz de covarianza del ajuste (de curve_fit).
            alpha_confidence (float): El nivel de confianza deseado (por ejemplo, 0.95 para 95%).

        Returns:
            tuple: Una tupla con (y_fit, lower_conf, upper_conf, lower_pred, upper_pred).
        """
        n = len(y_data)
        p = len(popt)
        dof = max(0, n - p)

        # Valores ajustados de la curva
        y_fit = func(x_data, *popt)

        # t-estadístico para el nivel de confianza y grados de libertad
        t_crit = t.ppf(alpha_confidence + (1 - alpha_confidence) / 2, dof)

        # Calcular el Jacobiano de la función
        J = np.zeros((n, p))
        epsilon = 1e-6
        for i in range(p):
            p_temp = popt.copy()
            p_temp[i] += epsilon
            J[:, i] = (func(x_data, *p_temp) - y_fit) / epsilon

        # Error estándar del ajuste (SE_fit)
        cov_y_fit = J @ pcov @ J.T
        SE_fit = np.sqrt(np.diag(cov_y_fit))

        # --- Bandas de confianza ---
        lower_conf = y_fit - t_crit * SE_fit
        upper_conf = y_fit + t_crit * SE_fit

        # --- Bandas de predicción ---
        sigma_res = np.sqrt(np.sum((y_data - y_fit)**2) / dof)
        SE_pred = np.sqrt(SE_fit**2 + sigma_res**2)

        lower_pred = y_fit - t_crit * SE_pred
        upper_pred = y_fit + t_crit * SE_pred

        return y_fit, lower_conf, upper_conf, lower_pred, upper_pred
    return (calcular_bandas_de_ajuste,)


@app.cell
def _(go, np):
    def loss_k_plot(loss_k, k):
        ks = list(range(1,50))
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=ks, y=loss_k, mode="lines", showlegend=False))
        if ~np.isnan(loss_k[k-1]):
        
            _fig.add_trace(go.Scatter(x=[k], y=[loss_k[k-1]], mode="markers", marker=dict(color="red", size=7), name="Selected K", showlegend=False))
        _fig.update_layout(width=600,height=600)
        _fig.update_xaxes(title="K neighbors")
        _fig.update_yaxes(title="RMSE ºC")
        return _fig
    return (loss_k_plot,)


@app.cell
def _(
    calculate_AUC_astm,
    get_k_nearest_auc_neighbors_from_auc,
    go,
    select_mat_temp_conf,
):
    def plot_cu_ni(df, df_plotter, df_plotter_auc, k):
        _fig = go.Figure()
        auc = calculate_AUC_astm(df)["AUC"].to_numpy()[0]
        near_main_df = select_mat_temp_conf(df_plotter,get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))
        _fig.add_trace(go.Scatter(x=df["Cu"], y=df["Ni"], mode="markers", marker=dict(size=10, color= "#C72020", line=dict(color="#651010", width=1)), name="Family"))
        _fig.add_trace(go.Scatter(x=near_main_df["Cu"], y=near_main_df["Ni"], mode="markers",marker=dict(color="#C77320",line=dict(color="#8C5217", width=1)), name="Neighbor"))
        _fig.update_layout(width=600,height=600)
        _fig.update_xaxes(title="Cu")
        _fig.update_yaxes(title="Ni")
        return _fig
    return (plot_cu_ni,)


if __name__ == "__main__":
    app.run()
