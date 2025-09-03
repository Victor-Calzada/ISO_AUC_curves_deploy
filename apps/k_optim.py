import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _(mo, pl):
    file = mo.notebook_location() / "public" / "df_plotter.csv"
    file2 = mo.notebook_location() / "public" / "df_plotter_astm_auc.csv"


    df_plotter_big = pl.read_csv(str(file))
    df_plotter_auc_big = pl.read_csv(str(file2))

    d_c_prov = pl.DataFrame({"Product_Form": ["F", "F", "F"],
     "Cu":[0.14,0.14,0.14], "Ni":[0.11,0.11,0.11], "Mn":[1.3, 1.3, 1.3], "P":[0.017,0.017,0.017], 
     "Temperature_Celsius":[279.7778, 279.7778, 279.7778], "Fluence_1E19_n_cm2":[0.609,1.310, 3.960],
     "DT41J_Celsius":[19.555556, 25.333333, 68.333333]})


    rangex_Cu = [df_plotter_big["Cu"].min() - 0.025, df_plotter_big["Cu"].max() + 0.025]
    rangey_Ni = [df_plotter_big["Ni"].min() - 0.075, df_plotter_big["Ni"].max() + 0.075]

    rangex_Fl = [df_plotter_big["Fluence_1E19_n_cm2"].min() - 1, df_plotter_big["Fluence_1E19_n_cm2"].max() + 1]
    rangey_41j = [df_plotter_big["DT41J_Celsius"].min() - 10, df_plotter_big["DT41J_Celsius"].max() + 10]
    cols = ["Fluence_1E19_n_cm2"]
    objective= "DT41J_Celsius"

    MAX_K=800
    return (
        MAX_K,
        d_c_prov,
        df_plotter_auc_big,
        df_plotter_big,
        rangex_Cu,
        rangex_Fl,
        rangey_41j,
        rangey_Ni,
    )


@app.cell
def _():
    # drop_case = mo.ui.dropdown(options={"Outside Case":d_c_prov, "Case 1":select_mat_temp_conf(df_plotter_big,df_plotter_auc_big[0]), "Case 2":select_mat_temp_conf(df_plotter_big,df_plotter_auc_big[1]), "Case 3":select_mat_temp_conf(df_plotter_big,df_plotter_auc_big[2]), "Case 4":select_mat_temp_conf(df_plotter_big,df_plotter_auc_big[3]), "Case 5":select_mat_temp_conf(df_plotter_big,df_plotter_auc_big[4])}, label="Select Case",value="Outside Case")
    # drop_case
    return


@app.cell
def _(mo):
    drop_case = mo.ui.dropdown(options={"Outside Case":-1, "Case 1":6, "Case 2":9, "Case 3":11, "Case 4":16, "Case 5":20}, label="Select Case",value="Outside Case")
    drop_func = mo.ui.dropdown(options={"Exponential":0, "Logarithmic":1}, label="Select f(x)", value="Exponential")


    mo.hstack([drop_case, drop_func], justify="start")
    return drop_case, drop_func


@app.cell
def _(drop_func, func_exp, func_log, jac_func_exp, jac_func_log, np):

    if drop_func.value == 0:

        func = func_exp
        jac_func = jac_func_exp
        bounds = None
    else:

        func = func_log
        jac_func = jac_func_log
        bounds = ([-np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, 1])

    return bounds, func, jac_func


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
def _():
    # d_c = drop_case.value
    # df_plotter_auc = polars_remove_existing_rows(df=df_plotter_auc_big, )
    # df_plotter = polars_remove_existing_rows(df=df_plotter_big, selected_df=drop_case.value)
    return


@app.cell
def _(MAX_K, mo):

    max_k_val = mo.ui.number(start=50, stop=MAX_K,step=5, value=200,label="Max K traveler")
    return (max_k_val,)


@app.cell
def _(max_k_val):
    K_SELECT=max_k_val.value
    return (K_SELECT,)


@app.cell
def _(K_SELECT, mo):
    k_n = mo.ui.number(start=1, stop=K_SELECT, label="K traveler")
    return (k_n,)


@app.cell
def _(dow_butt, k_n, max_k_val, mo):
    mo.hstack([k_n, max_k_val, dow_butt], gap=5, justify="start")

    return


@app.cell
def _(k_n, loss_k, loss_k_plot, mo):
    _fig_k = loss_k_plot(loss_k, k_n.value)
    mo_fig_k = mo.ui.plotly(_fig_k)
    return (mo_fig_k,)


@app.cell
def _(d_c, df_plotter, df_plotter_auc, k_n, mo, plot_cu_ni_new):
    # _fig = plot_cu_ni_altair(d_c, df_plotter, df_plotter_auc, k_n.value, rangex_Cu, rangey_Ni)
    # fig_cu_ni = mo.ui.altair_chart(_fig)
    _fig = plot_cu_ni_new(d_c, df_plotter, df_plotter_auc, k_n.value)
    fig_cu_ni = mo.ui.plotly(_fig)
    return (fig_cu_ni,)


@app.cell
def _(
    bounds,
    d_c,
    df_plotter,
    df_plotter_auc,
    drop_func,
    fig_cu_ni,
    func,
    k_explore,
    k_n,
    mo,
    np,
    plt_conf_k,
):
    loss_k, coef, obs = k_explore(d_c, df_plotter_auc, df_plotter, bounds=bounds,func=func)
    fig,popt = plt_conf_k(d_c, df_plotter, df_plotter_auc, k_n.value, coef, loss_k, paths=fig_cu_ni.value)
    mo_fig = mo.ui.plotly(fig)
    a, b, c, alpha = popt
    if np.isnan(a):
        if drop_func.value == 0:
            md_func = mo.md(r"### $\Delta T_{41J}= a\cdot \phi ^{\alpha}+b(1-e^{c\cdot \phi})$")
        else:
            md_func = mo.md(r"### $\Delta T_{41J}= a\cdot \phi ^{\alpha}+b\cdot log(\phi + 1)+c$")
    else:
        if drop_func.value == 0:
            md_func = mo.md(f"### $\Delta T_{{41J}}= {a:.3f}\\cdot \\phi ^{{{alpha:.3f}}}+{b:.3f}(1-e^{{{c:.3f}\\cdot \\phi}})$")
        else:
            md_func = mo.md(f"### $\Delta T_{{41J}}= {a:.3f}\\cdot \\phi ^{{{alpha:.3f}}}+{b:.3f}\\cdot \\log(\\phi + 1)+{c:.3f}$")
    return loss_k, md_func, mo_fig, obs


@app.cell
def _(k_n, loss_k, loss_k_plot, mo, obs):
    mo_fig_obs = mo.ui.plotly(loss_k_plot(loss_k, k_n.value, obs))
    return (mo_fig_obs,)


@app.cell
def _():
    # _x = np.linspace(0.01, 12, 200)
    # auc=[]
    # for co in coef:
    #     _popt,_=co
    #     if _popt is np.nan:
    #         continue
    #     val = func(_x, *_popt)
    #     auc.append(simpson(y=val, x=_x, axis=0))
    return


@app.cell
def _():
    # _x = np.array(range(max_k_val.value))
    # _y = np.array(auc)
    # derivada_y_central = np.gradient(_y, _x)
    return


@app.cell
def _():
    # _y = np.array(auc)
    # # _y = _y/np.max(_y)
    # _der = derivada_y_central #/np.max(derivada_y_central)
    # _dif = np.diff(_y)
    # _og_y = _y[:-1]
    # _per = _dif/_og_y*100
    # _fig = go.Figure()
    # _fig.add_trace(go.Scatter(x=list(range(max_k_val.value)), y=_y, mode="lines+markers",marker=dict(size=2.5),name="auc", showlegend=True))
    # _fig.add_trace(go.Scatter(x=list(range(1,max_k_val.value)), y=_per, mode="lines+markers",marker=dict(size=2.5),name="per", showlegend=True))
    # _fig.add_trace(go.Scatter(x=list(range(max_k_val.value)), y=_der, mode="lines+markers",marker=dict(size=2.5), showlegend=True,name="derivada"))
    # _fig.add_vline(x=k_n.value-1, line_width=2, line_dash="dash", line_color="red")
    # _fig
    return


@app.cell
def _(md_func):
    md_func
    return


@app.cell
def _(fig_cu_ni, mo, mo_fig, mo_fig_k, mo_fig_obs):
    tab = mo.ui.tabs({
        "RMSE vs k": mo_fig_k,
        "RMSE vs #Obs": mo_fig_obs,
        "Cu Ni": fig_cu_ni
    })
    mo.hstack([tab, mo_fig], justify="start",gap=0, widths=[1,1])
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
    def func_exp(x, a, b, c, alpha):
        return (a*x**alpha)+(b*(1-np.exp(c*x)))

    def jac_func_exp(x, a, b, c, alpha):
        x = np.asarray(x)
        ga = x ** alpha

        gb = 1 - np.exp(c*x)
        gc = -b*x*np.exp(c*x)
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_x = np.where(x > 0, np.log(x), 0.0)
        galpha = a*(x**alpha)*ln_x
        return np.vstack([ga, gb, gc, galpha]).T

    def func_log(x, a, b, c, alpha):
        return (a*x**alpha)+(b*np.log(x))+c

    def jac_func_log(x, a, b, c, alpha):
        """
        Jacobiano analítico de la función 'func' con respecto a los parámetros a, b, c, alpha.
        """
        J = np.zeros((len(x), 4))

        # Derivada parcial con respecto a 'a'
        J[:, 0] = x**alpha

        # Derivada parcial con respecto a 'b'
        J[:, 1] = np.log(x + 1e-12) # Se añade 1e-12 para evitar log(0)

        # Derivada parcial con respecto a 'c'
        J[:, 2] = 1.0

        # Derivada parcial con respecto a 'alpha'
        # Nota: np.log() es el logaritmo natural
        J[:, 3] = a * (x**alpha) * np.log(x + 1e-12) # Se añade 1e-12 para evitar log(0)

        return J


    return func_exp, func_log, jac_func_exp, jac_func_log


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
    K_SELECT,
    calculate_AUC_astm,
    curve_fit,
    func,
    get_k_nearest_auc_neighbors_from_auc,
    mean_squared_error,
    np,
    select_mat_temp_conf,
):
    def k_explore(df, df_plotter_auc, df_plotter, cols='Fluence_1E19_n_cm2',objective='DT41J_Celsius', func=func, bounds=None):
        k_loss_mse=[]
        d_c_auc = calculate_AUC_astm(df)
        auc = d_c_auc["AUC"].to_numpy()[0]
        X = df[cols].to_numpy()
        y = df[objective].to_numpy()
        loss_mse=[]
        coef = []
        obs = []
        if bounds is None:
            bounds = ([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])
        else:
            bounds = bounds
        for k in range(1, K_SELECT+1):
            aux_df_k = select_mat_temp_conf(df_plotter,get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))
            si_aux = aux_df_k.shape
            X_k = aux_df_k[cols].to_numpy()
            y_k = aux_df_k[objective].to_numpy()

            X_train = np.concatenate([X, X_k])
            y_train = np.concatenate([y, y_k])
            obs.append(len(y_train))
            try:
                popt, pcov =curve_fit(func, X_train.flatten(), y_train, maxfev=5000, bounds=bounds)
                loss_mse.append(np.sqrt(mean_squared_error(y, func(X.flatten(), *popt))))
                coef.append((popt,pcov))
            except RuntimeError as e:
                loss_mse.append(np.nan)
                coef.append((np.nan,np.nan))
            except TypeError as t:
                loss_mse.append(np.nan)
                coef.append((np.nan,np.nan))
        return loss_mse, coef, obs
    return (k_explore,)


@app.cell
def _(func, np):
    def pred_from_fit(coef, k, func=func):
        popt, pcov = coef[k-1]
        x = np.linspace(0.01, 12, 200)
        return func(x, *popt), x
    return


@app.cell
def _(
    calculate_AUC_astm,
    d_c,
    df_plotter,
    df_plotter_auc,
    get_k_nearest_auc_neighbors_from_auc,
    k_n,
    mo,
    pl,
    select_mat_temp_conf,
):
    def select_anomaly(df, df_plotter, df_plotter_auc, k):
        auc = calculate_AUC_astm(df)["AUC"].to_numpy()[0]
        near_main_df = select_mat_temp_conf(df_plotter,get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))
        df = df.with_columns(pl.lit(1).alias("Family"))
        near_main_df = near_main_df.with_columns(pl.lit(0).alias("Family"))
        aux_concat_df = pl.concat([df, near_main_df])
        return aux_concat_df
    download_df = select_anomaly(d_c, df_plotter, df_plotter_auc, k_n.value)
    filename = f"fam_traveler_k{str(k_n.value)}.csv"
    dow_butt = mo.download(data=download_df.write_csv(), filename=filename, label="Download fam/travelers as CSV") 
    return (dow_butt,)


@app.cell
def _(
    calculate_AUC_astm,
    func,
    get_k_nearest_auc_neighbors_from_auc,
    np,
    pl,
    select_mat_temp_conf,
):
    from delta_method import delta_method
    def test_delta_func(df, df_plotter, df_plotter_auc, k, coef, func=func):
        popt, pcov = coef[k-1]
        auc = calculate_AUC_astm(df)["AUC"].to_numpy()[0]
        near_main_df = select_mat_temp_conf(df_plotter,get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))
        aux_concat_df = pl.concat([df, near_main_df])
        y, x = aux_concat_df["DT41J_Celsius"].to_numpy(),  aux_concat_df["Fluence_1E19_n_cm2"].to_numpy()
        x_new = np.linspace(0.01, 10, 200)
        return delta_method(popt=popt, pcov=pcov, x_new=x_new, x=x, y=y, f=func, alpha=0.05)
    return


@app.cell
def _(
    calcular_bandas_de_ajuste_diego,
    calculate_AUC_astm,
    func,
    get_k_nearest_auc_neighbors_from_auc,
    go,
    jac_func,
    np,
    pl,
    rangex_Fl,
    rangey_41j,
    select_mat_temp_conf,
):

    def plt_conf_k(df, df_plotter, df_plotter_auc, k, coef, loss_k, func = func, jac_func = jac_func, paths = None):
        fig = go.Figure()
        obs = np.nan
        popt = np.nan,np.nan,np.nan,np.nan
        if ~np.isnan(loss_k[k-1]):
            popt, pcov = coef[k-1]
            # y_new_fit, x_new = pred_from_fit(coef, k, func)
            auc = calculate_AUC_astm(df)["AUC"].to_numpy()[0]
            near_main_df = select_mat_temp_conf(df_plotter,get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))
            aux_concat_df = pl.concat([df, near_main_df])
            obs = aux_concat_df.shape[0]
            y, x = aux_concat_df["DT41J_Celsius"].to_numpy(),  aux_concat_df["Fluence_1E19_n_cm2"].to_numpy()
            x_new,y_fit, lower_conf, upper_conf, lower_pred, upper_pred = calcular_bandas_de_ajuste_diego(func, x_data=x, y_data=y, popt=popt, pcov=pcov, jac_func=jac_func)
            indx = np.argsort(x_new)



            fig.add_trace(go.Scatter(x=x_new[indx], y=lower_pred[indx], mode="lines", line=dict(color='#C4C4C4', dash='dot'),opacity=0.7,name="pred_bound", legendgroup="pred", hoverinfo="none"))
            fig.add_trace(go.Scatter(x=x_new[indx], y=upper_pred[indx], mode="lines", line=dict(color='#C4C4C4', dash='dot'), fill="tonexty", opacity=0.7,name="pred_bound", showlegend=False, legendgroup="pred", hoverinfo="none"))

            fig.add_trace(go.Scatter(x=x_new[indx], y=lower_conf[indx], mode="lines",line=dict(color="#2ca02c", dash='dash'), name="conf_bound", legendgroup="conf", hoverinfo="none"))
            fig.add_trace(go.Scatter(x=x_new[indx], y=upper_conf[indx], mode="lines",line=dict(color="#2ca02c", dash='dash'),fill="tonexty", name="conf_bound", showlegend=False, legendgroup="conf", hoverinfo="none"))

            fig.add_trace(go.Scatter(x=x_new[indx], y=y_fit[indx], mode="lines",line=dict(color="black"), name="Prediction", hoverinfo="none"))

            fig.add_trace(go.Scatter(x=near_main_df["Fluence_1E19_n_cm2"], y=near_main_df["DT41J_Celsius"], mode="markers",marker=dict(color="#C77320",line=dict(color="#8C5217", width=1)), name="Travelers"))
            fig.add_trace(go.Scatter(x=df["Fluence_1E19_n_cm2"], y=df["DT41J_Celsius"], mode="markers", marker=dict(size=10, color= "#C72020", line=dict(color="#651010", width=1)), name="Family"))

            if (paths is not None) and (len(paths)>0):
                o = 1
                for path in paths:
                    if path["Family"]:
                        continue
                    else:
                        s_df = pl.DataFrame(path)
                        columns_to_cast = ["Cu",  "Ni",  "Mn",  "P",  "Temperature_Celsius"]
                        cast_expressions = [pl.col(col).cast(pl.Float64) for col in columns_to_cast]
                        s_df = s_df.with_columns(cast_expressions)
                        sel_path = select_mat_temp_conf(near_main_df, s_df)
                        sel_path = sel_path.sort("Fluence_1E19_n_cm2")
                        fig.add_trace(go.Scatter(x=sel_path["Fluence_1E19_n_cm2"], y=sel_path["DT41J_Celsius"], mode="lines+markers",
                                                 marker=dict(#color = colors_vd[o-1],
                                                             size=8),
                                                 line=dict(#color = colors_vd[o-1],
                                                           width=3), 
                                                 name=f"Traveler {o}", hoverinfo="none"))
                        o+=1
        fig.update_layout(width=800, height=700)
        fig.add_annotation(text=f"RMSE = {loss_k[k-1]:.2f} ºC", showarrow=False, font=dict(size=18), xref="paper", yref="paper", x=0.03, y=0.95)
        fig.add_annotation(text=f"K = {k}", showarrow=False, font=dict(size=18), xref="paper", yref="paper", x=0.03, y=0.90)
        fig.add_annotation(text=f"#Obs = {obs}", showarrow=False, font=dict(size=18), xref="paper", yref="paper", x=0.03, y=0.85)
        fig.update_xaxes(title="Fluence_1E19_n_cm2", range=rangex_Fl)
        fig.update_yaxes(title="DT41J_Celsius", range=rangey_41j)
        return fig, popt
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
    return (t,)


@app.cell
def _(np, t):

    def calcular_bandas_de_ajuste_new(func, x_data, y_data, popt, pcov, alpha_confidence=0.95, jac_func=None):
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
        n = 200
        x_fit = np.linspace(0.01, 12, n)

        p = len(popt)
        dof = max(0, n - p)
        # Valores ajustados de la curva
        y_fit_data = func(x_data, *popt)
        y_fit = func(x_fit, *popt)
        # t-estadístico para el nivel de confianza y grados de libertad
        t_crit = t.ppf(alpha_confidence + (1 - alpha_confidence) / 2, dof)

        if jac_func is not None:
            J = jac_func(x_fit, *popt)
        # Calcular el Jacobiano de la función
        else:
            J = np.zeros((n, p))
            epsilon = 1e-6
            for i in range(p):
                p_temp = popt.copy()
                p_temp[i] += epsilon
                J[:, i] = (func(x_fit, *p_temp) - y_fit) / epsilon

        # Error estándar del ajuste (SE_fit)
        cov_y_fit = J @ pcov @ J.T
        SE_fit = np.sqrt(np.diag(cov_y_fit))

        # --- Bandas de confianza ---
        lower_conf = y_fit - t_crit * SE_fit
        upper_conf = y_fit + t_crit * SE_fit

        # --- Bandas de predicción ---
        sigma_res = np.sqrt(np.sum((y_data - y_fit_data)**2) / dof)
        SE_pred = np.sqrt(SE_fit**2 + sigma_res**2)

        lower_pred = y_fit - t_crit * SE_pred
        upper_pred = y_fit + t_crit * SE_pred

        return y_fit, lower_conf, upper_conf, lower_pred, upper_pred
    return


@app.cell
def _(np, t):
    def calcular_bandas_de_ajuste_diego(func, x_data, y_data, popt, pcov, alpha_confidence=0.05, jac_func=None):
        n = len(x_data); p = len(popt); dof = max(1, n - p)
        y_hat = func(x_data, *popt)
        resid = y_data - y_hat
        s_sq = np.sum(resid**2) / dof

        xfit = np.linspace(np.min(x_data), 12, 500)
        yfit = func(xfit, *popt)

        J = jac_func(xfit, *popt)  

        var_mean = np.einsum("ij,jk,ik->i", J, pcov, J)  # diag(J Cov J^T)

        se_mean = np.sqrt(var_mean)

        tcrit = t.ppf(1 - alpha_confidence/2, dof)
        ci_lo = yfit - tcrit * se_mean
        ci_hi = yfit + tcrit * se_mean
        se_pred = np.sqrt(var_mean + s_sq)
        pi_lo = yfit - tcrit * se_pred
        pi_hi = yfit + tcrit * se_pred
        return xfit,yfit,ci_lo, ci_hi,pi_lo,pi_hi
    return (calcular_bandas_de_ajuste_diego,)


@app.cell
def _(np, t):
    def calcular_bandas_de_ajuste_con_jac(func, jac_func, x_data, y_data, popt, pcov, alpha_confidence=0.95):
        """
        Calcula las bandas de confianza y de predicción usando el Jacobiano analítico.
        """
        n = len(x_data)
        n_fit = 200 # Número de puntos para la curva ajustada y las bandas
        x_fit = np.linspace(0.01, 10, n_fit)

        p = len(popt)
        dof = max(0, n - p)

        # Valores ajustados de la curva
        y_fit_data = func(x_data, *popt)
        y_fit = func(x_fit, *popt)

        # t-estadístico para el nivel de confianza
        t_crit = t.ppf(alpha_confidence + (1 - alpha_confidence) / 2, dof)

        # Calcular el Jacobiano de la función para los puntos de la curva de ajuste
        J = jac_func(x_fit, *popt)

        # Error estándar del ajuste (SE_fit)
        cov_y_fit = J @ pcov @ J.T
        SE_fit = np.sqrt(np.diag(cov_y_fit))

        # --- Bandas de confianza ---
        lower_conf = y_fit - t_crit * SE_fit
        upper_conf = y_fit + t_crit * SE_fit

        # --- Bandas de predicción ---
        sigma_res = np.sqrt(np.sum((y_data - y_fit_data)**2) / dof)
        SE_pred = np.sqrt(SE_fit**2 + sigma_res**2)

        lower_pred = y_fit - t_crit * SE_pred
        upper_pred = y_fit + t_crit * SE_pred

        return y_fit, lower_conf, upper_conf, lower_pred, upper_pred
    return


@app.cell
def _(go, np):
    def loss_k_plot(loss_k, k, other=None):
        ks = list(range(1,len(loss_k)+1))
        _fig = go.Figure()
        if other is None:
            _fig.add_trace(go.Scatter(x=ks, y=loss_k, mode="lines+markers",marker=dict(size=2.5), showlegend=False))
            if ~np.isnan(loss_k[k-1]):

                _fig.add_trace(go.Scatter(x=[k], y=[loss_k[k-1]], mode="markers", 
                                          marker=dict(color="red", size=7), name="Selected K", showlegend=False))
            _fig.update_xaxes(title="K travelers")
        else:
            _fig.add_trace(go.Scatter(x=other, y=loss_k, mode="lines+markers",marker=dict(color="orange",size=2.5), showlegend=False))
            if ~np.isnan(loss_k[k-1]):
                _fig.add_trace(go.Scatter(x=[other[k-1]], y=[loss_k[k-1]], mode="markers", 
                                          marker=dict(color="red", size=7), name="Selected K", showlegend=False))
            _fig.update_xaxes(title="# Observations")
        _fig.update_layout(width=600,height=600)

        _fig.update_yaxes(title="RMSE ºC")
        return _fig
    return (loss_k_plot,)


@app.cell
def _(
    calculate_AUC_astm,
    get_k_nearest_auc_neighbors_from_auc,
    go,
    rangex_Cu,
    rangey_Ni,
    select_mat_temp_conf,
):
    def plot_cu_ni(df, df_plotter, df_plotter_auc, k):
        _fig = go.Figure()
        auc = calculate_AUC_astm(df)["AUC"].to_numpy()[0]
        near_main_df = select_mat_temp_conf(df_plotter,get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))
        _fig.add_trace(go.Scatter(x=df["Cu"], y=df["Ni"], mode="markers", marker=dict(size=10, color= "#C72020", line=dict(color="#651010", width=1)), name="Family"))
        _fig.add_trace(go.Scatter(x=near_main_df["Cu"], y=near_main_df["Ni"], mode="markers",marker=dict(color="#C77320",line=dict(color="#8C5217", width=1)), name="Travelers"))
        _fig.update_layout(width=600,height=600)
        _fig.update_xaxes(title="Cu", range=rangex_Cu)
        _fig.update_yaxes(title="Ni", range=rangey_Ni)
        return _fig
    return


@app.cell
def _(
    calculate_AUC_astm,
    get_k_nearest_auc_neighbors_from_auc,
    pl,
    rangex_Cu,
    rangey_Ni,
    select_mat_temp_conf,
):
    import plotly.express as px
    def plot_cu_ni_new(df, df_plotter, df_plotter_auc, k):

        df = calculate_AUC_astm(df)
        auc = df["AUC"].to_numpy()[0]
        near_main_df = select_mat_temp_conf(df_plotter_auc,get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))
        df = df[0,["Product_Form",  "Cu",  "Ni",  "Mn",  "P",  "Temperature_Celsius"]]
        df = df.with_columns(pl.lit(True).alias("Family"))
        df = df.with_columns(pl.lit(3.5).alias("Size"))
        near_main_df = near_main_df[:,["Product_Form",  "Cu",  "Ni",  "Mn",  "P",  "Temperature_Celsius"]]
        near_main_df = near_main_df.with_columns(pl.lit(False).alias("Family"))
        near_main_df = near_main_df.with_columns(pl.lit(0.5).alias("Size"))
        plt_df = pl.concat([df, near_main_df])


        _fig = px.scatter(plt_df, x="Cu", y="Ni", color="Family", size="Size",
                          color_discrete_map={True: "#C72020", False: "#C77320"},
                          hover_data=["Product_Form",  "Cu",  "Ni",  "Mn",  "P",  "Temperature_Celsius", "Family"])
        _fig.update_layout(showlegend=False)
        _fig.update_xaxes(title="Cu", range=rangex_Cu)
        _fig.update_yaxes(title="Ni", range=rangey_Ni)
        _fig.update_layout(width=600,height=600)
        return _fig
    return (plot_cu_ni_new,)


@app.cell
def _(
    calculate_AUC_astm,
    get_k_nearest_auc_neighbors_from_auc,
    select_mat_temp_conf,
):
    import altair as alt

    @alt.theme.register('plotly_theme', enable=True)
    def plotly_theme():
        return alt.theme.ThemeConfig({
            'config': {
                'background': '#FFFFFF',
                'axis': {
                    'gridColor': '#EEEEEE',
                    'gridWidth': 1,
                    'labelFont': 'sans-serif',
                    'titleFont': 'sans-serif'
                },
                'range': {
                    'category': [
                        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
                    ]
                },
                'title': {
                    'font': 'sans-serif'
                },
                'header': {
                    'titleFont': 'sans-serif'
                },
                'legend': {
                    'labelFont': 'sans-serif',
                    'titleFont': 'sans-serif'
                }
            }
        })

    def plot_cu_ni_altair(df, df_plotter, df_plotter_auc, k, rangex_Cu, rangey_Ni):
        """
        Recrea el gráfico de dispersión de Cu vs. Ni usando la librería Altair.

        Args:
            df (pd.DataFrame): DataFrame con los datos de "Family".
            df_plotter (pd.DataFrame): DataFrame con los datos de "Travelers".
            df_plotter_auc (pd.DataFrame): DataFrame para el cálculo de la AUC.
            k (int): Número de vecinos más cercanos a seleccionar.
            rangex_Cu (list): Rango del eje X [min, max].
            rangey_Ni (list): Rango del eje Y [min, max].

        Returns:
            altair.Chart: Objeto Altair Chart.
        """
        # El código para seleccionar los datos de 'Travelers'
        auc = calculate_AUC_astm(df)["AUC"].to_numpy()[0]
        near_main_df = select_mat_temp_conf(df_plotter, get_k_nearest_auc_neighbors_from_auc(df_plotter_auc, auc, k))

        # Definir los gráficos de dispersión
        family_chart = alt.Chart(df).mark_point(
            size=100,
            strokeWidth=1,
            stroke="#651010"
        ).encode(
            x=alt.X("Cu", title="Cu", scale=alt.Scale(domain=rangex_Cu)),
            y=alt.Y("Ni", title="Ni", scale=alt.Scale(domain=rangey_Ni)),
            color=alt.value("#C72020"),
            tooltip=["Cu", "Ni"]
        ).properties(
            title="Cu vs Ni"
        )

        travelers_chart = alt.Chart(near_main_df).mark_point(
            size=100,
            strokeWidth=1,
            stroke="#8C5217"
        ).encode(
            x="Cu",
            y="Ni",
            color=alt.value("#C77320"),
            tooltip=["Cu", "Ni"]
        )

        # Combinar los gráficos y añadir la leyenda
        combined_chart = (family_chart + travelers_chart).properties(
            width=400,
            height=400
        )





        return combined_chart
    return


@app.cell
def _(pl):
    def polars_rows_exist(df: pl.DataFrame, selected_df: pl.DataFrame) -> bool:
        """
        Check if all rows from selected_df exist in df.

        Args:
            df: The Polars DataFrame to check against.
            selected_df: The Polars DataFrame containing the rows to find.

        Returns:
            True if all rows in selected_df exist in df, False otherwise.
        """
        # Check if a left semi join of selected_df with df has the same number of rows as selected_df.
        # A semi join keeps only the rows from the left DataFrame that have a match in the right DataFrame.
        joined_df = selected_df.join(df, on=selected_df.columns, how='semi')
        return joined_df.height == selected_df.height
    return


@app.cell
def _(pl):
    def polars_remove_existing_rows(df: pl.DataFrame, selected_df: pl.DataFrame) -> pl.DataFrame:
        """
        Removes all rows from df that also exist in selected_df.

        Args:
            df: The Polars DataFrame from which to remove rows.
            selected_df: The Polars DataFrame containing the rows to be removed.

        Returns:
            A new Polars DataFrame with the rows from selected_df removed.
        """
        # A left anti join keeps only the rows from the left DataFrame (df)
        # that do NOT have a match in the right DataFrame (selected_df).
        # This is the most efficient way to achieve the desired outcome.
        return df.join(selected_df, on=df.columns, how='anti')
    return (polars_remove_existing_rows,)


@app.cell
def _(np):
    import matplotlib.cm as cm

    # Número total de colores que deseas
    num_colors = 800

    # Divide el número total entre las dos paletas
    num_colors_half = num_colors // 2

    # Genera la primera mitad de colores con la paleta 'viridis'
    colors_values_1 = np.linspace(0, 1, num_colors_half)
    viridis_colors_rgba = cm.viridis(colors_values_1)
    viridis_hex_colors = [
        '#{:02x}{:02x}{:02x}'.format(
            int(r * 255),
            int(g * 255),
            int(b * 255)
        )
        for r, g, b, a in viridis_colors_rgba
    ]

    # Genera la segunda mitad de colores con la paleta 'plasma'
    colors_values_2 = np.linspace(0, 1, num_colors - num_colors_half)
    plasma_colors_rgba = cm.plasma(colors_values_2)
    plasma_hex_colors = [
        '#{:02x}{:02x}{:02x}'.format(
            int(r * 255),
            int(g * 255),
            int(b * 255)
        )
        for r, g, b, a in plasma_colors_rgba
    ]

    # Combina ambas listas para obtener los 800 colores
    colors_vd = viridis_hex_colors + plasma_hex_colors


    return


if __name__ == "__main__":
    app.run()
