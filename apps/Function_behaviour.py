import marimo

__generated_with = "0.14.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go
    import numpy as np
    return go, mo, np


@app.cell
def _(mo):
    drop_func = mo.ui.dropdown(options={"Exponential":0, "Logarithmic":1}, label="Select f(x)", value="Exponential")
    drop_func
    return (drop_func,)


@app.cell
def _(np):
    def func_exp(x, a, b, c, alpha):
        return (a*x**alpha)+(b*(1-np.exp(c*x)))

    def func_log(x, a, b, c, alpha):
        return (a*x**alpha)+(b*np.log(x))+c

    return func_exp, func_log


@app.cell
def _(drop_func, func_exp, func_log, mo):
    if drop_func.value == 0:

        func = func_exp
        md_func = mo.md(r"## $\Delta T_{41J}= a\cdot \phi ^{\alpha}+b(1-e^{c\cdot \phi})$")
    else:

        func = func_log
        md_func = mo.md(r"### $\Delta T_{41J}= a\cdot \phi ^{\alpha}+b\cdot log(\phi + 1)+c$")
    return func, md_func


@app.cell
def _(mo):
    a=mo.ui.slider(0, 100, value=43.78,step=0.01,full_width=True, label="a")
    b=mo.ui.slider(-100, 100, value=-7.533 ,step=0.01,full_width=True,label="b")
    c=mo.ui.slider(0, 2, value=0.192,step=0.001,full_width=True,label="c")
    alpha=mo.ui.slider(0, 2,value=0.311,step=0.001,full_width=True, label="alpha")
    return a, alpha, b, c


@app.cell
def _(a, alpha, b, c, md_func, mo):
    mo.hstack([mo.vstack([a,b,c,alpha], gap=0), md_func], gap=0,align="center")
    return


@app.cell
def _(a, alpha, b, c, func, go, mo, np):
    x = np.linspace(0.01, 12, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=func(x, a.value, b.value, c.value, alpha.value), mode='lines', name='f(x)'))
    fig.update_yaxes(range=[-10, 260])
    fig.update_layout(width=600, height=600)
    mo.ui.plotly(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
