import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback, dcc, html

df = pd.read_csv("data.csv")

app = Dash()

plotly_figure = px.line(df[df["country"] == "Canada"], x="year", y="pop")

app.layout = html.Div(
    [
        html.H1(children="Title of Dash App", style={"textAlign": "center"}),
        dcc.Dropdown(df["country"].unique(), "Canada", id="dropdown-selection"),
        dcc.Graph(id="graph-content"),
    ]
)


@callback(Output("graph-content", "figure"), Input("dropdown-selection", "value"))
def update_graph(value):
    dff = df[df["country"] == value]
    return px.line(dff, x="year", y="pop")


app.run(debug=True)
