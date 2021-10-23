#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
import urllib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from flask import request
from IPython.display import display, HTML


# In[12]:

# df_url = "https://drive.google.com/uc?export=download&id=11o7ffsWxu09zm75DtojsHPHRkZWl6fjl"
# urllib.request.urlretrieve(df_url, "df_combined.sav")
df_combined = pickle.load(open('df_combined.pkl', 'rb'))
benchmark_df = pickle.load(open('benchmark.pkl', 'rb'))
benchmark_weights = pickle.load(open('benchmark_weights.pkl', 'rb'))
df_portfolio_weights_after_rebalancing = pickle.load(open('df_portfolio_weights_after_rebalancing.pkl', 'rb'))
sector_tickers = pickle.load(open('sector_tickers.pkl', 'rb'))


# In[21]:


def sector_benchmark(tickers):
    sector_benchmark = benchmark_weights[benchmark_weights.index.get_level_values("ticker").isin(tickers)]

    forward_1m_change = df_combined['close'].unstack().pct_change(1).shift(-1)
    sector_benchmark_df = pd.DataFrame(index=sector_benchmark.index)
    sector_benchmark_df = sector_benchmark_df.join(forward_1m_change.stack().to_frame('forward_rets'), how='left')
    sector_benchmark_df = sector_benchmark_df.loc[:benchmark_df.index.get_level_values(0).unique()[-2]]
    return sector_benchmark_df

# In[25]:
macro_list = ['YC/USA3M - Rate', 'YC/USA2Y - Rate', 'YC/USA5Y - Rate', 'YC/USA10Y - Rate', 'vix', 'gold']

sector_names = ['Financial Services', 'Consumer Cyclical', 'Utilities', 'Healthcare', 
                'Basic Materials', 'Consumer Defensive', 'Technology', 'Real Estate', 'Energy', 
                'Industrials', 'Communication Services']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
navbarcurrentpage = {
    'text-decoration' : 'underline',
    'text-decoration-color' : '251, 0, 0',
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)'
    }
    
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# Create server variable with Flask server object for use with gunicorn
server = app.server

def get_header(): 
    header = html.Div([
    html.Div([], className = 'col-2'),
    html.Div([
                html.H1(children='Portfolio Performance Dashboard',
                        style = {'textAlign' : 'center'}
                )],
                className='col-8',
                style = {'padding-top' : '1%'}
            )],
    className = 'row',
    style = {'height' : '4%'}
    )
    return header
    
def get_navbar(p = 'portfolio'):
    navbar_portfolio = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Portfolio Overview',
                        style = navbarcurrentpage),
                href='/apps/portfolio-overview'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Macro Trends'),
                href='/apps/macro'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')],
        
        className = 'row',
        style = {'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
        )
        
    navbar_macro = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Portfolio Overview'),
                href='/apps/portfolio-overview'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Macro Trends',
                        style = navbarcurrentpage),
                href='/apps/macro'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')],
        
        className = 'row',
        style = {'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
        )
    
    if p == 'portfolio':
        return navbar_portfolio
    elif p == 'macro':
        return navbar_macro

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

portfolio = html.Div([
    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar 
    get_navbar('portfolio'),

    dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x} 
                 for x in sector_names],
        value=sector_names[0]
    ),
    dcc.Graph(id = 'performance_plot'),
    dcc.Graph(id = 'tickers_plot')
]) 

macro = html.Div([
    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar 
    get_navbar('macro'),

    dcc.Dropdown(
        id="dropdown2",
        options=[{"label": x, "value": x} 
                 for x in macro_list],
        value=macro_list[:2],
        multi=True
    ),
    dcc.Graph(id='macro')
])   
    
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/portfolio-overview':
         return portfolio
    elif pathname == '/apps/macro':
         return macro
    else:
        return portfolio # This is the "home page"

@app.callback(
    Output("macro", "figure"), 
    [Input("dropdown2", "value")])
def update_bar_chart(dims):
    fig = px.scatter_matrix(
        df_combined, dimensions=dims)
    return fig
    
@app.callback(Output(component_id='performance_plot', component_property= 'figure'),
              [Input(component_id='dropdown', component_property= 'value')])
def sector_performance_graph(sector_name):
    print(sector_name)
    tickers = sector_tickers[sector_name]
    sector_portfolio = df_portfolio_weights_after_rebalancing.copy()
    sector_portfolio = sector_portfolio[sector_portfolio.index.get_level_values("ticker").isin(tickers)]
    sector_portfolio = sector_portfolio.join(benchmark_df, how='left')
    sector_portfolio = sector_portfolio[sector_portfolio['weights'] != 0]

    wealth = sector_portfolio["forward_rets"].groupby("date").mean()
    wealth = wealth.shift(1)
    start_date = wealth.index[0]
    wealth.loc[start_date] = 0

    compare_df = pd.DataFrame(index=wealth.index)
    compare_df[f'{sector_name} portfolio'] = (wealth+1).cumprod()
    
    benchmark_wealth = sector_benchmark(tickers)
    benchmark_wealth = benchmark_wealth["forward_rets"].groupby("date").mean()
    benchmark_wealth.loc[start_date] = 0
    compare_df[f'{sector_name} benchmark'] = (benchmark_wealth+1).cumprod()

    fig = px.line(compare_df, title = f'{sector_name} portfolio performance against benchmark')
    return fig
    
@app.callback(Output(component_id='tickers_plot', component_property= 'figure'),
              [Input(component_id='dropdown', component_property= 'value')])
def generate_sector_tickers_graph(sector_name):
    tickers = sector_tickers[sector_name]
    sector_portfolio = df_portfolio_weights_after_rebalancing.copy()
    sector_portfolio = sector_portfolio[sector_portfolio.index.get_level_values("ticker").isin(tickers)]
    sector_portfolio = sector_portfolio.join(benchmark_df, how='left')
    sector_portfolio = sector_portfolio[sector_portfolio['weights'] != 0]

    ticker_count = sector_portfolio["forward_rets"].groupby("date").count()
    ticker_count = ticker_count.shift(1)
    start_date = ticker_count.index[0]

    compare_df = pd.DataFrame(index=ticker_count.index)
    compare_df[f'{sector_name} portfolio'] = ticker_count+1
    
    benchmark_ticker_count = sector_benchmark(tickers)
    benchmark_ticker_count = benchmark_ticker_count["forward_rets"].groupby("date").count()
    compare_df[f'{sector_name} benchmark'] = benchmark_ticker_count+1

    fig = px.line(compare_df, title = f'{sector_name} portfolio ticker count against benchmark')
    return fig
    
if __name__ == '__main__': 
    app.run_server()
