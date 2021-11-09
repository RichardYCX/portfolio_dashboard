#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
import urllib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dash_table import DataTable
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings

import base64

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
spx = pickle.load(open('spx.pkl', 'rb'))
features_impt_bm = pickle.load(open('features_impt_bm.pkl', 'rb'))
features_impt_port = pickle.load(open('features_impt_port.pkl', 'rb'))
features_impt_diff = pickle.load(open('features_impt_diff.pkl', 'rb'))
states = pickle.load(open('states.pkl', 'rb'))
simulation = pickle.load(open('simulation.pkl', 'rb'))
regimes = mpimg.imread('regimes.jpg')
simulation_img = mpimg.imread('simulation.png')


# In[21]:
simulation_img = base64.b64encode(open('simulation.png', 'rb').read()).decode('ascii')
regimes = base64.b64encode(open('regimes.jpg', 'rb').read()).decode('ascii')

def sector_benchmark(tickers):
    sector_benchmark = benchmark_weights[benchmark_weights.index.get_level_values("ticker").isin(tickers)]

    forward_1m_change = df_combined['close'].unstack().pct_change(1).shift(-1)
    sector_benchmark_df = pd.DataFrame(index=sector_benchmark.index)
    sector_benchmark_df = sector_benchmark_df.join(forward_1m_change.stack().to_frame('forward_rets'), how='left')
    sector_benchmark_df = sector_benchmark_df.loc[:benchmark_df.index.get_level_values(0).unique()[-2]]
    return sector_benchmark_df
    
def portfolio_tickers():
    portfolio_tickers = df_portfolio_weights_after_rebalancing.copy()
    portfolio_tickers = portfolio_tickers[portfolio_tickers['weights'] != 0].reset_index()
    portfolio_tickers = portfolio_tickers[portfolio_tickers['date'] == max(portfolio_tickers['date'])]
    portfolio_tickers.sort_values('weights', axis=0, ascending=False, inplace=True)
    portfolio_tickers.drop(columns=['date'], inplace=True)
    portfolio_tickers['weights'] = portfolio_tickers['weights'] * 100 
    portfolio_tickers.rename(columns={"weights": "position size (%)"}, inplace=True)
    return portfolio_tickers
    
def portfolio_performance_graph():
    portfolio = df_portfolio_weights_after_rebalancing.copy()
    portfolio = portfolio.join(benchmark_df, how='left')
    wealth = (portfolio['weights']*portfolio['forward_rets'].values).groupby(level=0).sum()
    wealth = wealth.shift(1) #becomes the actual return (not the fwd return)
    start_date = wealth.index[0]
    wealth.loc[start_date] = 0

    compare_df = pd.DataFrame(index=wealth.index)
    compare_df['portfolio'] = (wealth+1).cumprod()

    benchmark_wealth = (spx['Close'].pct_change().loc[wealth.index])
    benchmark_wealth.loc[start_date] = 0
    compare_df['benchmark'] = (benchmark_wealth+1).cumprod()

    tracking_error = np.sqrt(sum([val**2 for val in wealth - benchmark_wealth])) * 100
    sharpe_ratio = wealth.mean()/wealth.std() * np.sqrt(252/12)
    duration = compare_df['portfolio'].shape[0]/12
    annual_returns = ((compare_df['portfolio'].iloc[-1])**(1/duration)-1)*100
    ir = (annual_returns - ((compare_df['benchmark'].iloc[-1])**(1/duration)-1)*100)/tracking_error
    
    compare_df['portfolio'] = compare_df['portfolio'] * 100
    compare_df['benchmark'] = compare_df['benchmark'] * 100

    fig = px.line(compare_df, title=f"{duration:.2f} years backtest - Cum VA(Ann.Rtn={annual_returns:.2f}%, Sharpe={sharpe_ratio:.2f}, IR={ir:.2f}), TE={tracking_error:.2f}%",
                    labels={
                     "value": "performance (%)"
                 })
    fig.update_layout(hovermode="x unified")
    return fig

def portfolio_returns():
    portfolio = df_portfolio_weights_after_rebalancing.copy()
    portfolio = portfolio.join(benchmark_df, how='left')
    wealth = (portfolio['weights']*portfolio['forward_rets'].values).groupby(level=0).sum()
    wealth = wealth.shift(1) #becomes the actual return (not the fwd return)
    start_date = wealth.index[0]
    wealth.loc[start_date] = 0

    compare_df = pd.DataFrame(index=wealth.index)
    compare_df['portfolio'] = (wealth+1).cumprod()

    benchmark_wealth = (spx['Close'].pct_change().loc[wealth.index])
    benchmark_wealth.loc[start_date] = 0
    compare_df['benchmark'] = (benchmark_wealth+1).cumprod()

    tracking_error = np.sqrt(sum([val**2 for val in wealth - benchmark_wealth])) * 100
    sharpe_ratio = wealth.mean()/wealth.std() * np.sqrt(252/12)
    duration = compare_df['portfolio'].shape[0]/12
    annual_returns = ((compare_df['portfolio'].iloc[-1])**(1/duration)-1)*100
    return wealth, annual_returns


def vix_regimes_graph():
    fig = px.imshow(regimes)
    fig.update_traces(hovertemplate=None, hoverinfo='skip')
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def simulation_distribution_graph():
    mc_sim = ((simulation.iloc[-1,:]-simulation.iloc[0,:])/simulation.iloc[0,:]-1)
    fig = px.histogram(mc_sim, title="Average teminal values of 1000 10-year simulation",
                      labels={'value':'month'})
    fig.add_vline(x=0,line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(showlegend=False) 
    return fig

def simulation_graph():
    fig = px.imshow(simulation_img)
    fig.update_traces(hovertemplate=None, hoverinfo='skip')
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
    
# In[25]:
macro_list = ['YC/USA3M - Rate', 'YC/USA2Y - Rate', 'YC/USA5Y - Rate', 'YC/USA10Y - Rate', 'vix', 'gold']

sector_names = ['Financial Services', 'Consumer Cyclical', 'Utilities', 'Healthcare', 
                'Basic Materials', 'Consumer Defensive', 'Technology', 'Real Estate', 'Energy', 
                'Industrials', 'Communication Services']

attribution_factors = ['macro_market_risk', 'macro_volatility', 'micro_earnings', 'micro_financial_risk',
                       'micro_size', 'micro_valuation']

# external_stylesheets = 'https://codepen.io/chriddyp/pen/bWLwgP.css'
external_stylesheets = [dbc.themes.BOOTSTRAP]
navbarcurrentpage = {
    'text-decoration' : 'underline',
    'font-weight': 'bold',
    'text-decoration-color' : '100, 0, 0',
	'text-shadow': '0px 0px 1px rgb(5, 251, 252)'
    }

titleStyle = {
    'text-decoration-color' : '255, 0, 0',
    'textAlign' : 'center',
    'border-top': '2px solid black',
    'border-bottom': '2px solid black'
    }
    
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# Create server variable with Flask server object for use with gunicorn
server = app.server

def get_header(): 
    header = html.Div([
    html.Div([], className = 'col-2'),
    html.Div([
                html.H1(children='Oracle Dashboard',
                        style = {'textAlign' : 'center', 'font-weight': 'bold'}
                )],
                className='col-8',
                style = {'padding-top' : '1%'}
            )],
    className = 'row',
    style = {'height' : '4%'}
    )
    return header
    
def get_navbar(p = 'portfolio'):
    navbar_portfolio = html.Div([dbc.Row([
    

        dbc.Col([], width=1),

        dbc.Col([
            dcc.Link(
                html.H4(children = 'Portfolio Performance',
                        style = navbarcurrentpage),
                href='/apps/portfolio-performance'
                )
        ],width=3,
        className='six columns'),

        dbc.Col([
            dcc.Link(
                html.H4(children = 'Portfolio Attribution'),
                href='/apps/portfolio-attribution'
                )
        ],width=3,
        className='six columns'),
        
        dbc.Col([
            dcc.Link(
                html.H4(children = 'Stress Test'),
                href='/apps/stress-test'
                )
        ],width=2,
        className='six columns'),
        
        dbc.Col([
            dcc.Link(
                html.H4(children = 'Macro Trends'),
                href='/apps/macro'
                )
        ],width=3,
        className='six columns'),

        dbc.Col([], width=1)])],
        
        className = 'row'
        )
    
    navbar_attribution = html.Div([dbc.Row([
    

        dbc.Col([], width=1),

        dbc.Col([
            dcc.Link(
                html.H4(children = 'Portfolio Performance'),
                href='/apps/portfolio-performance'
                )
        ],width=3,
        className='six columns'),

        dbc.Col([
            dcc.Link(
                html.H4(children = 'Portfolio Attribution',
                        style = navbarcurrentpage),
                href='/apps/portfolio-attribution'
                )
        ],width=3,
        className='six columns'),
        
        dbc.Col([
            dcc.Link(
                html.H4(children = 'Stress Test'),
                href='/apps/stress-test'
                )
        ],width=2,
        className='six columns'),
        
        dbc.Col([
            dcc.Link(
                html.H4(children = 'Macro Trends'),
                href='/apps/macro'
                )
        ],width=3,
        className='six columns'),

        dbc.Col([], width=1)])],
        
        className = 'row'
        )
        
    navbar_stresstest = html.Div([dbc.Row([
    

        dbc.Col([], width=1),

        dbc.Col([
            dcc.Link(
                html.H4(children = 'Portfolio Performance'),
                href='/apps/portfolio-performance'
                )
        ],width=3,
        className='six columns'),

        dbc.Col([
            dcc.Link(
                html.H4(children = 'Portfolio Attribution'),
                href='/apps/portfolio-attribution'
                )
        ],width=3,
        className='six columns'),
        
        dbc.Col([
            dcc.Link(
                html.H4(children = 'Stress Test',
                        style = navbarcurrentpage),
                href='/apps/stress-test'
                )
        ],width=2,
        className='six columns'),
        
        dbc.Col([
            dcc.Link(
                html.H4(children = 'Macro Trends'),
                href='/apps/macro'
                )
        ],width=3,
        className='six columns'),

        dbc.Col([], width=1)])],
        
        className = 'row'
        )
        
    navbar_macro = html.Div([dbc.Row([
    

        dbc.Col([], width=1),

        dbc.Col([
            dcc.Link(
                html.H4(children = 'Portfolio Performance'),
                href='/apps/portfolio-performance'
                )
        ],width=3,
        className='six columns'),

        dbc.Col([
            dcc.Link(
                html.H4(children = 'Portfolio Attribution'),
                href='/apps/portfolio-attribution'
                )
        ],width=3,
        className='six columns'),
        
        dbc.Col([
            dcc.Link(
                html.H4(children = 'Stress Test'),
                href='/apps/stress-test'
                )
        ],width=2,
        className='six columns'),
        
        dbc.Col([
            dcc.Link(
                html.H4(children = 'Macro Trends',
                        style = navbarcurrentpage),
                href='/apps/macro'
                )
        ],width=3,
        className='six columns'),

        dbc.Col([], width=1)])],
        
        className = 'row'
        )
    
    if p == 'portfolio':
        return navbar_portfolio
    elif p == 'attribution':
        return navbar_attribution
    elif p == 'stress-test':
        return navbar_stresstest
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
    dbc.Row([
        html.H4('Portfolio Overview', style=titleStyle), 
        dbc.Col([
            html.H4('Performance', style = {'textAlign' : 'center'}),
            dcc.Graph(figure=portfolio_performance_graph())
        ], width=9),
        dbc.Col([
            html.H4('Current Tickers', style = {'textAlign' : 'center'}),
            DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in portfolio_tickers().columns],
                data=portfolio_tickers().to_dict('records'),
                page_action='none',
                style_table={'height': '400px', 'overflowY': 'auto'},
            )
        ], width=3)
    ]),
    dbc.Row([
        html.H4('Sector Overview', style=titleStyle),
        dcc.Dropdown(
            id="dropdown",
            options=[{"label": x, "value": x} 
                     for x in sector_names],
            value=sector_names[0])
    ]),
    dbc.Row([
        dbc.Col([
                html.H4('Sector performance',
                        style = {'textAlign' : 'center'}),
                dcc.Graph(id = 'performance_plot')
            ], width=6),

            dbc.Col([
                html.H4('Sector tickers',
                        style = {'textAlign' : 'center'}),
                dcc.Graph(id = 'tickers_plot')
            ], width=6),
    ], className="row")
]) 

attribution = html.Div([
    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar 
    get_navbar('attribution'),
    dbc.Row([
        html.H4('Micro/Macro Factor Attribution', style=titleStyle),
        dcc.Dropdown(
            id="dropdown3",
            options=[{"label": x, "value": x} 
                     for x in attribution_factors],
            value=attribution_factors[0]
        )
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id = 'attribution_plot')
        ], width=6),

        dbc.Col([
            dcc.Graph(id = 'attribution_diff_plot')
        ], width=6)
    ], className="row")
])   

stress_test = html.Div([
    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar 
    get_navbar('stress-test'),
    dbc.Row([  
        html.H4('Historical VIX Regimes', style=titleStyle),
        html.Img(src='data:image/jpg;base64,{}'.format(regimes), style={'height':'400px', 'width':'auto',
                                                          'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'})
    ]),
    dbc.Row([  
        html.H4('Portfolio Simulation (1000 times) with initial capital of $1000', style=titleStyle)]),
    dbc.Row([
        dbc.Col([
            html.H5('Simulation Histogram',
                        style = {'textAlign' : 'center'}),
            dcc.Graph(figure=simulation_distribution_graph())
        ], width=6),

        dbc.Col([
            html.H5('Simulation Results',
                        style = {'textAlign' : 'center'}),
            html.Img(src='data:image/png;base64,{}'.format(simulation_img), style={'height':'auto', 'width':'100%'})
        ], width=6)
    ], className="row")
]) 

macro = html.Div([
    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar 
    get_navbar('macro'),
    dbc.Row([
        html.H4('Macro Factors Scatterplot', style=titleStyle),
        dcc.Dropdown(
            id="dropdown2",
            options=[{"label": x, "value": x} 
                     for x in macro_list],
            value=macro_list[:2],
            multi=True
        ),
        dcc.Graph(id='macro')
    ])
])   
    
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/portfolio-performance':
         return portfolio
    elif pathname == '/apps/portfolio-attribution':
         return attribution
    elif pathname == '/apps/stress-test':
         return stress_test
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

@app.callback(
    Output("attribution_plot", "figure"), 
    [Input("dropdown3", "value")])
def attribution_graph(factor):
    fig = go.Figure(layout = {"title":factor})
    fig = fig.add_trace(go.Scatter(x=features_impt_port.index,y=features_impt_port[factor], name = "Portfolio"))
    fig = fig.add_trace(go.Scatter(x=features_impt_bm.index,y=features_impt_bm[factor], name = "Benchmark"))
    return fig

@app.callback(
    Output("attribution_diff_plot", "figure"), 
    [Input("dropdown3", "value")])
def attribution_diff_graph(factor):
    fig = px.line(features_impt_diff[factor], title="Difference between Portfolio and Benchmark")
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
    
    compare_df[f'{sector_name} portfolio'] = compare_df[f'{sector_name} portfolio'] * 100
    compare_df[f'{sector_name} benchmark'] = compare_df[f'{sector_name} benchmark'] * 100
    
    fig = px.line(compare_df, title = f'{sector_name} portfolio performance against benchmark',
                    labels={
                     "value": "performance (%)"
                 })
    fig.update_layout(hovermode="x unified")
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

    fig = px.line(compare_df, title = f'{sector_name} portfolio ticker count against benchmark',
                    labels={
                     "value": "ticker count"
                 })
    fig.update_layout(hovermode="x unified")
    return fig
    
if __name__ == '__main__': 
    app.run_server(debug=True)
