#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px 
from wordcloud import WordCloud


# In[2]:


import dash
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash.dependencies as dd


# In[3]:


import base64
from io import BytesIO


# # GET DATA

# In[5]:


all_granules_titles_df = pd.read_csv('all_granules_titles.csv', header=0)
grouped_pages_granules_df = pd.read_csv('grouped_pages_granules.csv', header=0)
H_S_df = pd.read_csv('H_S_df.csv', header=0)
H_df = pd.read_csv('H_df.csv', header=0)
S_df = pd.read_csv('S_df.csv', header=0)


# # ViSUALIZATION GRAPHS

# In[6]:


date_ls = grouped_pages_granules_df.dateIssued.to_list()[::-1]


# In[7]:


titles_table = all_granules_titles_df[['dateIssued','title','granuleClass']]
titles_table['title'] = [i.title() for i in titles_table['title']]
titles_table.columns = ['dateIssued','Title','Granule Class']


# In[15]:


def get_wordcloud(wc_records,date):
    wc_records_date = wc_records[wc_records['dateIssued']==date]
    wc = WordCloud(background_color='white')
    wc.generate_from_frequencies(dict(zip(wc_records_date[wc_records_date.columns[0]].to_list(),wc_records_date[wc_records_date.columns[1]].to_list())))
    wc_img = wc.to_image()
    with BytesIO() as buffer:
        wc_img.save(buffer, 'png')
        image = base64.b64encode(buffer.getvalue()).decode()
        wordcloud_src = "data:image/png;base64," + image
    return wordcloud_src


# In[16]:


wordcloud_src = get_wordcloud(H_S_df,date_ls[0])




# # DEPLOYMENT

# In[17]:


app = dash.Dash(external_stylesheets=[dbc.themes.SKETCHY])
server = app.server


# In[24]:


app.layout = html.Div([
    html.Div([
        html.H1('Congressional Records Daily Briefing', style={'fontSize': '36px','margin-top': '20px', 'margin-bottom': '20px', }),
        html.Div(children=[
            # html.H3('More details information can refer to daily digest in '),
            html.P(
                'Congressional record daily briefing dashboard shows emerging political potentials by analyzing titles of the House and Senate documents requested from GovInfo API. People can also check out the daily digests released by the congress record to know additional information.',
                style={'fontSize': '24px'}),
            html.A(href="https://www.congress.gov/congressional-record",
                   children=[html.Button('Latest Daily Digest', className="btn btn-warning")],
                   style={'float': 'right', 'margin-right': '50px', 'margin-top': '0px', 'margin-bottom': '20px',
                          'display': 'inline-block'}),
        ], style={'float': 'left', 'display': 'inline-block'}),
    ], style={#'width': '95%',
              'margin-left': '20px',
              'margin-right': '20px',
              'display': 'inline-block',
              }),

    html.Div([
        html.H2("What's discussed by the House and Senate?", className='card-header'),
        html.Div([
            html.Div([
                html.Label("Date", style={'fontSize': '18px'}, className='form-label mt-4'),
                dcc.Dropdown(
                    id="date-filter1",
                    options=[
                        {"label": date, "value": date}
                        for date in date_ls
                    ],
                    value=date_ls[0],
                    clearable=False,
                    # className="form-select",
                ),
            ], className='form-group', style={'margin-bottom': '20px', }),
            html.Div([
                html.Div([
                    html.P('Daily Word Cloud'),
                    html.Img(id='wordcloud-img', src=wordcloud_src, style={'width': '100%'}),
                    # "data:image/png;base64,"
                ], style={'width': '40%', 'textAlign': 'center', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                    html.P('Daily Document Titles'),
                    dash_table.DataTable(
                        id='titles_table',
                        data=titles_table[['Title', 'Granule Class']].to_dict('records'),
                        columns=[{"name": 'Title', "id": 'Title'}, {"name": 'Granule Class', "id": 'Granule Class'}],
                        style_cell={
                            'font_size': '12px',
                            'font_family': 'sans-serif',
                            'fontWeight': 'normal',
                            'textAlign': 'left',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'maxWidth': 5
                        },
                        style_header={
                            'font_size': '14px',
                            'font_family': 'sans-serif',
                            'textAlign': 'center',
                            'fontWeight': 'bold',
                        },
                        style_cell_conditional=[
                            {'if': {'column_id': 'Granule Class'}, 'width': '25%'},
                        ],
                        page_size=7,
                        filter_action="native",
                        filter_options={"case": "insensitive"},
                        page_action="native",
                        tooltip_data=[
                            {
                                column: {'value': str(value), 'type': 'markdown'}
                                for column, value in row.items()
                            } for row in titles_table[['Title', 'Granule Class']].to_dict('records')
                        ],
                        tooltip_duration=None,
                    ),
                ], style={'width': '55%', 'textAlign': 'center', 'float': 'right', 'display': 'inline-block'}),
            ], )
        ], className='card-body'
        ),
    ], className="card border-primary mb-3",
        style={'width': '93%',
               'float': 'left',
               'margin-left': '50px',
               'margin-right': '50px',
               'display': 'inline-block',
               }),

    html.Div([
        html.H2('Top words in Senate and House', className='card-header'),
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Class", style={'fontSize': '18px'}, className='form-label mt-4'),
                    dcc.Dropdown(
                        id="gran-class-filter",
                        options=[
                            {"label": gran_class, "value": gran_class}
                            for gran_class in ['Senate', 'House']
                        ],
                        value="House",
                        clearable=False,
                        # className="form-select",
                    ),
                ], className='form-group', style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Date", style={'fontSize': '18px'}, className='form-label mt-4'),
                    dcc.Dropdown(
                        id="date-filter2",
                        options=[
                            {"label": date, "value": date}
                            for date in date_ls
                        ],
                        value=date_ls[0],
                        clearable=False,
                        # className="form-select",
                    ),
                ], className='form-group', style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            html.Div([
                dcc.Graph(
                    id="word-freq-treemap-chart", config={"displayModeBar": False},  # figure = treemap_fig,
                ),
                # dash_table.DataTable(topic_docs_table.to_dict('records'),
                #
                # ),
            ]),
        ], className='card-body'
        ),
    ], className="card border-primary mb-3",
        style={'width': '93%',
               'float': 'left',
               'margin-left': '50px',
               'display': 'inline-block',
               }),

    html.Div([
        html.H2("How active the Congress has been?", className='card-header'),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label("Index type", style={'fontSize': '18px'}, className='form-label mt-4'),
                        dcc.Dropdown(
                            id="activity-type-filter",
                            options=[
                                {"label": activity_type, "value": activity_type}
                                for activity_type in ['Record Granules', 'Record Pages']
                            ],
                            value="Record Granules",
                            clearable=False,
                            # className="form-select",
                        ),
                    ], className="form-group"),
                dcc.Graph(
                    id="activity-chart", config={"displayModeBar": False}
                ),
            ], className='card-body'
        ),
    ], className="card border-primary mb-3",
        style={'width': '93%',
               'float': 'left',
               'margin-left': '50px',
               'display': 'inline-block',
               }),  # style={'width': '40%','display': 'inline-block'}

],
)


# In[25]:


@app.callback([Output('wordcloud-img','src'),Output('titles_table','data')],
              Input("date-filter1", "value"))
def update_table(date):
    df_table = titles_table[titles_table['dateIssued']==date]
    wc_records_date = H_S_df[H_S_df['dateIssued']==date]
    
    wc = WordCloud(background_color='white')
    wc.generate_from_frequencies(dict(zip(wc_records_date[wc_records_date.columns[0]].to_list(),wc_records_date[wc_records_date.columns[1]].to_list())))
    wc_img = wc.to_image()
    with BytesIO() as buffer:
        wc_img.save(buffer, 'png')
        image = base64.b64encode(buffer.getvalue()).decode()
        wordcloud_src = "data:image/png;base64," + image
               
    return wordcloud_src,df_table.to_dict('records')


# In[26]:


@app.callback(Output("word-freq-treemap-chart",'figure'),
              [Input("date-filter2", "value"),Input("gran-class-filter","value")])
def update_treemap(date,gran_class):
    if gran_class == 'Senate':
        wc_records = S_df
    else:
        wc_records = H_df
        
    wc_records_date = wc_records[wc_records['dateIssued']==date]
    wc_records_date.columns = ['Word','Count','Count_pct','Date Issued']
    fig = px.treemap(wc_records_date,
                     path = ['Word'],
                     values = 'Count',
                    color_discrete_sequence=px.colors.qualitative.T10)
    return fig


# In[27]:


@app.callback(Output("activity-chart", "figure"), Input("activity-type-filter", "value"))
def update_charts(activity_type):
    if activity_type == 'Record Granules':
        activity_chart = {
            "data": [
                {
                    "x": grouped_pages_granules_df['dateIssued'],
                    "y": grouped_pages_granules_df['granulesCount'],
                    "type": "bar",
                },
            ],
            "layout": {"title": 'Number of Record Granules per Day',
                       "colorway": ['#20B2AA']},
        }
    else:
        activity_chart = {
            "data": [
                {
                    "x": grouped_pages_granules_df['dateIssued'],
                    "y": grouped_pages_granules_df['pages'],
                    "type": "bar",
                },
            ],
            "layout": {"title": 'Number of Record Pages per Day',
                       "colorway": ['#FFA07A']},
        }

    return activity_chart


# In[ ]:


if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:





# In[ ]:




