from django_plotly_dash import DjangoDash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
import pandas as pd
import nltk
from nltk.corpus import stopwords
from io import BytesIO
import base64


stopwords = nltk.corpus.stopwords.words('portuguese')
special_characters = ['!', '-', '?', '(', ')', ':', '.', ',', '@', "#",
                        '*', '&', '%', '$', "'", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        '"', '[', ']', '{', '}', '´', '`', '+']


df = pd.read_csv('dashboard/data/output.txt', delimiter='\t')
copy_df = df.copy()


#process data to get frequency of words
copy_df['Comentário'] = copy_df.apply(lambda row: nltk.word_tokenize(row['Comentário']), axis=1)
copy_df['Comentário'] = copy_df['Comentário'].apply(lambda x: [item for item in x if item not in special_characters])
copy_df['Comentário'] = copy_df['Comentário'].apply(lambda x: [item for item in x if item not in stopwords])
copy_df['Comentário'] = copy_df['Comentário'].apply(lambda x: " ".join(x))
dfreq = copy_df.Comentário.str.split(expand=True).stack().value_counts().rename_axis('words').reset_index(name='counts')

app = DjangoDash('RomaDash', external_stylesheets=[dbc.themes.BOOTSTRAP])

def plot_wordcloud(data):
    d = {a: x for a, x in data.values}
    wc = WordCloud(background_color='white', width=700, height=130)
    wc.fit_words(d)
    return wc.to_image()

fig = px.histogram(df, x="Classificação", width=700, height=250)

fig.update_layout(
    font=dict(
        color="black",
        size=12
    ),
    title=dict(
        font=dict(
            size=18
        )
    ),
    margin=dict(
        pad=5,
        b=5,
        t=20)
)
fig.update_layout({'plot_bgcolor': '#fff'})
fig.update_yaxes(title="Quantidade", title_font=dict(size=14, color='black'))
fig.update_xaxes(title="Classificação", title_font=dict(size=14, color='black'))

data_card_total = go.Figure(go.Indicator(
    value = len(df.index),
    title = "Total de Comentários",
    number={"font":{"size":60, "color":"black"}}
))

data_card_total.update_layout(
    margin=dict(pad=5, t=20, b=5),
)

data_card_total.add_layout_image(
    dict(
        source="/static/assets/icon-total.svg",
        x=0, y=0.5,
        sizex=0.5, sizey=0.5,
        xanchor="left", yanchor="middle"
    )
)

data_card_good = go.Figure(go.Indicator(
    value = df.loc[df.Sentimento == 'Bom', 'Sentimento'].count(),
    title = "Comentários bons",
    number={"font":{"size":60, "color":"#24A676"}}
))

data_card_good.update_layout(
    margin=dict(pad=5, t=20, b=5),

)

data_card_good.add_layout_image(
    dict(
        source="/static/assets/icon-good.svg",
        x=0, y=0.5,
        sizex=0.5, sizey=0.5,
        xanchor="left", yanchor="middle"
    )
)

data_card_bad = go.Figure(go.Indicator(
    value = df.loc[df.Sentimento == 'Ruim', 'Sentimento'].count(),
    title = "Comentários ruins",
    number={"font":{"size":60, "color":"#5743D9"}}
))

data_card_bad.update_layout(
    margin=dict(pad=5, t=20, b=5),
)

data_card_bad.add_layout_image(
    dict(
        source="/static/assets/icon-bad.svg",
        x=0, y=0.5,
        sizex=0.5, sizey=0.5,
        xanchor="left", yanchor="middle"
    )
)

app.layout = html.Div([
    html.Div([
        dbc.Row(
            [
                dbc.Col(dcc.Graph(
                    id='datacardtotal',
                    figure=data_card_total,
                    style={"height":"130px", "box-shadow": "5px 4px 30px rgba(0, 0, 0, 0.08)"}),
                    md=4
                ),
                dbc.Col(dcc.Graph(
                    id='datacardgood',
                    figure=data_card_good,
                    style={"height": "130px", "box-shadow": "5px 4px 30px rgba(0, 0, 0, 0.08)"}),
                    md=4
                ),
                dbc.Col(dcc.Graph(
                    id='datacardbad',
                    figure=data_card_bad,
                    style={"height": "130px", "box-shadow": "5px 4px 30px rgba(0, 0, 0, 0.08)"}),
                    md=4
                ),
            ],
            style={'margin-bottom': "30px"}
        ),
    ]),
    html.Div([
        dbc.Row([
            dbc.Col(children=[
                dbc.Row(
                    dbc.Jumbotron([
                        html.H4("Principais tópicos de reclamação"),
                        dcc.Graph(
                        id='histogram',
                        figure=fig)
                    ], style={
                        "backgroundColor": "#fff",
                        "box-shadow": "5px 4px 30px rgba(0, 0, 0, 0.08)",
                        "padding": "20px"}),
                ),
                dbc.Row(
                    dbc.Jumbotron([
                        html.Img(id="image_wc")
                    ], style={
                        "backgroundColor": "#fff",
                        "box-shadow": "5px 4px 30px rgba(0, 0, 0, 0.08)",
                        "padding": "20px",
                        "marginTop": "-1px"}),
                ),
            ],
            md=5,
            ),
            dbc.Col([
                dbc.Jumbotron([
                    dbc.Row([
                        dbc.Col(html.H4("Tabela de Classificação"), md=8),
                        dbc.Col([
                            dbc.Button("Exportar", id="btn", outline=True, color="secondary", className="mr-1 rounded-pill",
                                style={"display": "inline-flex", "justify-content": "center", "align-items": "center"}),
                            dcc.Dropdown(
                                id="filter",
                                placeholder="Filtrar classificação",
                                options=[
                                    {"label": "Não-Informativo", "value": "Não-Informativo"},
                                    {"label": "Funcionalidade", "value": "Funcionalidade"},
                                    {"label": "Confiabilidade", "value": "Confiabilidade"},
                                    {"label": "Usabilidade", "value": "Usabilidade"},
                                    {"label": "Eficiência", "value": "Eficiência"},
                                    {"label": "Portabilidade", "value": "Portabilidade"},
                                    {"label": "Segurança", "value": "Segurança"}
                                ],
                                style={"display": "inline-flex", "justify-content": "flex-end", "align-items": "end", "width": "200px", "marginBottom": "-14px"})
                        ],
                        md=4
                        )
                    ]),
                    dcc.Graph(
                        id='table'),
                    dcc.Download(
                        id="download"),
                ], style={
                    "backgroundColor": "#fff",
                    "box-shadow": "5px 4px 30px rgba(0, 0, 0, 0.08)",
                    "padding": "20px"}),
            ],
            md=7
            ),
        ]),
    ]),],
    style={
    "backgroundColor": "#f2f2f2",
    "overflowX": "hidden",
    "overflowY": "auto",
    "padding": "5px"})

@app.callback(Output('image_wc', 'src'), [Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    plot_wordcloud(data=dfreq).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(Output('table', 'figure'), [Input('filter', 'value')])
def update_table(selected_value):
    if selected_value is None:
        table = go.Figure(data=[go.Table(
            header=dict(values=["Data", "Comentário", "Classificação"],
                        fill_color='#6371BF',
                        align='left',
                        font=dict(color='white', size=14)),
            cells=dict(values=[df.Data, df.Comentário, df.Classificação],
                    fill_color='white',
                    align='left',
                    font=dict(color='black', size=14)))
        ])
        table.update_layout(margin=dict(pad=5, b=5, t=30, r=5, l=5))
    else:
        filtered_df = df.loc[df["Classificação"]==selected_value]
        table = go.Figure(data=[go.Table(
            header=dict(values=["Data", "Comentário", "Classificação"],
                        fill_color='#6371BF',
                        align='left',
                        font=dict(color='white', size=14)),
            cells=dict(values=[filtered_df.Data, filtered_df.Comentário, filtered_df.Classificação],
                    fill_color='white',
                    align='left',
                    font=dict(color='black', size=14)))
        ])
        table.update_layout(margin=dict(pad=5, b=5, t=30, r=5, l=5))

    return table

@app.callback(Output("download", "data"), [Input("btn", "n_clicks")])
def generate_csv(n_nlicks):
    return send_data_frame(df.to_csv, filename="comentarios.csv")
