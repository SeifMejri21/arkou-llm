import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import dash_bootstrap_components as dbc

import json
def read_json(file_path):
    try:
        f = open(file_path)
        data = json.load(f)
        f.close()
    except Exception as e:
        print(e)
        data = []
    return data



clean_preview_data = read_json("/mnt/c/Users/Administrator/PycharmProjects/arkou/misc_files/clean_preview_data.json")

# typed_keyz = ["type", "name", "place_id", "adresses", "goole_maps_url", "city", "reviews_score", "reviews_count", "reviews_urls", "reviews_count_str", "update_location_url"] 
typed_keyz = ["type", "name", "place_id", "goole_maps_url", "city", "reviews_score", "reviews_count", "reviews_urls", "reviews_count_str", "update_location_url"] 
typed_keyz = ["type", "name",  "city", "reviews_score", "reviews_count", "goole_maps_url", "reviews_urls"] 
# clean_preview_data = [{k:el[k] for k in typed_keyz if k in list(el.keys()) else None} for el in clean_preview_data]
clean_preview_data = [{k: el[k] if k in el else None for k in typed_keyz} for el in clean_preview_data]

setted_names = []
new_clean_preview_data = []
# for el in clean_preview_data:
#     if el['name'] not in setted_names:
#         setted_names.append(el['name'])
#         new_clean_preview_data.append(el)

# setted_palces = list(set([d['name'] for d in clean_preview_data]))

for el in clean_preview_data:
    if el['name'] not in setted_names:
        setted_names.append(el['name'])
        clean_el = {}
        for k in typed_keyz:
                if k in list(el.keys()):
                    if el[k] is None: clean_el[k] = el[k]
                    elif el[k] == []: clean_el[k] = None
                    elif k =='type' and el[k] != [] : clean_el[k] = el[k][5:] 
                    elif isinstance(el[k], str) or isinstance(el[k], int) or isinstance(el[k], float) or isinstance(el[k], bool):  clean_el[k] = el[k]
                    else: clean_el[k] = None
                else: clean_el[k] = None
        # print(el['goole_maps_url'])
        # print(clean_el['goole_maps_url'])
        # print(clean_el)
        new_clean_preview_data.append(clean_el)

print("clean_preview_data: ",len(clean_preview_data))
print("clean_preview_data setted: ",len(setted_names))
print("new_clean_preview_data: ",len(new_clean_preview_data))

df = pd.DataFrame(new_clean_preview_data)

df = df.where(pd.notnull(df), None)
df = df.map(lambda x: '' if x == [] else x)

print(df[:10])

# df["google_maps_url"] = df["goole_maps_url"].apply(lambda x: f"[Google Maps Link]({x})" if pd.notnull(x) else "")
# df["google_maps_url"] = df["goole_maps_url"].apply(lambda x: f"[Google Maps Link]({x}){{:target=\"_blank\"}}" if pd.notnull(x) else "")
# df["google_maps_url"] = df["goole_maps_url"].apply(lambda x: f"[Link]({x}){{:target=\"_blank\"}}" if pd.notnull(x) else "")

df["google_maps_url"] = df["goole_maps_url"].apply(lambda x: f'<a href="{x}" target="_blank">Google Maps Link</a>' if pd.notnull(x) else "")
df["reviews_urls"] = df["reviews_urls"].apply(lambda x: f'<a href="{x}" target="_blank">Reviews Link</a>' if pd.notnull(x) else "")


# for key, vaal in clean_preview_data[0].items(): print(key, type(vaal))

# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

PAGE_SIZE = 6

app.layout = dbc.Container([
    html.H1(
        "Djerba Touristic Database", 
        style={"textAlign": "center", "color": "#007bff"}
    ),    
        dash_table.DataTable(
        id='table',
        sort_action="native",
        filter_action="native",
        # columns=[{'id': x, 'name': x, 'presentation': 'markdown'} if x == 'goole_maps_url' else {'id': x, 'name': x} for x in df.columns],
        columns=[
            {'name': 'Type', 'id': 'type'},
            {'name': 'Name', 'id': 'name'},
            {'name': 'City', 'id': 'city'},
            {'name': 'Reviews Score', 'id': 'reviews_score'},
            {'name': 'Reviews Count', 'id': 'reviews_count'},
            # {'name': 'Google Maps URL', 'id': 'google_maps_url', 'presentation': 'markdown'}
            {'name': 'Google Maps Link', 'id': 'google_maps_url', 'presentation': 'markdown'},       
            {'name': 'Reviews Link', 'id': 'reviews_urls', 'presentation': 'markdown'}        ],
        page_size=PAGE_SIZE,
        data=df.to_dict('records'),
        # style_table={'height': '400px', 'overflowY': 'auto'},
        style_table={'overflowX': 'auto'},
        style_header={
            'textAlign': 'center',  # Center column headers
            'fontWeight': 'bold' , "color": "#007bff"
        },
        style_cell={
            'textAlign': 'center',  # Center cell content
            'padding': '10px'
        },
        markdown_options={"html": True},
    )
])

if __name__ == '__main__':
    app.run(debug=True)
