import streamlit as sl
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances


from dotenv import load_dotenv
from utils.b2 import B2
from utils.modeling import *

load_dotenv()
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
key_id=os.environ['B2_KEYID'],
secret_key=os.environ['B2_APPKEY'])
b2.set_bucket(os.environ['B2_BUCKETNAME'])
df = b2.get_df('countypres_2000-2020.csv')
df_president = b2.get_df('president.csv')
df['county_fips'] = df['county_fips'].astype('Int64')
df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)  


# REMOTE_DATA = '/Users/juanc/OneDrive/GitHubStuff/Project/Web_App/simple_streamlit/data/countypres_2000-2020.csv'
# df = pd.read_csv(REMOTE_DATA, dtype={"county_fips": str})  

# df['county_fips'] = df['county_fips'].astype(str).str.zfill(5) 

# PRESIDENT_DATA = '/Users/juanc/OneDrive/GitHubStuff/Project/Web_App/simple_streamlit/data/president.csv'
# df_president = pd.read_csv(PRESIDENT_DATA)

APP_TITLE = 'Election Results' 



                                    # INTERACTIVE
def display_president(year):
    president =  df_president[(df_president['year'] == year)]
    candidate = president['candidate'].values[0]
    party = president['party'].values[0]
    sl.markdown(f"## Presidential Winner: {candidate}")

def display_time_filters(df):
    year_list = list(df['year'].unique())
    year_list.sort()
    year = sl.sidebar.selectbox('Year', year_list, len(year_list)-1)
    sl.header(f'{year}')
    return year

def display_state_filter(df):
    state_list = [''] + list(df['state'].unique())
    state_list.sort()
    state = sl.sidebar.selectbox('State', state_list, len(state_list)-1)
    return state 

def display_county_filter(df, state):
    df = df[(df['state'] == state)]
    county_list = [''] + list(df['county_name'].unique())
    county_list.sort()
    county = sl.sidebar.selectbox('County', county_list, len(county_list)-1)
    return county

def get_state_results(year, state):
    data = df
    data = data[(data['year'] == year) & (data['state'] == state)]
    grouped = data.groupby('candidate')['candidatevotes'].sum() 
    state_result = grouped.sort_values(ascending=False).head()
    state_result.index.name = 'Candidate'
    state_result.name = 'Votes'
    return state_result

    
def get_county_results(year, state, county):
    data = df
    data = data[(data['year'] == year) & (data['state'] == state) & (data['county_name'] == county)]
    grouped = data.groupby('candidate')['candidatevotes'].sum() 
    county_result = grouped.sort_values(ascending=False).head()
    county_result.index.name = 'Candidate'
    county_result.name = 'Votes'
    return county_result

def get_state_map(year):
    grouped = df.groupby(['year', 'state', 'state_po', 'county_name', 'candidate', 'party'])['candidatevotes'].sum().reset_index()
    results = grouped.sort_values(by=['year', 'state', 'state_po', 'county_name', 'candidatevotes', 'party'], ascending=[True, True, True, True, False, False])

    winners = results[(results['year'] == year)] 
    state_candidate_votes = winners.groupby(['state', 'state_po', 'candidate', 'party'])['candidatevotes'].sum().reset_index()
    idx_max_votes = state_candidate_votes.groupby('state')['candidatevotes'].idxmax()
    winners = state_candidate_votes.loc[idx_max_votes]

# Map
    winners['hover_text'] = winners['candidate'] + ": " + winners['candidatevotes'].astype(str)

    fig = px.choropleth(winners, 
                    locations='state_po', 
                    color='party', 
                    locationmode='USA-states', 
                    hover_name='candidate', 
                    hover_data = ('candidatevotes',),
                    title=f'US Election {year} - Winners by State',
                    scope='usa',
                    color_discrete_map={'DEMOCRAT' : 'blue', 'REPUBLICAN': 'red'} )
    return fig


def get_county_map(year, state):
    grouped = df.groupby(['year', 'state', 'state_po', 'county_name', 'county_fips', 'candidate', 'party'])['candidatevotes'].sum().reset_index()
    grouped = df[(df['year'] == year) & (df['state'] == state)]
    results = grouped.sort_values(by=['year', 'state', 'state_po', 'county_name', 'county_fips', 'candidatevotes', 'party'], ascending=[True, True, True, True, True, False, False])
    
    winners = results[(results['year'] == year)] 
    county_candidate_votes = winners.groupby(['state', 'state_po', 'county_name', 'candidate', 'county_fips', 'party'])['candidatevotes'].sum().reset_index()
    idx_max_votes = county_candidate_votes.groupby('county_fips')['candidatevotes'].idxmax()
    winners = county_candidate_votes.loc[idx_max_votes]
# Map
    fig = px.choropleth(winners, 
                    geojson=counties, 
                    locations='county_fips', 
                    color='party', 
                    hover_name='candidate', 
                    hover_data = ('candidatevotes', 'state','county_name'),
                    title=f'US Election {year} - Winners by County',
                    scope='usa',
                    color_discrete_map={'DEMOCRAT' : 'blue', 'REPUBLICAN': 'red'} )
    return fig



                        #DISTANCE MATRIX 


def clean_data(df):
    gr_df = df.groupby(['year', 'state', 'party'])[['candidatevotes', 'totalvotes']].sum().reset_index()
    gr_df['vote_percentage'] = (gr_df['candidatevotes'] / gr_df['totalvotes'] * 100).round(2)
    features = gr_df[['year', 'state', 'party', 'candidatevotes', 'totalvotes', 'vote_percentage']]
    df_features = features[['year','state', 'party', 'vote_percentage']]
    df_features = df_features[df_features['party'].isin(['DEMOCRAT', 'REPUBLICAN'])]
    df_features = df_features.pivot(index = ['state', 'party'] , columns = 'year', values = 'vote_percentage')
    democrat_df = df_features.loc[(slice(None), 'DEMOCRAT'), :].droplevel(1)
    republican_df = df_features.loc[(slice(None), 'REPUBLICAN'), :].droplevel(1)
    difference_df = democrat_df - republican_df
    filtered_df = difference_df.drop("DISTRICT OF COLUMBIA")
    return filtered_df


cmap = mcolors.LinearSegmentedColormap.from_list("", ["red","white","blue"]) 

def state_year(df):
    fig = (plt.figure(figsize=(15,15)))
    sns.heatmap(df, annot=np.abs(df), cmap= cmap, center=0, cbar = False)
    # plt.title('Election By Year & State: Annotated by Difference in Vote %')
    plt.xlabel('Year')
    plt.ylabel('State')
    return fig


def year_year(df):
    euclidean_distances_year = pd.DataFrame(
    pairwise_distances(df.T),
    index=df.columns,
    columns=df.columns) 

    fig = (plt.figure(figsize=(15,15)))
    sns.heatmap(euclidean_distances_year, annot=False, cmap='RdBu_r', cbar = False)
    # plt.title('Euclidean Distance Between Each Election Year')
    plt.xlabel('Year')
    plt.ylabel('Year')
    plt.text(0.5, -0.1, 'Red indicates difference, Blue indicates similarity', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    return fig




def state_state(df):
    euclidean_distances_state = pd.DataFrame(
    pairwise_distances(df),
    index=df.index,
    columns=df.index) 
   
    fig = (plt.figure(figsize=(15,15)))
    sns.heatmap(euclidean_distances_state, annot=False, cmap="RdBu_r", cbar = False)
    # plt.title('Euclidean Distance Between Each State For All Years')
    plt.xlabel('State')
    plt.ylabel('State')
    plt.text(0.5, -0.2, 'Red indicates difference, Blue indicates similarity', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    return fig



def main():

    sl.set_page_config(APP_TITLE)

    sl.markdown("""
    #### TESTING THE TEST How To:
    Interactvive Portion(Election Results):
    - Filter on sidebar through Year, State, and County
    - Depending on setting, Tables and Results will change
    
    Static Portion(Similarity Matrix): 
    - Is not interactive
    - Diplays three similarty matrices 
    """)

    sl.title(APP_TITLE)
    
    # sl.write(df.shape)
    # sl.write(df.columns)
    # sl.write(df.head())


    year = (display_time_filters(df))
    president = display_president(year)    
    state = (display_state_filter(df))
    county = (display_county_filter(df, state))


    state_result = get_state_results(year,state)
    county_result = get_county_results(year, state, county)

    col1, col2 = sl.columns(2)
    with col1:     
        sl.header(f'{state} State Election Results')
        sl.table(state_result)
    with col2:
        sl.header(f'{county} County Election Results')
        sl.table(county_result)

    
    sl.plotly_chart(get_state_map(year))
    sl.plotly_chart(get_county_map(year, state))

    sl.title(""" Similarity Matrix """)

    percentage = clean_data(df)
    sl.header(f'Election By Year & State: Annotated by Difference in Vote %')
    sl.write(state_year(percentage))
    sl.header(f'Euclidean Distance Between Each Election Year')
    sl.write(year_year(percentage))
    sl.header(f'Euclidean Distance Between Each State For All Years')
    sl.write(state_state(percentage))
    
    

    
    





if __name__ == "__main__":
    main() 
 
