## section 0: read libraries 
import pandas as pd 
import numpy as np 
import geopandas as gpd 
import networkx as nx
import seaborn as sns 
import os
import re 

## section I: read and transform the data 

def data_read(path):
    """ read the Scopus dataset and return only the required columns
    """
    # read the files 
    files = [files for files in os.listdir(f'{path}') if files.endswith(".csv")]
    ds = pd.DataFrame()
    for file in files:
        read = pd.read_csv("../data/" + file).drop(['Link', "Source"], axis =1)
        ds = pd.concat([ds, read], axis =0)

    ds = ds[ds['Affiliations'].notna()]
    ds = ds.reset_index().drop('index', axis =1).reset_index()
    ds.head(2)
    columns = ["Auhtors", "Title", "Year", "Affiliation", "Abstract", "Document Type"]
    data = ds[columns]
    #return the dataset 
    return data

## section II: get the authors and affilaition 

def get_affiliation(ds): 
""" extract authors and affilaition from the data 
    """
    # read cities
    path = '../data/cities/'
    cities = pd.read_csv(path + 'worldcities.csv')
    # take the normal cities names 
    cities_one = cities.drop(['city', 'admin_name'], axis =1)
    cities_one.rename({'city_ascii': 'city'}, axis =1, inplace =True)
    # merge the second cities names 
    cities_two = cities.drop(['city_ascii', 'admin_name'], axis =1)
    ## third dataset
    cities_three = cities.drop(['city', 'city_ascii',], axis =1)
    cities_three.rename({'admin_name' : 'city'}, axis =1, inplace = True)
    ## concat both datasets 
    drop = ['University', 'Man', 'Bo']
    cities = pd.concat([cities_one, cities_two, cities_three])
    cities = cities[~(cities['city'].isin(drop))]
    cities['city'] = cities['city'].map(str)
    cities.drop_duplicates(subset = ['city'], inplace = True)
    ## take cities names 
    places = [f'{str(city)},' for city in cities['city'].tolist()]
    # authors 
    ## get first author (sta)
    first_author = ds['Authors'].apply(lambda x: x.split(',')[0])
    first_author = pd.DataFrame(first_author).reset_index()
    first_author.columns = ['index', 'first author']

    ## get second author (des)
    sec_id = []
    sec_auth = []
    for i, v in enumerate(ds['Authors']):
    # print(i)
        for s in v.split(','):
            # print(i,s)
            sec_id.append(i)
            sec_auth.append(s)
    second_author = pd.DataFrame(sec_id, sec_auth).reset_index()
    second_author.columns = ['second author', 'index']
    authors= pd.merge(first_author, second_author, on = 'index')
    authors["author value"] = 1
    authors = authors.groupby(["first author","second author"], sort=False, as_index=False).sum()
    # affiliation 
    # ## get first affi (sta)
    first_aff = ds['Affiliations'].apply(lambda x: str(x).split(',')[0])
    first_aff = pd.DataFrame(first_aff).reset_index()
    first_aff.columns = ['index', 'first aff']
    # ## get affi author (des)
    sec_aff_id = []
    sec_aff = []
    for i, v in enumerate(ds['Affiliations']):
         # print(i)
        # print(v)
        if ';' in str(v):
            for s in v.split(';'):
                sec_aff_id.append(i)
                sec_aff.append(s)
        if ';' not in str(v):
            sec_aff_id.append(i)
            sec_aff.append(v)

    second_aff = pd.DataFrame(sec_aff_id, sec_aff).reset_index()
    second_aff.columns = ['second aff', 'index']
    second_aff['second aff'] = second_aff['second aff'].apply(lambda x: str(x).split(',')[0])  
    affiliation= pd.merge(first_aff, second_aff, on = 'index')
    affiliation["aff value"] = 1
    affiliation = affiliation.groupby(["first aff","second aff"], sort=False, as_index=False).sum()

    aff_freq = pd.concat([affiliation['first aff'], affiliation['second aff']], ignore_index = True).value_counts().to_frame().reset_index()
    aff_freq.columns = ['aff', 'aff freq']
    affiliation = pd.merge(affiliation, aff_freq, left_on = 'first aff', right_on = 'aff', how = 'left').drop('aff', axis =1)
    ## cities
    ## get cities
    first_city = ds['Affiliations'].apply(lambda x: str(x).split(';')[0])
    first_city = pd.DataFrame(first_city).reset_index()
    first_city.columns = ['index', 'first city']
    ## get second city (des)
    sec_city_id = []
    sec_city = []
    for i, v in enumerate(ds['Affiliations']):
            # print(i)
        for s in str(v).split(';'):
            # print(i,s)
            sec_city_id.append(i)
            sec_city.append(s)
    second_city = pd.DataFrame(sec_city_id, sec_city).reset_index()
    second_city.columns = ['second city', 'index']
    dest= pd.merge(first_city, second_city, on = 'index')          
### get cities from affiliations 
    ci_one = {i: next((x for x in cities['city'] if x in v), np.nan) for i, v in enumerate(dest['first city'])}
    ci_two = {i: next((x for x in cities['city'] if x in v), np.nan) for i, v in enumerate(dest['second city'])}
    # ### clean 
    cities_dest = pd.concat([pd.Series(ci_one), pd.Series(ci_two)], axis =1).reset_index()
    cities_dest.columns = ['index', 'OrgCity', 'DestCity']
    cities_dest.drop('index', axis =1, inplace = True)
    ## clean citites 
    def clean_city(x):
        text = str(x).strip(',').strip()
        return text
    cities_dest['OrgCity'] = cities_dest['OrgCity'].apply(clean_city) 
    cities_dest['DestCity'] = cities_dest['DestCity'].apply(clean_city) 
    ### get x,y geo-data 
    #### org 
    city_first = pd.merge(cities_dest, cities, left_on = "OrgCity", right_on = 'city', how = 'left')[['OrgCity', 'lat', 'lng', 'country', 'iso3']]
    city_first.columns = ["OrgCity", 'OrgLat', "OrgLng", 'country_org', 'iso3_org']
    ### dest
    city_second = pd.merge(cities_dest, cities, left_on = "DestCity", right_on = 'city', how = 'left')[['DestCity', 'lat', 'lng', 'country', 'iso3']]
    city_second.columns = ["DestCity", 'DestLat', "DestLng", 'country_dest', 'iso3_dest']
    ## concat 
    cities_geo = pd.concat([city_first, city_second], axis = 1)
    geo_gis = pd.concat([dest, cities_geo], axis =1)
    add = pd.merge(geo_gis, affiliation, on = 'index')
    add['first aff'] = add['first aff'] + ' (' + add['OrgCity'] + ')'   
    add['second aff'] = add['second aff'] + ' (' + add['OrgCity'] + ')'

    return add 

## section III: get the themes 
 
 def get_themes(ds): 

def clean(text):
    import re 
    ## stops words and lem 
    # corpus stemming 
    from nltk import word_tokenize
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem import WordNetLemmatizer 
        '''Clean the corpus'''
        text = str(text).lower().strip('ltd').strip('elsevier')
        text = text.replace(']', '')
        text = text.replace('[', '')
        text = text.replace('©', '')
        text = text.replace('abstract', '')
        text = text.replace('chapter', '')
        text = re.sub(r'\d+[a-z]+', '', text)
        text = re.sub('[“”()*;‘~%/&]', '', text)
        text = re.sub(r'[.,„’:?!—-−°<]', '', text)
        text = re.sub(r'\d+[a-z]+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\d{2}', '', text)
        text = re.sub(r'[-]', ' ', text)
        text = text.replace('elsevier', '')
        return text.replace("[^A-Za-z ]", "").strip() 

    ds['Abstract'] = ds['Abstract'].apply(lambda x: clean(x))

    # stemmer = SnowballStemmer(language='english')
    lemmatizer = WordNetLemmatizer() 
    # create the function for stemming  
    def stemming(x):
            # tokanization 
        tokens = word_tokenize(x)
        # lemmatization 
        lem = [lemmatizer.lemmatize(word) for word in tokens]
        # stemming
        # stem = [stemmer.stem(token) for token in tokens]
        return ' '.join([i for i in lem]) 

    ds['Abs'] = pd.DataFrame(ds['Abstract'].apply(stemming))
    ## Topic modelling
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    tf_vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    matrix = tf_vectorizer.fit_transform(ds['Abs'])
    dtm = pd.DataFrame(matrix.toarray(), columns = tf_vectorizer.get_feature_names_out())
    # Initiate the model 
    model = LatentDirichletAllocation(n_components=7, random_state = 42)
    # fit transform the feature matrix
    model.fit(matrix)
    n_words = 15
    feature_names = tf_vectorizer.get_feature_names_out()

    topic_list = []
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()][::-1][:n_words]
        top_n = ' '.join(top_features)
        topic_list.append(f"topic_{'_'.join(top_features[:3])}") 
        # print the themes and most vommon words 
        print(f"Topic {topic_idx}: \n {top_n}")
## import local funtions 
    from functions import get_topics_terms_weights, print_topics_udf, get_topics_udf, getTermsAndSizes
    nmf_feature_names = tf_vectorizer.get_feature_names_out()
    nmf_weights = model.components_
    topics = get_topics_terms_weights(nmf_weights, nmf_feature_names)
    amounts = model.transform(matrix) * 100

    # Set it up as a dataframe
    topics = pd.DataFrame(amounts, columns=topic_list)
    # assign each cluster to the abstract 
    ds['Cluster'] = np.argmax(topics.values, axis=1)

    return ds 

## section IV: network analysis 

def network_analysis(data):
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout
    add = data.copy()
    m = add[~(add['first aff'].str.contains('nan'))]
    out = m[~(m['second aff'].str.contains('nan'))]
    d = out.sort_values('aff freq', ascending = False)
    # Create a graph from a pandas dataframe
    G = nx.from_pandas_edgelist(d, 
                                source = "second aff", 
                                target = "first aff", 
                                edge_attr = "aff value", 
                                create_using = nx.Graph())
    # Degree centrality
    fig,ax = plt.subplots(1,3, figsize = (14, 5))
    ####### first
    # degree centrality 
    degree_dict = nx.degree_centrality(G)
    degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['centrality'])
    # Plot top 10 nodes
    degree_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar", ax =ax[0], color = 'black', legend = False)
    ax[0].set_title('Degree of centrality', fontsize = 8)

    ####### second
    # Betweenness centrality
    betweenness_dict = nx.betweenness_centrality(G)
    betweenness_df = pd.DataFrame.from_dict(betweenness_dict, orient='index', columns=['centrality'])
    # Plot top 10 nodes
    betweenness_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar", ax = ax[1], color = 'black', legend = False)
    ax[1].set_title('Between centrality', fontsize = 8)
    ####### third 
    # Closeness centrality
    closeness_dict = nx.closeness_centrality(G)
    closeness_df = pd.DataFrame.from_dict(closeness_dict, orient='index', columns=['centrality'])
    # Plot top 10 nodes
    closeness_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar", ax =ax[2], color = 'black', legend = False)
    ax[2].set_title('Closeness centrality', fontsize = 8)
    plt.tight_layout()

## section V: field evolution 
def get_spatial_evolution(data, year1, year2, year3):
    import geopandas as gpd
    import geopandas as gpd
    ### import the world map 
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[world['continent'] != 'Antarctica']
    data = pd.read_excel('../output/final_dataset.xlsx')
    data.dropna(subset = ['OrgCity', 'DestCity'], inplace = True)
    ## turn coordinates into geodata 
    # turn lat and lng values into geodata 
    Dest_geo_points = [Point(x,y) for x,y in zip(data['DestLng'], data['DestLat'])]
    Dest = gpd.GeoDataFrame(data, geometry = Dest_geo_points)

    Org_geo_points = [Point(x,y) for x,y in zip(data['OrgLng'], data['OrgLat'])]
    Org = gpd.GeoDataFrame(data, geometry = Org_geo_points)

## print the three maps 
    with plt.style.context(("seaborn", "ggplot")):
        for set in [year1, year2, year3]:
            one =  Org.query(f'Year <= {set}')
            done = data.query(f'Year <= {set}')
            fig, ax = plt.subplots(figsize=(19,18)) 
            
            world.plot(figsize=(15,15), edgecolor="grey", color="white", ax =ax, zorder =0)
            one.plot(ax =ax, c= 'black', markersize = one['aff freq'], zorder =1, edgecolor = "pink")
            
            for slat,dlat, slon, dlon, num_flights, src_city, dest_city in zip(done["OrgLat"], done["DestLat"], done["OrgLng"], done["DestLng"], done["aff freq"], done["first aff"], done["second aff"]):
                plt.plot([slon , dlon], [slat, dlat], linewidth=num_flights/150, color="black", alpha= 0.1)
            plt.title(f"Knowledge production connection map: Birth ({set})")
            
