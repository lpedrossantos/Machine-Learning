from sklearn import tree
dtc_Gini = tree.DecisionTreeClassifier()
dtc_Gini

matrix = df.values
nc = df.shape[1]
table_x = matrix[:,3:nc-2]
table_y = matrix[:,nc-1]
table_y = table_y.astype('int')

type(table_y)

dtc_Gini=dtc_Gini.fit(table_x,table_y)
dtc_Gini

import pydot
column_names = ['hit_artist', 'danceability', 'energy', 'key', 'loudness',
       'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'duration_ms', 'time_signature', 'chorus_hit',
       'sections']

dot_data = tree.export_graphviz(dtc_Gini, out_file="hit_Gini.dot",
                                feature_names= column_names,
                                class_names=['flop','hit'],
                                filled=True, rounded=True,
                                special_characters=True)

(graph,) = pydot.graph_from_dot_file('hit_Gini.dot')
graph.write_png('hit_Gini.png')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('hit_Gini.png')

plt.figure(figsize = (100,100))
imgplot = plt.imshow(img)


hit_artist_bin_df = df[['artist','n_hit','n_flop','hit']].copy()
hit_artist_bin_df

hit_artist_bin_df = binning(hit_artist_bin_df)
hit_artist_bin_df = hit_artist_bin_df.drop(columns=['n_flop_bin','n_hit_bin'])
hit_artist_bin_df

for i, val in enumerate(hit_artist_bin_df.index):
    hit = hit_artist_bin_df.iloc[i]['n_hit']
    flop = hit_artist_bin_df.iloc[i]['n_flop']
    
    prob_hit = hit / (hit + flop)
    prob_flop = flop / (hit+flop)
    
    hit_artist_bin_df.loc[val,'hit_artist'] = round((hit*prob_hit + flop*prob_flop) / (hit + flop),5)

    merge_feature_means_into_df(df,hit_artist_bin_df,'artist','n_hit')
merge_feature_means_into_df(df,hit_artist_bin_df,'artist','n_flop')

#df = df.drop(columns=['hit_artist'])

column_names = ['track', 'artist', 'uri','n_hit','n_flop','danceability', 'energy', 'key', 'loudness',
       'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'duration_ms', 'time_signature', 'chorus_hit',
       'sections', 'decade', 'hit']
df = reindex_columns(df,column_names)
df


from mlxtend.preprocessing import TransactionEncoder
#Get words of the names of songs
tracks = df["track"]
tracks = tracks.to_numpy(dtype=str)
tracks = np.char.split(tracks)
#Create dataframe of the words
te = TransactionEncoder()
te_ary = te.fit(tracks).transform(tracks)
binary_database = pd.DataFrame(te_ary,columns=te.columns_)
binary_database

from mlxtend.frequent_patterns import apriori
import re
length = binary_database.shape
#Compute Frequent Itemset for the words of the songs
frequent_itemsets = apriori(binary_database, min_support=0.01, use_colnames=True)
#Create column with the number of occurences for each word
frequent_itemsets["number_of_occ"] = frequent_itemsets["support"].multiply(length[0])
frequent_itemsets.sort_values(by="number_of_occ",ascending=False)

#Other way of performing binning
new_df_bin = binning(new_df_bin,4, optimize = False)
new_df_bin = new_df_bin.round(2)
columns_names = []
for i, feature in enumerate(new_df_bin.columns):
    for j, val in enumerate(pd.unique(new_df_bin[feature])):
        col = '{}_{}'.format(feature,val)
        columns_names.append(col)
ohtenc = OneHotEncoder()
matrix_items = new_df_bin.values
matrix_items = ohtenc.fit(matrix_items).transform(matrix_items)
matrix_items = pd.DataFrame.sparse.from_spmatrix(matrix_items, columns = columns_names)
matrix_items

unique_vals = pd.Series(new_df_bin.nunique())
unique_vals_bin = pd.Series(new_df_bin.nunique())
pd.concat([unique_vals,unique_vals_bin], axis = 1, keys = ['before binning','after binning'])