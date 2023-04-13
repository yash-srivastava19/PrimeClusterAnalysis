import cohere
import umap
import altair as alt 
from annoy import AnnoyIndex
import pandas as pd
import numpy as np 

# Load the API thing.
API_KEY = '----add your key here-----'
co_client = cohere.Client(API_KEY)

L = 1000 # Number of Primes. 

df = pd.read_csv('primes_100k.csv')
df = df[0:L]  # Just for testing. If ya believe, load all of it.

embeds = np.array(co_client.embed(texts=list(df['Num'].apply(str)), model='large').embeddings)

inst_1 = AnnoyIndex(embeds.shape[1], 'angular')
for i in range(L):
  inst_1.add_item(i, embeds[i])

# Save the thing.
inst_1.build(12)  # 12 trees for the search.
inst_1.save('test_primes.ann')
print('Done mi boy')

# Load the thing.
inst_1.load('test_primes.ann')

# Plot the thing
reducer = umap.UMAP(n_neighbors=12) 
umap_embeds = reducer.fit_transform(embeds)
df_explore = pd.DataFrame(data={'Number': df['Num'].apply(str)})
df_explore['x'] = umap_embeds[:,0]
df_explore['y'] = umap_embeds[:,1]

# Plot
chart = alt.Chart(df_explore).mark_circle(size=45).encode(
    x= alt.X('x',
        scale=alt.Scale(zero=False)
    ),
    y= alt.Y('y',
        scale=alt.Scale(zero=False)
    ),

    tooltip=['Number']
).properties(
    title='Primes Embedding',
    width=700,
    height=400,
)

chart.interactive()
