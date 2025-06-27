#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import nbformat
import plotly
import bokeh


# In[2]:


mgp_tweets = pd.read_csv('data/mgp_full_tweets.csv')


# # Latent Dirichlet Allocation (LDA)  on Term Frequency-Inverse Document Frequency (TF-IDF)

# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# `mgp_tweets['text']` contains the raw tweet texts
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

tfidf = vectorizer.fit_transform(mgp_tweets['text'])
feature_names = vectorizer.get_feature_names_out()


# In[4]:


num_topics = 5  # adjust number of topics

lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_W = lda_model.fit_transform(tfidf)
lda_H = lda_model.components_


# In[5]:


def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx+1}:")
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

num_top_words = 30
display_topics(lda_model, feature_names, num_top_words)


# In[15]:


# Assign the dominant topic to each tweet
mgp_tweets['dominant_topic'] = np.argmax(lda_W, axis=1)


# In[16]:


top_tweets = 20  # Number of top tweets to display for each topic

for topic_idx in range(num_topics):
    print(f"\nTopic {topic_idx + 1}:")
    # Get the indexes of tweets that belong to the current topic
    indexes = np.where(mgp_tweets['dominant_topic'] == topic_idx)[0]
    # Sort the indexes by their strength association with the topic
    sorted_indexes = indexes[np.argsort(lda_W[:, topic_idx][indexes])[::-1]]
    # Print the top tweets for the topic
    for i in sorted_indexes[:top_tweets]:
        print(f" - {mgp_tweets.iloc[i]['text']}")


# In[17]:


mgp_tweets.columns


# In[18]:


import matplotlib.pyplot as plt

# Count the number of tweets in each topic
tweet_counts = mgp_tweets['dominant_topic'].value_counts().sort_index()

# Create simple bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(1, num_topics + 1), tweet_counts)
plt.xlabel('Topics')
plt.ylabel('Number of Tweets')
plt.title('Distribution of Tweets by Dominant Topic')
plt.xticks(range(1, num_topics + 1))
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import numpy as np

def plot_top_words_for_first_topic(model, feature_names, num_top_words, filename='mgp_topic1.png'):
    # Extract the weights for the first topic
    topic_weights = model.components_[0]
    
    # Get the indices of the top words in this topic
    top_indices = topic_weights.argsort()[-num_top_words:][::-1]
    
    # Extract the top words and their corresponding weights
    top_words = [feature_names[i] for i in top_indices]
    weights = [topic_weights[i] for i in top_indices]
    
    # Plotting
    y_pos = np.arange(len(top_words))
    
    plt.figure(figsize=(10, 8))
    plt.barh(y_pos, weights, align='center', color='#008080')  # Change the color here
    plt.yticks(y_pos, top_words)
    plt.gca().invert_yaxis()  # To display the highest weights at the top
    plt.xlabel('Weight')
    plt.title('Top Words in Topic 1 by Weight MGP')
    
    # Save the figure before showing it
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(filename)
    
    # Then display the plot
    plt.show()

# Now call the function with the necessary arguments
plot_top_words_for_first_topic(lda_model, feature_names, 20)


# # Latent Semantic Analysis (LSA) on tweet matrix GloVe vectors

# In[21]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[26]:


import nltk
nltk.download('punkt_tab')


# In[27]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# download the nltk stop and punctuation packages
nltk.download('punkt')
nltk.download('stopwords')

# reload CSV file
mgp_tweets = pd.read_csv('data/mgp_full_tweets.csv')

# preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    
    # Lowercase and remove stopwords
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stopwords.words('english')]
    
    return tokens

# create new column for processed tweets from raw tweet 'text'column
mgp_tweets['processed_tweets'] = mgp_tweets['text'].apply(preprocess_text)



# In[29]:


from gensim.downloader import load

glove_model = load("glove-twitter-100")


# In[30]:


# Function to convert tokens to vectors using the GloVe model
def tokens_to_vectors(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.key_to_index:
            vectors.append(model[token])
    return vectors

# Vectorize the tokenized tweets
mgp_tweets['tweet_vectors'] = mgp_tweets['processed_tweets'].apply(lambda tokens: tokens_to_vectors(tokens, glove_model))

# Check the results
print(mgp_tweets['tweet_vectors'].head())


# In[31]:


import numpy as np

# Function to calculate the average vector for each tweet
def average_vectors(vectors):
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # return a zero vector if there are no vectors
        return np.zeros(glove_model.vector_size)

# Apply function to average the vectors for each tweet in new column 'average_vector'
mgp_tweets['average_vector'] = mgp_tweets['tweet_vectors'].apply(average_vectors)

# check the results
print(mgp_tweets['average_vector'].head())


# In[32]:


# stack all average vectors to create the matrix for SVD
tweet_matrix = np.vstack(mgp_tweets['average_vector'])

# Check the shape of the matrix (it should have as many rows as tweets and as many columns as dimensions in the GloVe vectors)
print(tweet_matrix.shape)


# # Truncated SVD **note** 
# I used this cause it can handle the negative values of vector, but it just wasn't any good

# In[33]:


from sklearn.decomposition import TruncatedSVD


num_topics = 7  

# Perform Truncated SVD on the tweet_matrix
svd_model = TruncatedSVD(n_components=num_topics, random_state=42)
U = svd_model.fit_transform(tweet_matrix)  # U now contains your tweet-topic association




# In[34]:


import numpy as np

num_top_tweets = 15  # Number of top tweets you want to examine per topic

# Loop through each component
for i in range(num_topics):
    # Get the column for the current component
    component = U[:, i]
    
    # Get the indices that would sort this component
    sorted_indices = np.argsort(component)[::-1]
    
    print(f"Top tweets for Component {i+1}:")

    # Display the top tweets for this component
    for idx in sorted_indices[:num_top_tweets]:
        print(f" - {mgp_tweets.iloc[idx]['text']}")
    print("\n" + "-"*50 + "\n")


# # still need to make these positive 

# In[35]:


# Convert the U matrix to absolute values before summing
topic_sums_abs = np.sum(np.abs(U), axis=0)  # Sum over each topic column using absolute values
total_sum_abs = np.sum(topic_sums_abs)  # Total sum of all associations using absolute values
topic_proportions_abs = topic_sums_abs / total_sum_abs  # Normalize to get proportions

# These proportions are now all positive and can be visualized directly

# Square the U matrix before summing
topic_sums_squared = np.sum(np.square(U), axis=0)  # Sum over each topic column using squared values
total_sum_squared = np.sum(topic_sums_squared)  # Total sum of all associations using squared values
topic_proportions_squared = topic_sums_squared / total_sum_squared  # Normalize to get proportions

# These proportions are based on squared values, ensuring positivity



# In[36]:


#simple bar chart
topic_labels = [f"Topic {i+1}" for i in range(num_topics)]
plt.figure(figsize=(10, 6))
plt.bar(topic_labels, topic_proportions_abs, color='skyblue') 
plt.xlabel('Topics')
plt.ylabel('Proportion of Corpus')
plt.title('Proportion of Corpus by LSA Topic (Absolute-Value Adjusted)')
plt.xticks(rotation=45)
plt.show()


# In[37]:


plt.figure(figsize=(10, 6))
plt.bar(topic_labels, topic_proportions_squared, color='skyblue')  # or use topic_proportions_squared
plt.xlabel('Topics')
plt.ylabel('Proportion of Corpus')
plt.title('Proportion of Corpus by LSA Topic (Squared-Adjusted)')
plt.xticks(rotation=45)
plt.show()


# # GloVe and NMF

# ### Don't need to do too much preprocessing here because it's vectorizing full words based on the tweet context

# In[38]:


# Load CSV file
mgp_tweets = pd.read_csv('data/mgp_full_tweets.csv')

# preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    
    # Lowercase and remove stopwords
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stopwords.words('english')]
    
    return tokens

# new column for processed tweets to vectorize
mgp_tweets['processed_tweets'] = mgp_tweets['text'].apply(preprocess_text)

# Check the processed tweets
print(mgp_tweets['processed_tweets'].head())


# ### Later on when I need to do visualizaitons, I need to make sure long tweets have linebreaks in them for viewing on the plot

# In[39]:


def insert_line_breaks(text, char_limit=50):
    words = text.split()
    processed_text = ""
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > char_limit:
            processed_text += "<br>"
            current_length = 0
        processed_text += word + " "
        current_length += len(word) + 1  # Account for the space
    
    return processed_text.strip()

# new column with the broken up tweets
mgp_tweets['tweet_text_with_breaks'] = mgp_tweets['text'].apply(insert_line_breaks)

# Verify the results
print(mgp_tweets[['text', 'tweet_text_with_breaks']].head())


# ### Vectorization with GloVe
# 
# We use the GloVe model, trained on 2 billion tweets, for word vectorization. This model yields 100-dimensional vectors, which encode semantic relationships between words specifically on Twitter, making this form of vectorization more suitable for capturing the meaning behind how language is being used on Twitter.
# 
# Here we:
# 1. **Glove Model and Vectors**: The pre-trained GloVe model is loaded to map tweet tokens to vectors.
# 2. **Token Vectorization**: Each token in a tweet is converted to its corresponding 100-dimensional GloVe vector. Tokens not found in the GloVe vocabulary are omitted.
# 3. **Vector Storage**: The vectors for each tweet are aggregated and stored
# 

# In[41]:


# Function to convert tokens to vectors using the GloVe model
def tokens_to_vectors(tokens, model):
    vectors = []
    for token in tokens:
        # Check if the token exists in the GloVe model
        if token in model.key_to_index:
            vectors.append(model[token])
        else:
            # Handle out-of-vocabulary tokens if necessary, e.g., by ignoring them or using a placeholder
            pass
    return vectors

# Apply the function to your tokenized tweets
mgp_tweets['tweet_vectors'] = mgp_tweets['processed_tweets'].apply(lambda tokens: tokens_to_vectors(tokens, glove_model))

# Check the result
print(mgp_tweets['tweet_vectors'].head())


# ### Averaging GloVe Vectors
# For each tweet, we average its token vectors into a single 100-dimensional vector, simplifying analysis. Tweets without valid tokens receive a zero vector.

# In[42]:


import numpy as np

# Function to calculate the average vector for each tweet
def average_vectors(vectors):
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # Return a zero vector if there are no vectors
        return np.zeros(glove_model.vector_size)

# Apply the function to average the vectors for each tweet
mgp_tweets['average_vector'] = mgp_tweets['tweet_vectors'].apply(average_vectors)

# Check the result
print(mgp_tweets['average_vector'].head())


# ### Stack the Vectors to create the matrix for NMF

# In[43]:


# Stack all average vectors to create the matrix for NMF
tweet_matrix = np.vstack(mgp_tweets['average_vector'])

# Check the shape of the matrix (it should have as many rows as tweets and as many columns as dimensions in the GloVe vectors)
print(tweet_matrix.shape)


# # Non-Negative Matrix Factorization (NMF) 
# 
# As suggested by its name, NMF requires non-negative data. I chose to apply the absolute value to handle this because it preserves the magnitude of vector components. This step enables decomposition into tweet-topic and topic-term matrices and should be more accurate at recognizing topics than LDA with TF-IDF, which uses word frequency/rarity scores
# 

# In[44]:


num_topics = 7 #Keeping in a stand-alone cell for iteration purposes


# In[45]:


# Apply the absolute value to the tweet matrix
tweet_matrix_abs = np.abs(tweet_matrix)

# Then apply NMF as before
nmf_model = NMF(n_components=num_topics, random_state=42)
W_abs = nmf_model.fit_transform(tweet_matrix_abs)  # Tweet-topic matrix
H_abs = nmf_model.components_  # Topic-term matrix

print(W_abs.shape)  # (number of tweets, number of topics)
print(H_abs.shape)  # (number of topics, vector dimensions)


# ### Examine top tweets for each topic
# 
# Next, I print the top 50 tweets associated with each topic to help determine the connecting characteristics and assign a topic title. But before I assign the titles, I will also consider the distribution as graphed below.

# In[46]:


# Number of top tweets to display for each topic
top_tweets = 50

for topic_idx, topic in enumerate(H_abs):
    print(f"Topic {topic_idx + 1}:\n")

    # Get indices of top tweets for this topic
    top_tweet_indices = W_abs[:, topic_idx].argsort()[-top_tweets:][::-1]

    for tweet_idx in top_tweet_indices:
        print(f" - {mgp_tweets.iloc[tweet_idx]['text']}")
    
    print("\n" + "-"*50 + "\n")


# In[47]:


import matplotlib.pyplot as plt

# Calculate the sum of associations for each topic
topic_sums = W_abs.sum(axis=0)

# Normalize to get proportions
topic_proportions = topic_sums / topic_sums.sum()

# Topic labels (you can customize these to be more descriptive)
topic_labels = [f"Topic {i+1}" for i in range(num_topics)]

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(topic_labels, topic_proportions, color='#008080')
plt.xlabel('Topics')
plt.ylabel('Proportion of Tweets')
plt.title('Proportion of Tweets by Topic, MGP')
plt.xticks(rotation=45)
plt.savefig('mgp_topic_distribution.png')
plt.show()


# In[48]:


# Assuming `W_abs` is the document-topic matrix from NMF
mgp_tweets['dominant_topic'] = np.argmax(W_abs, axis=1)


# In[49]:


mgp_tweets['dominant_topic'].value_counts()


# In[50]:


tweet_index = 0  # Example index of a tweet
tweet_topic_association = W_abs[tweet_index, :]
print(f"Associations of tweet {tweet_index} with topics: {tweet_topic_association}")


# In[51]:


tweet_text = mgp_tweets.iloc[0]['text']
print(f"Tweet 0: {tweet_text}")


# ## Create New Analysis Dataframe incorporating topic titles and linebreak tweets for later visualization

# In[52]:


# After qualitative review of the top 50 tweets in each category, I feel like these titles
# encompass the overall unifying theme of the topic
topic_names = [
    "Voice for Working Class",
    "Digital & Community Engagement",
    "Endorsements & Policy Priorities",
    "Voter Mobilization Efforts",
    "Anti-Extremism",
    "Volunteer & Fundraising",
    "Defending Rights & Freedoms"
]

# Create a new DataFrame for analysis
analysis_df = pd.DataFrame()

# Add the unprocessed tweet texts and tweet texts with line breaks
analysis_df['tweet_text'] = mgp_tweets['text']
analysis_df['tweet_text_with_breaks'] = mgp_tweets['tweet_text_with_breaks']  # Include the column with line breaks

# Add the association values for each topic with custom column names
for i, topic_name in enumerate(topic_names):
    analysis_df[topic_name] = W_abs[:, i]

analysis_df.head()


# In[53]:


import pandas as pd

# Set the maximum column width to, say, 400 characters
pd.set_option('display.max_colwidth', 1000)

# Now when you display the DataFrame, the 'tweet_text' column should show more content
analysis_df.head()


# In[54]:


sorted_topic_dfs = {}  # Initialize an empty dictionary to store sorted DataFrames for each topic

# Loop over all topic names and create a sorted DataFrame for each topic
for topic_name in topic_names:
    sorted_df = analysis_df.sort_values(by=topic_name, ascending=False)  # Sort by the current topic
    sorted_topic_dfs[topic_name] = sorted_df  # Store the sorted DataFrame in the dictionary


# In[55]:


def display_top_tweets(topic_input, top_n=5):
    """
    Displays the top N tweets for a given topic or custom column along with their association strengths for all topics.

    Parameters:
    - topic_input: Can be the exact name of the topic, a format like 'Topic_1', or a custom column like 'Anti-Extremism'.
    - top_n: The number of top tweets to display (default is 5).
    """
    # Handling custom column names like 'Anti-Extremism'
    if topic_input in analysis_df.columns:
        # If the input is a direct column name, use it to sort and display top tweets
        sorted_df = analysis_df.sort_values(by=topic_input, ascending=False)
        return sorted_df.head(top_n)[['tweet_text'] + topic_names + [topic_input]]
    
    # Existing logic to handle topic names and 'Topic_X' formats
    elif topic_input.startswith("Topic_"):
        try:
            index = int(topic_input.split('_')[1]) - 1  # Convert 'Topic_X' to an index
            topic_name = topic_names[index]
        except (IndexError, ValueError):
            print(f"Invalid topic input: '{topic_input}'. Please use a valid topic number or name.")
            return
    else:
        topic_name = topic_input
    
    if topic_name in topic_names:
        sorted_df = sorted_topic_dfs[topic_name]
        return sorted_df.head(top_n)[['tweet_text'] + topic_names]
    else:
        print(f"Topic '{topic_name}' not found.")


# In[56]:


# Using the format 'Topic_X'
display_top_tweets('Topic_1', 30)


# In[57]:


display_top_tweets('Topic_5', 30)


# # Preparing and Visualizing Tweets by Topic Interactively
# 
# The dataframe needs to be "melted" to pair each tweet with topic weights, in order to plot the top 50 tweets per category. This is because tweets can show traits of multiple topics. This way, we can see which tweets are most aligned with each topic.

# In[58]:


# Melt the analysis_df DataFrame to "long" format including both tweet text columns
melted_df_mgp = analysis_df.melt(id_vars=['tweet_text', 'tweet_text_with_breaks'], 
                             value_vars=topic_names,
                             var_name='Topic', 
                             value_name='Weight')

# Filter the top 50 tweets for each topic by weight
top_tweets_per_topic_mgp = melted_df_mgp.groupby('Topic').apply(lambda x: x.nlargest(50, 'Weight')).reset_index(drop=True)


# In[67]:


from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.transform import jitter, factor_cmap  # Corrected import
from bokeh.palettes import Category20  # Import a palette with enough colors
import pandas as pd
import numpy as np

output_notebook()

# top_tweets_per_topic_mgp is the DataFrame we just melted
top_tweets_per_topic_mgp['Weight_Jittered'] = top_tweets_per_topic_mgp['Weight'] + np.random.uniform(-0.02, 0.02, size=top_tweets_per_topic_mgp.shape[0])
source = ColumnDataSource(top_tweets_per_topic_mgp)

# Create a unique list of topics for color mapping
unique_topics = top_tweets_per_topic_mgp['Topic'].unique().tolist()

p1 = figure(sizing_mode="stretch_width", x_range=unique_topics, tools="")

# Adjust the title
p1.title.text = "Top 50 Tweets in Each Topic - Marie Glusenkamp Perez"  # Set the title text
p1.title.align = 'center'  # Center the title
p1.title.text_font_size = '16pt'  # Increase the title font size
p1.title.text_font = "helvetica"  # Optional: Change the font type
p1.title.text_color = "DarkSlateGrey"  # Optional: Change the font color

# Use factor_cmap for color differentiation
p1.circle(x=jitter('Topic', width=0.6, range=p1.x_range), y='Weight_Jittered', source=source, size=10,
         line_color='DarkSlateGrey', fill_alpha=0.6, hover_fill_color='firebrick', hover_alpha=1.0,
         color=factor_cmap('Topic', palette=Category20[len(unique_topics)], factors=unique_topics))

hover = HoverTool()
hover.tooltips = [("Tweet", "@tweet_text_with_breaks{safe}")]
p1.add_tools(hover)

# Customize label sizes and styles
p1.xaxis.major_label_orientation = 0.785  # 45-degree tilt
p1.xaxis.major_label_text_font_size = "12pt"
p1.yaxis.major_label_text_font_size = "12pt"

p1.xaxis.axis_label = "Topic"
p1.yaxis.axis_label = "Weight"
p1.axis.axis_label_text_font_size = "14pt"

p1.grid.grid_line_alpha = 0.3

show(p1)


# In[63]:


from bokeh.plotting import output_file

output_file("mgp_topic_scatter.html")  # Sets the output file name
show(p1)  # This will now save and open the file in browser


# In[40]:


import pandas as pd

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool


output_file('MGP_topics.html')

show(p1)


# # Deluzio Tweets

# In[68]:


# Load CSV file
deluzio_tweets = pd.read_csv('data/Deluzio_Tweets.csv')

# Example preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    
    # Lowercase and remove stopwords
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stopwords.words('english')]
    
    return tokens

# Assuming your tweets are in a column named 'tweet_text'
deluzio_tweets['processed_tweets'] = deluzio_tweets['text'].apply(preprocess_text)

# Check the processed tweets
print(deluzio_tweets['processed_tweets'].head())


# In[69]:


deluzio_tweets['tweet_vectors'] = deluzio_tweets['processed_tweets'].apply(lambda tokens: tokens_to_vectors(tokens, glove_model))

# Check the result
print(deluzio_tweets['tweet_vectors'].head())


# In[70]:


deluzio_tweets['average_vector'] = deluzio_tweets['tweet_vectors'].apply(average_vectors)

# Check the result
print(deluzio_tweets['average_vector'].head())


# In[71]:


# Stack all average vectors to create the matrix for NMF
tweet_matrix = np.vstack(deluzio_tweets['average_vector'])

# Check the shape of the matrix (it should have as many rows as tweets and as many columns as dimensions in the GloVe vectors)
print(tweet_matrix.shape)


# In[72]:


num_topics = 7


# In[73]:


# Apply the absolute value to the tweet matrix
tweet_matrix_abs = np.abs(tweet_matrix)

# Then apply NMF as before
nmf_model = NMF(n_components=num_topics, random_state=42)
W_abs = nmf_model.fit_transform(tweet_matrix_abs)  # Tweet-topic matrix
H_abs = nmf_model.components_  # Topic-term matrix

print(W_abs.shape)  # (number of tweets, number of topics)
print(H_abs.shape)  # (number of topics, vector dimensions)


# In[74]:


# Number of top tweets to display for each topic
top_tweets = 100

for topic_idx, topic in enumerate(H_abs):
    print(f"Topic {topic_idx + 1}:\n")

    # Get indices of top tweets for this topic
    top_tweet_indices = W_abs[:, topic_idx].argsort()[-top_tweets:][::-1]

    for tweet_idx in top_tweet_indices:
        print(f" - {deluzio_tweets.iloc[tweet_idx]['text']}")
    
    print("\n" + "-"*50 + "\n")


# In[75]:


import matplotlib.pyplot as plt

# Calculate the sum of associations for each topic
topic_sums = W_abs.sum(axis=0)

# Normalize to get proportions
topic_proportions = topic_sums / topic_sums.sum()

# Topic labels (you can customize these to be more descriptive)
topic_labels = [f"Topic {i+1}" for i in range(num_topics)]

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(topic_labels, topic_proportions, color='#7BAFD4')
plt.xlabel('Topics')
plt.ylabel('Proportion of Tweets')
plt.title('Proportion of Tweets by Topic, Chris Deluzio')
plt.xticks(rotation=45)
plt.savefig('deluzio_topic_distribution.png')
plt.show()


# In[76]:


# new column with the broken up tweets
deluzio_tweets['tweet_text_with_breaks'] = deluzio_tweets['text'].apply(insert_line_breaks)

# Verify the results
print(deluzio_tweets[['text', 'tweet_text_with_breaks']].head())


# In[77]:


# Define your topic names based on your analysis
topic_names = [
    "Union Solidarity & Local Empowerment",
    "Reproductive Rights & Fighting Extremism",
    "Community Events",
    "Jobs & Infrastructure",
    "Advocacy & Community Solidarity",
    "Corporate Greed & Economic Fairness",
    "Defending Rights & Democracy"
]

# Create a new DataFrame for analysis
analysis_df = pd.DataFrame()

# Add the unprocessed tweet texts and tweet texts with line breaks
analysis_df['tweet_text'] = deluzio_tweets['text']
analysis_df['tweet_text_with_breaks'] = deluzio_tweets['tweet_text_with_breaks']  # Include the column with line breaks

# Add the association values for each topic with custom column names
for i, topic_name in enumerate(topic_names):
    analysis_df[topic_name] = W_abs[:, i]

analysis_df.head()


# In[78]:


sorted_topic_dfs = {}  # Initialize an empty dictionary to store sorted DataFrames for each topic

# Loop over all topic names and create a sorted DataFrame for each topic
for topic_name in topic_names:
    sorted_df = analysis_df.sort_values(by=topic_name, ascending=False)  # Sort by the current topic
    sorted_topic_dfs[topic_name] = sorted_df  # Store the sorted DataFrame in the dictionary


# In[79]:


# Using the format 'Topic_X'
display_top_tweets('Topic_1', 30)


# In[80]:


# Melt the analysis_df DataFrame to "long" format including both tweet text columns
melted_df_deluzio = analysis_df.melt(id_vars=['tweet_text', 'tweet_text_with_breaks'], 
                             value_vars=topic_names,
                             var_name='Topic', 
                             value_name='Weight')

# Filter the top 50 tweets for each topic by weight
top_tweets_per_topic_deluzio = melted_df_deluzio.groupby('Topic').apply(lambda x: x.nlargest(50, 'Weight')).reset_index(drop=True)


# In[81]:


top_tweets_per_topic_deluzio


# In[82]:


from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.transform import jitter, factor_cmap  # Corrected import
from bokeh.palettes import Category20  # Import a palette with enough colors
import pandas as pd
import numpy as np

output_notebook()

# top_tweets_per_topic_deluziois the DataFrame we just prepared 
top_tweets_per_topic_deluzio['Weight_Jittered'] = top_tweets_per_topic_deluzio['Weight'] + np.random.uniform(-0.02, 0.02, 
                                                    size=top_tweets_per_topic_deluzio.shape[0])
source = ColumnDataSource(top_tweets_per_topic_deluzio)

# Create unique list of topics for color mapping
unique_topics = top_tweets_per_topic_deluzio['Topic'].unique().tolist()

p2 = figure(sizing_mode="stretch_width", x_range=unique_topics, tools="")
# Adjust the title
p2.title.text = "Top 50 Tweets in Each Topic  - Chris Deluzio"  # Set the title text
p2.title.align = 'center'  # Center the title
p2.title.text_font_size = '16pt'  # Increase the title font size
p2.title.text_font = "helvetica"  # Optional: Change the font type
p2.title.text_color = "DarkSlateGrey"  # Optional: Change the font color

# Use factor_cmap for color differentiation
p2.circle(x=jitter('Topic', width=0.6, range=p2.x_range), y='Weight_Jittered', source=source, size=10,
         line_color='DarkSlateGrey', fill_alpha=0.6, hover_fill_color='firebrick', hover_alpha=1.0,
         color=factor_cmap('Topic', palette=Category20[len(unique_topics)], factors=unique_topics))

hover = HoverTool()
hover.tooltips = [("Tweet", "@tweet_text_with_breaks{safe}")]
p2.add_tools(hover)

# Customize label sizes and styles
p2.xaxis.major_label_orientation = 0.785  # 45-degree tilt
p2.xaxis.major_label_text_font_size = "12pt"
p2.yaxis.major_label_text_font_size = "12pt"

p2.xaxis.axis_label = "Topic"
p2.yaxis.axis_label = "Weight"
p2.axis.axis_label_text_font_size = "14pt"

p2.grid.grid_line_alpha = 0.3

show(p2)


# In[83]:


import pandas as pd

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool


output_file('deluzio_topics.html')

show(p2)



