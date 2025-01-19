  
<h2 align="center">           </h2>
<a name="readme-top"></a>
<h2 align="center">Winning in Trump Country</h2>


<div align="center">
  <a href="https://github.com/samforwill/2024Strategies">
    <img src="images/MGP_For_Congress_Banner.png" alt="MGP Banner" style="width: 100%; max-width: 900px;">
    <img src="images/Deluzio_For_Congress_Banner.png" alt="Deluzio Banner" style="width: 100%; max-width: 900px;">
  </a>

<h3 align="center">Examining the Twitter Messaging Strategies of Two Democratic Newcomers Who Overcame the Red Tide in the 2022 Midterms </h3>

<div align= "left">  
<!-- TABLE OF CONTENTS -->
<!--details-->
  <summary> Table of Contents </summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#methodology">Methodology</a>
    </li>
    <li>
      <a href="#data-used">Data Used</a>
    </li>
    <li><a href="#selecting-the-candidates">Selecting the Candidates</a></li>
    <li><a href="#114th-congress-tweets-dataset-sentiment-classification">114th Congress Tweets Dataset Sentiment Classification</a></li>
    <li><a href="#topic-modeling---unsupervised">Topic Modeling - Unsupervised</a></li>
      <ul><li><a href="#baseline-model">Baseline Model</a></li></ul>
      <ul><li><a href="#advanced model">Advanced Model</a></li></ul>
    <li><a href="#marie-gluesenkamp-pérez-topics">Marie gluesenkamp Pérez Topics</a></li>  
    <li><a href="#chris-deluzio-topics">Chris Deluzio Topics</a></li>   
    <li><a href="#topic-comparisons-between-candidates">Topic Comparisons Between Candidates</a></li>
    <li><a href="#insights-and-conclusions">Insights and Conclusions</a></li>
    <!--< li><a href="#future-work">Future Work</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>-->
   </ol>
<!--/details-->


## Introduction

 This project applies Natural Language Processing (NLP) techniques to analyze the twitter messaging strategies of Marie gluesenkamp Pérez (WA-03) and Chris Deluzio (PA-17), Democratic newcomers who competed in two of the most challenging districts for Democrats in the 2022 midterm cycle.<br />
 
 Given the 2022 midterms were marked by the defeats of many election deniers and January 6th apologists, a secondary focus of this study is to assess the difference in our candidates' messaging strategies against distinct types of opponents— one faced Joe Kent in WA, a 'Kooky' nominee who fully embraced the 2020 election conspiracies, and the other faced Jeremy Shaffer in PA, a mainstream Republican who acknowledged, though reluctantly, Joe Biden's 2020 victory. 
 
 <p align="right">(<a href="#readme-top">back to top</a>)</p>
 
## Methodology
 
I used classification models to analyze a dataset of 5000 Twitter and Facebook posts by members of the 114th Congress. The dataset was pre-labeled with categories including bias, message nature, and political affiliation. My goal was to train the models to classify tweets based on these labels, which I then applied to my two candidates' tweets leading up to the 2022 midterm elections.
 
I also applied unsupervised topic modeling techniques, beginning with Latent Dirichlet Allocation (LDA) as a baseline method and then using Non-Negative Matrix Factorization (NMF) on Twitter GloVe vectors for refined clustering. After generating the topic groupings, I searched through each candidate's Tweet corpus for words closely associated with these topics to compare how often each candidate messaged on these topics to assess differing strategies. I used cosine similarity calculations within the tweet vector space to determine which words were most semantically similar in each topic. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Data Used
1. PVI score data was sourced from the [Cook Political Report](https://www.cookpolitical.com/cook-pvi/2023-partisan-voting-index/118-district-map-and-list).
2. 2022 Midterm Results were sourced from [The Daily Kos](https://www.dailykos.com/stories/2022/9/21/1742660/-The-ultimate-Daily-Kos-Elections-guide-to-all-of-our-data-sets).
3. The campaign tweets from Marie gluesenkamp Pérez and Chris Deluzio were hand-copied from their twitter accounts [@MGPforCongress](https://twitter.com/mgpforcongress) and [@ChrisforPA](https://twitter.com/chrisforPA)

4. The 114th Congress tweets addended with characterization inputs was sourced from Crowdflower's Data For Everyone Library via [Kaggle](https://www.kaggle.com/datasets/crowdflower/political-social-media-posts/data).

5. GloVe models and vector arrays were sourced from [Jeffrey Pennington, Richard Socher, and Christopher D. Manning of Stanford](https://nlp.stanford.edu/projects/glove/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Selecting the Candidates

**SYNOPSIS:** I determined which candidates to focus on through comparing their 2022 electoral margins with their district's Partisan Voter Index scores (PVI). I ultimately landed on Marie gluesenkamp Pérez in WA-03, and Chris Deluzio in PA-17. Below documents the step-by-step process of determining the candidates of focus<br />

<details>
<summary><b><big>Expand for Detailed Walk-Through Below</big></b></summary>

To identify standout candidates, I devised a 'Performance' metric by calculating the difference between each district's Partisan Voter Index (PVI) and the candidate's electoral margin in 2022. PVI measures how partisan the district is compared to the nation as a whole, based on how the constituents of those districts voted in previous presidential elections. This approach identified those who significantly outperformed their district's typical partisan lean.

![Overperformance](images/Overperformance.png)

Of the top 18 overperforming candidates indicated in the graph above by district title, I narrowed my focus to first-time candidates to avoid any influence of incumbency effects. Mary Peltola from Alaska was also excluded due to the state's use of Ranked Choice Voting, which, while I am personally a fan of RCV, complicates direct comparison of candidates in this context. <br />

That left me with 6 candidates to consider, all having overperformed their districts' partisan lean by at least 5 points.  The following 4 candidates greatly overperformed in their districts, but were eliminated from consideration for various reasons:

<img src="images/Candidates.png" alt="Candidates" width="600" style="display: block; margin: auto;">

Emilia Sykes would have been fun to analyze (and I love her glasses), but she deleted her campaign account following the election. Adam Frisch, who just barely fell short of victory in CO-03, was initially a candidate of interest, but was excluded due to the sheer volume of his tweets, which, thanks to Elon Musk's recent termination of free API access for Twitter, made data collection too labor-intensive. 

But ultimately, I found myself drawn to the candidate who arguably pulled off the biggest flip of the midterms. Her unique campaign and distinctive messaging strategy provided ample material for analysis, ultimately leading me to...

<img src="images/MGP.png" alt="MGP" width="600" style="display: block; margin: auto;">


Marie gluesenkamp Pérez! She faced cuckoo-bird Joe Kent, who expressed some extreme views like supporting the arrest of Dr. Anthony Fauci and endorsing the claims of a stolen 2020 election. In fact, he became the candidate for WA-03 after successfully primarying the serving Republican Congressperson, Jaime Herrera Beutler, one of only 10 republicans who voted to impeach Donald Trump following the events of January 6th.<br />


The next candidate I wanted to assess took a little more research to come to a decision, but I wanted to find a Democrat who overperformed in their district, while contending against an opponent who was a more mainstream Republican. I landed on...

<img src="images/Deluzio.png" alt="Deluzio" width="600" style="display: block; margin: auto;">

Chris Deluzio! He competed in a pure toss-up district and significantly outperformed against Jeremy Shaffer, who notably tried to sidestep affirming or denying the 2020 election fraud claims, and even released an ad promising to "protect women's healthcare." <br />

### Tweet Collection
As mentioned before, the termination of free API access meant manually compiling tweets for Chris Deluzio and Marie gluesenkamp Pérez, and then using a custom parsing script to organize and format these tweets into a structured dataset for analysis. Tweets were manually copied, separated by a '|' delimiter, and then organized into a corpus of around 1000 total tweets. [candidate notebook](MGP and Delozio.ipynb).

</details>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 114th Congress Tweets Dataset Sentiment Classification


Diving into this 2013-2014 dataset of politicians' social media felt like sorting through a cursed time capsule—fascinating, nostalgic, but ultimately reflecting an unrecognizable reality. Many of the key players in Congress, whose tweets I wrangled here, have fizzled or been replaced. And among the 5000 posts, not a single mention of 'Donald Trump'.

While the dataset does have its utility, its limitations were overwhelming. Nonetheless, for this exercise, I used some advanced classification modeling techniques to attempt to extract insights. The details are outlined below for those interested in the gritty process. The next section, however, is where we'll dive into the more interesting and fruitful analysis.

<details>
<summary><b><big>Detailed Process (For the Curious)</big></b></summary>

The dataset of 5000 tweets from 114th Congress members immediately presented a challenge: each tweet was tagged as "partisan" or "neutral," but provided no information on the political party of the tweeter. The data was presented like this:

| label |
| :----------- |
| From: Mitch McConnell (Senator from Kentucky) |
| From: Kurt Schrader (Representative from Oregon) |
| From: Michael Crapo (Senator from Idaho) |

To address this, I used this comprehensive member list from the [C-span 114th Summary Page](https://www.c-span.org/congress/members/?chamber=house&congress=114&all), including images, to match members' names and extract their partisan affiliation. The parsing code developed involved removal of name suffixes and resolving ambiguities in cases of shared last names, such as distinguishing between Rob Bishop (UT-01) and Mike Bishop (MI-08).<br />


After the data was addended to include each tweeter's political affilitation, several machine learning models were trained to classify tweets based on partisanship and content. This process encorporated Natural Language Processing (NLP) techniques, including TF-IDF vectorization for feature extraction and the application of multiple classifiers such as RandomForest, Naive Bayes, SVM, and Neural Networks within a pipeline structure optimized through GridSearchCV for hyperparameter tuning.<br />

The optimal models for each classification target—Party, Bias, and Message—were determined based on F1 scores and accuracy. Below are the best-performing models for each category, along with their respective confusion matrices:

- **Bias Prediction**: The Gaussian Naive Bayes Model emerged as the top performer for distinguishing between partisan and neutral tweets.
- **Party Affiliation Prediction**: The Multi-Layer Perceptron Classifier, a type of basic feedforward artificial neural network, was most effective in identifying the political party.
- **Message Category Prediction**: Gaussian Naive Bayes also proved to be the best model for categorizing the content of the tweets.

Word2Vec embeddings were able to enhancing the models' accuracy slightly. The final models were decent at discerning party affiliation, distinguishing between neutral and partisan messages, and categorizing the underlying message themes.<br />
![bias_and_party](images/bias_and_party.png)

![message](images/message.png)

These models were then applied to analyze the tweets of Marie gluesenkamp Pérez (MGP) and Chris Deluzio, in order to shed light on their campaign messaging strategies in the following section.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Analyzing Tweets with Trained Models

After fine-tuning the models on the 114th Congress dataset, I turned my attention to the real test: analyzing the campaign messages of Marie gluesenkamp Pérez (MGP) and Chris Deluzio. Here's the process:

### Analysis Workflow -- From Raw Tweets to Insights
#### Preprocessing Tweets:
Before feeding the tweets into the trained models, the corpus of each candidate was preprocessed:
1. **Tokenization**: The tweets were broken into individual words or 'tokens', making it easier for our models to analyze the text.
2. **Word Averaging with Word2Vec**: 
    - Word2Vec is a model that transforms words into vectors, capturing the semantic relationships between them. For example, Word2Vec understands that 'king' and 'queen' are related in a similar way as 'man' and 'woman'.
    - Word2Vec was used to convert the tokens into vectors, then averaged these vectors for each tweet. This process resulted in a numerical representation that captures the essence of each tweet, while making it digestible for machine learning models.
    - **Note on Lemmatization**: Typically, natural language processing might include a lemmatization step, where words are reduced to their base or 'lemma' (e.g., "running" becomes "run"). However, Word2Vec has the ability to discern the semantic meaning of words in their various forms, so I opted not to lemmatize our tokens. This allows us to retain the variations in the language used in the tweets.

### Normalization and Comparison:
Normalization was applied to the data for a balanced comparison of MGP's and Deluzio's messaging strategies:<br />
Balancing Volumes-- Due to the different numbers of tweets from each candidate, normalization allowed us to make comparisons based on tweet category proportions, not just total counts.

#### Visualization Analysis and Classifier Performance:

After classifying the tweets and normalizing the data, we employed visualizations to examine the differences in messaging strategies between Marie gluesenkamp Pérez and Chris Deluzio. Each visualization offers insights into specific aspects of their Twitter engagement, based on their classification results.

1. **Message Categories**: 

![cdmgp messages](images/cdmgp_messages.png)
The distribution of tweets across different message categories (e.g., 'policy', 'attack', 'media') provides insights into the focal points of each candidate's campaign. For example, a higher proportion of 'policy' tweets might indicate a campaign centered on substantive issues, while 'attack' tweets suggest a more confrontational approach.

2. **Partisan vs. Neutral Messages**: Visualizing the split between partisan and neutral tweets can reveal how each candidate balances broad appeal with targeted messaging to their base.

3. **Party Affiliation Predictions**: This visualization might show the predicted party alignment of tweets, offering a perspective on how closely each candidate aligns with their party's typical messaging.

**Performance Caveats**:
It's crucial to note the limitations in classifier performance when interpreting these visualizations. For instance, the 'message' category classifier achieved an accuracy of approximately 36.7%, with varying precision and recall across categories. This variability suggests that while some insights can be gleaned from the classified data, the findings should be taken with caution.

- Categories like 'personal' and 'policy' showed relatively better performance, but this was mostly due to the fact that the dataset was highly imbalanced. Since most of the messages were tagged as 'policy' and 'personal', the models learned to more often predict these categories 
- Categories with lower precision and recall, such as 'constituency' and 'other', had low classification reliability.

The model accuracy is severely limited on the data it was trained on. Due to the human-labeling process for this dataset, the errors and judgments by the topic labelers pass through the model during training, complicated futher by the large imbalance of the labels in certain categories. This problem is exacerbated during the prediction of tweets outside the 114th Congress dataset when introducing the tweetset of Marie gluesenkamp Pérez and Chris Deluzio. This issue could be mitigated by introducing the model to more data, more current data, and balancing along categories. This would be preferable to reduce the data dependency of the model and increase the robustness for general purposes.

While the visualizations provide a structured way to explore the candidates' messaging, the underlying limitations necessitate the use of different NLP techniques to glean important strategy insights. 

</details>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Topic Modeling - Unsupervised

## Baseline Model 
### Latent Dirichlet Allocation (LDA) on Term Frequency-Inverse Document Frequency (TF-IDF)

As a baseline, I used Latent Dirichlet Allocation (LDA) on Term Frequency-Inverse Document Frequency (TF-IDF) to analyze my candidates' tweets. TF-IDF measures the importance of words in a document (tweet) relative to the corpus (collection of all tweets in the campaign season). However, with only 1000 already-short tweets, LDA's effectiveness may be limited, and so I used this method as a baseline topic modeling method for comparison.<br/>

LDA uses these term frequencies to search for patterns and group things together into topics it thinks are related. It's up to the user to interpret these topics and discern underlying patterns. 

Sorting Marie gluesenkamp Pérez's tweetset into 5 topics created the following key word associations to each topic for MGP:

<img src="images/MGP_LDA.png" alt="MPG LDA" style="border: 2px solid #101010;"/><br/>



It seems like Topic 1 involves canvassing and GOTV messaging with terms like "volunteer", "join", "doors", "Vancouver" (big population center in the district where running up turnout numbers would be important to win). The other topics' words offer some hints at overarching themes, but they are not as easy to discern as the first topic.<br/>

TF-IDF scores words based on frequency and rarity, then LDA identifies topics based on these scores.  When determining topics, it assigns each word a weight indicating its importance to the topic.  To demonstrate this concept, below is a bar graph showing the importance weights for the words in MGP's first topic.

![MGP LDA](images/mgp_topic1.png)

Now, this is all well and good, but it *is* a baseline model, so let's not dive too deep into it and see if we can go ahead and up the ante a bit with more complex modeling.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Advanced Model -- Non-Negative Matrix Factorization (NMF) on 100-Dimensional Twitter GloVe Vectors

<!-- lol what a silly title. It seems almost designed to make you tune out... BUT DON'T! It's actually super cool and impressively useful for Topic Modeling.  -->

### GloVe (Global Vectors for Word Representation)

GloVe is an unsupervised learning algorithm designed by [these dudes](https://nlp.stanford.edu/projects/glove/) at Stanford. It can train on any corpus, but the GloVe model I used was performed on 2 billion tweets, which is important for a few reasons. First, GloVe trains on word-word co-occurence rates, but my model is trained specifically on how words are used together and semantically similar **on Twitter.** Considering the normal corpora used for text classification, Twitter is not newspaper articles, or books, or technical journals, so the word-word codependence rates that develop on twitter are, to a large degree, affected by the character limit itself! Also, the language is more vernacular, and tweets are designed to be shared, commented on, and interacted with. It's just a different semantic universe from other corpora.<br/>

So, given all these aspects of twitter language, I used a model that vectorizes every word into 100-dimensional vectors. Word embeddings can better handle polysemy (words with multiple meanings) by providing contextually appropriate vectors, whereas TF-IDF used in my baseline model treats each word instance identically regardless of semantic context.

### Non-Negative Matrix Factorization

Non-Negative Matrix Factorization (NMF) is a technique that decomposes high-dimensional datasets into lower-dimensional components. Compared to LDA on TF-IDF, NMF can handle denser data representations like GloVe embeddings more naturally, leveraging the semantic information embedded in word vectors. TF-IDF was like sorting through a giant word salad and counting the words that appear, but NMF with twitter-trained GloVe vectors knows that terms like 'Follow' and 'Mention' have related meaning in this semantic universe. This leads to better grouping and more interpretable and distinct topics.


### Process:
After some limited pre-processing, each word within the tweets was converted into a 100-dimensional vector using the GloVe model. The word vectors were averaged to produce a single vector to represents each tweet. These tweet vectors were stacked into a matrix, which served as the input for the NMF model to break down into associated topics. Given the non-negativity constraint inherent in NMF, absolute values of the tweet vectors were utilized to ensure all inputs were non-negative. (I also tried shifting the vector values to all exist in positive space, but it didn't yield a noticeable improvement in the resulting topics.) <br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Marie gluesenkamp Pérez Topics

Here is the distribution of unlabeled tweet topics that the model found to share semantic similarity (I found 7 topics to be the best grouping parameter).

![MGP Topic Distribution](images/mgp_topic_distribution.png)

Once the tweets were grouped , I went through the top 50 tweets associated with each topic, and found the tweets to be best described by the following themes: 

<details>
<summary><b><big>
1. MGP Topic 1 -- "Voice for Working Class"</big></b></summary>

<a href="https://twitter.com/MGPforCongress/status/1578454359788376065">
    <img src="images/mgp_working_class_tweet.png" alt="Working Class Tweet" />
</a>
</details>


<details>
<summary><b><big>
2. MGP Topic 2 -- "Digital & Community Engagement"</big></b></summary>

<a href="https://twitter.com/MGPforCongress/status/1580754873540542464">
    <img src="images/mgp_digital_tweet.png"  alt="Digital Tweet" />
</a>
</details>

<details>
<summary><b><big>
3. MGP Topic 3 -- "Endorsements & Policy Priorities"</big></b></summary>

<a href="https://twitter.com/MGPforCongress/status/1561818273330503681">
    <img src="images/mgp_policy_tweet.png"  alt="Policy Tweet" />
</a>
</details>

<details>
<summary><b><big>
4. MGP Topic 4 -- "Voter Mobilization Efforts"</big></b></summary>

<a href="https://twitter.com/MGPforCongress/status/1567961466850131969">
    <img src="images/mgp_mobilization_tweet.png"  alt="Mobilization Tweet" />
</a>
</details>

<details>
<summary><b><big>
5. MGP Topic 5 -- "Anti-Extremism"</big></b></summary>

<a href="https://twitter.com/MGPforCongress/status/1572635581280321537">
    <img src="images/mgp_extremism_tweet.png"  alt="Anti-Extremism Tweet" />
</a>
</details>


<details>
<summary><b><big>
6. MGP Topic 6 -- "Volunteer & Fundraising"</big></b></summary>

<a href="https://twitter.com/MGPforCongress/status/1584586583763988480">
    <img src="images/mgp_fundraising_tweet.png"  alt="Fundraising Tweet" />
</a>
</details>

<details>
<summary><b><big>
7. MGP Topic 7 -- "Defending Rights & Freedoms"</big></b></summary>

<a href="https://twitter.com/MGPforCongress/status/1570516774558527488">
    <img src="images/mgp_defend_tweet.png"  alt="Defending Tweet" />
</a>
</details>

The important thing to note here is that each tweet isn't individually put into one distinct category, but rather, each tweet is given a score for the extent to which it is associated with each topic found by NMF. This makes natural sense, because you can talk about multiple things in one statement-- A tweet like "My extreme opponent wants to ban abortion, but I will work to protect choice. That's why I'm endorsed by Planned Parenthood" would have high scores in Topics 3, 5, and 7, but would be less associated with the other topics. </br>

The interactive graph linked below shows the top 50 tweets associated with each category; hover mouse over datapoint to see full tweet.


[![MGP Topics](images/MGP_Topics.png)](https://samforwill.w3spaces.com/bokeh/mgp_topics.html)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Chris Deluzio Topics

Here is the distribution of tweet topics for Chris Deluzio that the model found to share semantic similarity.

![Deluzio Topic Distribution](images/deluzio_topic_distribution.png)

Once the tweets were grouped , I went through the top 50 tweets associated with each topic, and found the tweets to be best described by the following themes: 

<details>
<summary><b><big>
1. Deluzio Topic 1 -- "Union Solidarity & Local Empowerment"</big></b></summary>

<a href="https://twitter.com/ChrisForPA/status/1544675057737269252">
    <img src="images/deluzio_union_tweet.png"  alt="Union Tweet" />
</a>
</details>

<details>
<summary><b><big>
2. Deluzio Topic 2 -- "Reproductive Rights & Fighting Extremism"</big></b></summary>

<a href="https://twitter.com/ChrisForPA/status/1562466032601600001">
    <img src="images/deluzio_abortion_tweet.png"  alt="Abortion Tweet" />
</a>
</details>

<details>
<summary><b><big>
3. Deluzio Topic 3 -- "Community Events"</big></b></summary>

<a href="https://twitter.com/ChrisForPA/status/1568669438295306241">
    <img src="images/deluzio_community_tweet.png"  alt="Communmity Tweet" />
</a>
</details>

<details>
<summary><b><big>
4. Deluzio Topic 4 -- "Jobs & Infrastructure"</big></b></summary>

<a href="https://twitter.com/ChrisForPA/status/1554920539319283712">
    <img src="images/deluzio_jobs_tweet.png"  alt="Jobs Tweet" />
</a>
</details>

<details>
<summary><b><big>
5. Deluzio Topic 5 -- "Advocacy & Community Solidarity"</big></b></summary>

<a href="https://twitter.com/ChrisForPA/status/1552026970036240389">
    <img src="images/deluzio_advocacy_tweet.png"  alt="Advocacy Tweet" />
</a>
</details>

<details>
<summary><b><big>
6. Deluzio Topic 6 -- "Corporate Greed & Economic Fairness"</big></b></summary>

<a href="https://twitter.com/ChrisForPA/status/1585700173577158656">
    <img src="images/deluzio_corporations_tweet.png"  alt="Corporations Tweet" />
</a>
</details>

<details>
<summary><b><big>
7. Deluzio Topic 7 -- "Defending Rights & Democracy"</big></b></summary>

<a href="https://twitter.com/ChrisForPA/status/1572015179990114304">
    <img src="images/deluzio_defend_tweet.png"  alt="Defending Tweet" />
</a>
</details>


The interactive graph linked below shows the top 50 tweets associated with each category; hover mouse over datapoint to see full tweet.

[![Deluzio Topics](images/Deluzio_Topics.png)](https://samforwill.w3spaces.com/bokeh/deluzio_topics.html)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Topic Comparisons Between Candidates

To quantify and compare tweet frequency on specific topics for each candidate, I analyzed their tweet corpora using keywords and semantically similar terms identified in topic modeling. I used the Twitter-trained GloVe model and cosine similarity to assist in keyword selection to reduce bias. For each keyword, I printed the 50 nearest words using cosine similarity and then divided them into 2 groups -- relevant and irrelevant--  based on their semantic context.

Take the term 'extreme' as an example. The GloVe model identified similar terms like 'radical', 'dangerous', and 'far-right', alongside unrelated terms such as 'fitness', 'depression', and 'jihadist'. I then divided these into the relevant and irrelevant lists, calculated their average vectors, and used the GloVe model to isolate terms associated with my context and exclude terms outside the zone of interest.

The topic-words I chose to explore were:
1. 'extreme'
2. 'volunteer'
3. 'unions'
4. 'endorsement'
5. 'protect'
6. 'folks'
7. 'abortion'
8. 'manufacturing'
9. 'china'
10. 'corporations'

The candidates' tweets were searched for these terms along with a list of semantically-similar terms to gauge how frequently each candidate messaged on the associated topic. The interactive graph below shows the results of these queries, the exact terms used in each list, along with example tweets from each candidate for each category:

<div>
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Bokeh Plot</title>
    <style>
      html, body {
        box-sizing: border-box;
        display: flow-root;
        height: 100%;
        margin: 0;
        padding: 0;
      }
    </style>
    <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-3.4.0.min.js"></script>
    <script type="text/javascript">
        Bokeh.set_log_level("info");
    </script>
  </head>
  <body>
    <div id="cc7b3220-e3d3-416a-a1d2-4474519bbab1" data-root-id="p1171" style="display: contents;"></div>
  
    <script type="application/json" id="d21c16d9-0a79-4e96-9a3e-52413aabc49b">
      {"58198d52-01a3-4336-9311-35f3487b3d7e":{"version":"3.4.0","title":"Bokeh Application","roots":[{"type":"object","name":"Figure","id":"p1171","attributes":{"width":1000,"x_range":{"type":"object","name":"FactorRange","id":"p1170","attributes":{"factors":[["\"extreme\"","MGP"],["\"extreme\"","Deluzio"],["\"volunteer\"","MGP"],["\"volunteer\"","Deluzio"],["\"unions\"","MGP"],["\"unions\"","Deluzio"],["\"endorsement\"","MGP"],["\"endorsement\"","Deluzio"],["\"protect\"","MGP"],["\"protect\"","Deluzio"],["\"folks\"","MGP"],["\"folks\"","Deluzio"],["\"abortion\"","MGP"],["\"abortion\"","Deluzio"],["\"manufacture\"","MGP"],["\"manufacture\"","Deluzio"],["\"china\"","MGP"],["\"china\"","Deluzio"],["\"corporations\"","MGP"],["\"corporations\"","Deluzio"]]}},"y_range":{"type":"object","name":"DataRange1d","id":"p1173","attributes":{"start":0}},"x_scale":{"type":"object","name":"CategoricalScale","id":"p1181"},"y_scale":{"type":"object","name":"LinearScale","id":"p1182"},"title":{"type":"object","name":"Title","id":"p1174","attributes":{"text":"Candidate Tweet Proportions by Primary Topic","text_font_size":"16pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p1200","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p1167","attributes":{"selected":{"type":"object","name":"Selection","id":"p1168","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p1169"},"data":{"type":"map","entries":[["index",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAEAAAACAAAAAwAAAAQAAAAFAAAABgAAAAcAAAAIAAAACQAAAAoAAAALAAAADAAAAA0AAAAOAAAADwAAABAAAAARAAAAEgAAABMAAAA="},"shape":[20],"dtype":"int32","order":"little"}],["Topic",{"type":"ndarray","array":["extreme","extreme","volunteer","volunteer","unions","unions","endorsement","endorsement","protect","protect","folks","folks","abortion","abortion","manufacture","manufacture","china","china","corporations","corporations"],"shape":[20],"dtype":"object","order":"little"}],["Candidate",{"type":"ndarray","array":["MGP","Deluzio","MGP","Deluzio","MGP","Deluzio","MGP","Deluzio","MGP","Deluzio","MGP","Deluzio","MGP","Deluzio","MGP","Deluzio","MGP","Deluzio","MGP","Deluzio"],"shape":[20],"dtype":"object","order":"little"}],["Proportion",{"type":"ndarray","array":{"type":"bytes","data":"qll/Br+Jvj+QxuOkV0O4PyaHuMoUMM8/srhPE8VKxT+Kns7tteSgPzu16m6gjMQ/QXZhMRJKqz96Av+Z9dCiPyaHuMoUML8/zhP4z6yHxj+Kns7tteSwPwL/mfXQEsI/6txeSw5xtT/VqruBMlKyP4H5QHZhMaI/b6CMlMQXwD9uryWHuMpkPxIsu06tuqs/gflAdmExoj+QxuOkV0PIPw=="},"shape":[20],"dtype":"float64","order":"little"}],["Semantically_Similar_Words",{"type":"ndarray","array":["extreme, radical, extremist, extremists, extremism, dangerous, insanity, insane, chaos, violent, &lt;br&gt;far-right, rightwing, right-wing, anti-choice, homophobic, bigot, bigoted, MAGA, violent, insurrection, &lt;br&gt;xenophobic, fascists, oppressive, authoritarian, divisive, anti-gay, misogynistic, deranged, hateful, anti-democratic, &lt;br&gt;anti-democracy, conspiracy, conspiratorial, QAnon, insurrectionist","extreme, radical, extremist, extremists, extremism, dangerous, insanity, insane, chaos, violent, &lt;br&gt;far-right, rightwing, right-wing, anti-choice, homophobic, bigot, bigoted, MAGA, violent, insurrection, &lt;br&gt;xenophobic, fascists, oppressive, authoritarian, divisive, anti-gay, misogynistic, deranged, hateful, anti-democratic, &lt;br&gt;anti-democracy, conspiracy, conspiratorial, QAnon, insurrectionist","volunteer, volunteers, volunteering, register, donate, donations, donation, fundraiser, contribute, raise, &lt;br&gt;join, canvass, canvasser, canvassers, knock, support, supporters, outreach, fundraising, participate, &lt;br&gt;raising, donating, fundraise, pledge, contribution, contributions, organize, organizing, donor, donors","volunteer, volunteers, volunteering, register, donate, donations, donation, fundraiser, contribute, raise, &lt;br&gt;join, canvass, canvasser, canvassers, knock, support, supporters, outreach, fundraising, participate, &lt;br&gt;raising, donating, fundraise, pledge, contribution, contributions, organize, organizing, donor, donors","unions, union, unionize, solidarity, tradecraft, labor, collective, strike, striking","unions, union, unionize, solidarity, tradecraft, labor, collective, strike, striking","endorsement, endorsements, endorses, sponsorship, endorsing, endorsed, sponsor, sponsoring, endorse, sponsors, &lt;br&gt;sponsored","endorsement, endorsements, endorses, sponsorship, endorsing, endorsed, sponsor, sponsoring, endorse, sponsors, &lt;br&gt;sponsored","protect, protecting, defend, protected, preserve, protection, protects, prevent, restore, rights, &lt;br&gt;destroy, threaten, threatens, protecting, preventing, prevention, restrict, threatening, advocate, advocates, &lt;br&gt;strengthen, weaken, rights, freedom, freedoms, constitutional","protect, protecting, defend, protected, preserve, protection, protects, prevent, restore, rights, &lt;br&gt;destroy, threaten, threatens, protecting, preventing, prevention, restrict, threatening, advocate, advocates, &lt;br&gt;strengthen, weaken, rights, freedom, freedoms, constitutional","folks, folk, yall, class, small, average, regular, fellow, neighbor, neighbors","folks, folk, yall, class, small, average, regular, fellow, neighbor, neighbors","abortion, abortions, parenthood, pro-life, contraception, rape, incest, prolife, birth, prochoice, &lt;br&gt;pro-choice, pregnancy, adoption, pregnant, birthing, pregnancy, fertility, aborted, surrogate, contraceptive, &lt;br&gt;parenting, conceive, conception, fertility, ivf, miscarriage, contraceptives, abstinence, mother, motherhood, &lt;br&gt;pregnancy, breastfeeding, pregnancies, contraceptives, fertility, birth, prolife, surrogate, pregnancy, breastfeeding, &lt;br&gt;pregnancies, contraceptives, fertility, birth, premature, surrogate, miscarriage, ivf","abortion, abortions, parenthood, pro-life, contraception, rape, incest, prolife, birth, prochoice, &lt;br&gt;pro-choice, pregnancy, adoption, pregnant, birthing, pregnancy, fertility, aborted, surrogate, contraceptive, &lt;br&gt;parenting, conceive, conception, fertility, ivf, miscarriage, contraceptives, abstinence, mother, motherhood, &lt;br&gt;pregnancy, breastfeeding, pregnancies, contraceptives, fertility, birth, prolife, surrogate, pregnancy, breastfeeding, &lt;br&gt;pregnancies, contraceptives, fertility, birth, premature, surrogate, miscarriage, ivf","manufacture, manufacturers, manufacturer, manufacturing, manufactured, jobs, job, infrastructure, construction, industry, &lt;br&gt;industries, produce, products","manufacture, manufacturers, manufacturer, manufacturing, manufactured, jobs, job, infrastructure, construction, industry, &lt;br&gt;industries, produce, products","china, chinese, overseas, outsourcing, outsource","china, chinese, overseas, outsourcing, outsource","corporations, lobbyists, ceo, lobbyist, executives, ceos, billionaires, PAC, PACs, super-pac, &lt;br&gt;millionaires, consumers, corporate, corporates, wealthy, taxpayers, lobbying, multinationals, corrupt, taxes, &lt;br&gt;profit, profits","corporations, lobbyists, ceo, lobbyist, executives, ceos, billionaires, PAC, PACs, super-pac, &lt;br&gt;millionaires, consumers, corporate, corporates, wealthy, taxpayers, lobbying, multinationals, corrupt, taxes, &lt;br&gt;profit, profits"],"shape":[20],"dtype":"object","order":"little"}],["Sample_Tweets",{"type":"ndarray","array":["&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Joe Kent has now pulled ahead of Herrera Beutler &lt;br&gt;and it appears he will be our opponent in &lt;br&gt;November. Kent will only add to the dysfunction &lt;br&gt;paralyzing our country. With your backing we can &lt;br&gt;reject extremism and support the common good. I &lt;br&gt;look forward to earning your support!&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Extremists don\u2019t pass bills \u2013 they obstruct &lt;br&gt;progress. There is too much at stake to elect &lt;br&gt;representatives who tout lies or sit on their &lt;br&gt;hands. We need legislators who can get sh!t done.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Help elect a pro-choice mother who works in the &lt;br&gt;trades &amp; keep Republican extremist, Joe Kent out &lt;br&gt;of Congress","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;My right-wing opponent wants to put an abortion &lt;br&gt;ban in the Constitution, and is on the record &lt;br&gt;opposing exceptions for rape and incest victims. &lt;br&gt;He and his pals Dr. Oz &amp; Doug Mastriano are too &lt;br&gt;extreme for #PA17. Let's win this @JoshShapiroPA &lt;br&gt;@JohnFetterman&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Here\u2019s the world my extremist opponent Jeremy &lt;br&gt;Shaffer wants for western PA. He\u2019s a threat to &lt;br&gt;our freedom, and you better believe I\u2019m not going &lt;br&gt;to stand by and let him and Doug Mastriano take &lt;br&gt;away your right to choose. #PA17&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Honored to be endorsed by @NARAL . You'll always &lt;br&gt;know where I stand on protecting your freedom to &lt;br&gt;choose. And I'm going to fight hard to protect &lt;br&gt;you from extremists like Doug Mastriano and my &lt;br&gt;opponent and their plans to attack your rights.","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Thank you to everyone who has reached out about &lt;br&gt;volunteering for the campaign. No matter who I\u2019m &lt;br&gt;running against, we\u2019re going to need your help to &lt;br&gt;win in November! You can sign up on my website: &lt;br&gt;http://marieforcongress.com/volunteer/ or reach &lt;br&gt;out to volunteer@Marieforcongress.com&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;(3/3) I\u2019m ready to #FlipWA03 and get to work on &lt;br&gt;the issues that affect our communities most &lt;br&gt;deeply. We\u2019re counting on your grassroots support &lt;br&gt;to help us defeat Joe Kent and bring the voices &lt;br&gt;of working Washingtonians to Congress. Chip in &lt;br&gt;here: \u200b\u200b&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;You're proud to be endorsed by someone who used &lt;br&gt;tax payer dollars to hold private donor dinners? &lt;br&gt;Seems right.","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;#TeamDeluzio is powered by the people. No &lt;br&gt;corporate dark money. And I'm up against a &lt;br&gt;Republican who's dumped a million of his own $$$$ &lt;br&gt;to try to buy this seat. If you can manage, &lt;br&gt;please support our fight for the common good. &lt;br&gt;https://secure.actblue.com/donate/chris-deluzio-for-congress-social&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;No better way to kick off a union weekend of &lt;br&gt;action than with @AlleghenyLabor at the &lt;br&gt;@steelworkers ! These folks are fired up, ready &lt;br&gt;to fight for our common good, and knock doors &lt;br&gt;across #PA17 today. \u270a&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Despite the weather, #TeamDeluzio is putting the &lt;br&gt;work in! Shout out to all of our amazing &lt;br&gt;volunteers across #PA-17 who are fighting for our &lt;br&gt;common good.","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Joe Kent says he's all about \u201cLaw &amp;amp; Order\u201d &lt;br&gt;but wants to defund the FBI &amp;amp; local police. &lt;br&gt;Says he's pro worker, but is anti union. Says he &lt;br&gt;loves democracy, but won\u2019t accept the 2020 &lt;br&gt;election results. These mental gymnastics are &lt;br&gt;gold medal worthy. \ud83e\udd47&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Do you enjoy having weekends off? Or a 40-hour &lt;br&gt;work week? Overtime pay? Workday breaks? Thank a &lt;br&gt;union member! This Labor Day, let\u2019s celebrate the &lt;br&gt;workers who fought to secure the protections we &lt;br&gt;enjoy today. Let\u2019s keep raising the bar.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Hello from Pacific County, where we\u2019re having fun &lt;br&gt;in the rain at the South Bend Labor Day parade! &lt;br&gt;Our campaign is powered by volunteers like you. &lt;br&gt;Join us at our next event \u27a1\ufe0f &lt;br&gt;http://marieforcongress.com/volunteer/","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Loving that union-made coffee - and proud to &lt;br&gt;stand in solidarity with the 7707 McKnight Rd &lt;br&gt;@pghsbuxunited workers. #NoContractNoCoffee #PA17&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;I'm proud to have earned the endorsement of the &lt;br&gt;hardworking folks of @SEIUPA . On the campaign &lt;br&gt;trail and in Congress, I'll be fighting to &lt;br&gt;protect the union way of life in western PA. &lt;br&gt;#PA17 #1u&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Great to tour ATI Brackenridge and spend time &lt;br&gt;with the @steelworkers members whose hard work &lt;br&gt;makes the place run. We have the workers and &lt;br&gt;businesses that know how to make stuff right here &lt;br&gt;in western PA, and I'll always fight for our &lt;br&gt;union manufacturing jobs in #PA17.","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;I have officially filed to run for congress in &lt;br&gt;Washington State\u2019s 3rd Congressional District. I &lt;br&gt;am so thankful for everyone who has supported me &lt;br&gt;so far in this race and can\u2019t wait to flip this &lt;br&gt;seat blue. Join me! Endorse: &lt;br&gt;https://forms.gle/5txaRRoU7MxajMw5A Donate: &lt;br&gt;https://secure.actblue.com/donate/mgp?refcode=website&amp;amount=25&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;RT @WABuildingTrade: For our complete list of &lt;br&gt;endorsements, please visit: &lt;br&gt;https://t.co/rOQKSbjtVJ #unionpride #unionproud &lt;br&gt;#wabuildingtr\u2026&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;I\u2019m not taking a dime of corporate PAC money &lt;br&gt;because I work for the people of #WA03 \u2013 not &lt;br&gt;corporate special interests. Dark money has no &lt;br&gt;place in our democracy. Honored to receive the &lt;br&gt;endorsement of @StopBigMoney @LetAmericaVote! &lt;br&gt;https://t.co/PSofVAiX1u","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Honored to be endorsed by @NARAL . You'll always &lt;br&gt;know where I stand on protecting your freedom to &lt;br&gt;choose. And I'm going to fight hard to protect &lt;br&gt;you from extremists like Doug Mastriano and my &lt;br&gt;opponent and their plans to attack your rights.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Honored to have the endorsement of the Pittsburgh &lt;br&gt;Regional Building Trades Council. They support &lt;br&gt;Republicans and Democrats but are proudly &lt;br&gt;standing behind me in #PA17. I will always fight &lt;br&gt;for the backbone of Western PA\u2014our workers, jobs, &lt;br&gt;and unions.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;The difference is clear. I'm proud to be endorsed &lt;br&gt;by @PPact + @NationalNOWPAC and will always &lt;br&gt;defend the right to a safe, legal abortion.","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;In congress, I will take on politicians who try &lt;br&gt;to get between a woman and her doctor and fight &lt;br&gt;to protect abortion rights that the Supreme Court &lt;br&gt;put in jeopardy. It\u2019s more important than ever &lt;br&gt;that mothers like me stand up for our rights.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Senator Murray is a champion for the environment, &lt;br&gt;healthcare, reproductive freedom and has made &lt;br&gt;Washington a better state for working mothers &lt;br&gt;like me. @PattyMurray I am honored to have your &lt;br&gt;endorsement.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Joe Kent is an extremist who will undermine our &lt;br&gt;Constitution and make our communities less safe. &lt;br&gt;I will protect our democracy and support our &lt;br&gt;firefighters, officers, and first responders. &lt;br&gt;Extremism vs. a safe world is the choice in this &lt;br&gt;election.","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Unions lift up all workers and strengthen our &lt;br&gt;democracy. \u270a&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;No politician should tell you how to plan your &lt;br&gt;family. I'm in this race to defend and expand our &lt;br&gt;community's freedom, and that includes keeping &lt;br&gt;right-wing politicians out of the way of women's &lt;br&gt;reproductive decisions. #PA17&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;I joined the @PaDems + local leaders to stand for &lt;br&gt;freedom and choice today. And the contrast with &lt;br&gt;my right-wing opponent couldn\u2019t be starker:","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Last year, I paid more in taxes than Jeff Bezos, &lt;br&gt;the second-richest person in the country \u2013 you &lt;br&gt;read that right. This system doesn\u2019t work for &lt;br&gt;everyday folks. Stand with me to demand a fair &lt;br&gt;tax code that forces the ultra-wealthy to pay &lt;br&gt;their fair share: &lt;br&gt;https://secure.ngpvan.com/rpEHsxFarEiS39oH9kgHlQ2&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Most importantly, 97% of donations came from &lt;br&gt;grassroots donors and 83% of those donations are &lt;br&gt;from my fellow Washingtonians. It\u2019s clear our &lt;br&gt;campaign is for working people, not corporate &lt;br&gt;MAGA interests. Give your support here: &lt;br&gt;https://secure.actblue.com/donate/mgp?refcode=website&amp;amount=25&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Join me this Saturday for our Longview and &lt;br&gt;Vancouver canvass kick-offs! First we\u2019ll rally, &lt;br&gt;next I\u2019ll say a few words, and then we\u2019ll all go &lt;br&gt;knock on doors to tell our neighbors about this &lt;br&gt;important race. RSVP and learn about other &lt;br&gt;upcoming events here: &lt;br&gt;https://mobilize.us/marieforcongress/","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Coraopolis Fall Festival is a great #PA17 &lt;br&gt;community event today\u2014check it out! Always nice &lt;br&gt;to talk to folks in Cory and thanks to Mayor &lt;br&gt;Michael Dixon for being a great host!&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;A good start in PA, but we need federal action &lt;br&gt;too. Whether it's opening up banking for these &lt;br&gt;businesses or making sure doctors can prescribe &lt;br&gt;medical marijuana to my fellow veterans at the &lt;br&gt;VA, we need to legalize cannabis federally to &lt;br&gt;move forward.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;What a great and fired up crowd for tonight's AK &lt;br&gt;Valley meet &amp; greet at Harmar House. Folks all &lt;br&gt;across #PA17 are ready to put in the work to win &lt;br&gt;this thing!","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Joe Kent will vote for a national ban on abortion &lt;br&gt;and the GOP has made clear that this is a top &lt;br&gt;priority if they have the majority in November. &lt;br&gt;Electing pro-choice candidates has never been &lt;br&gt;more important. Join me in this fight!&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;My extremist opponent has released his first TV &lt;br&gt;ad. He wants to: \u274c Ban abortion everywhere \u274c &lt;br&gt;Defund the FBI \u274c Abandon Ukraine \u274c Arrest Dr. &lt;br&gt;Fauci I\u2019ve got exactly 4 weeks to show voters &lt;br&gt;what\u2019s at stake in this election. RT &amp;amp; chip &lt;br&gt;in to help me win \u2935\ufe0f https://t.co/RRm9Oksbin&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;If you\u2019re against abortion AND contraception, &lt;br&gt;you\u2019re an extremist. @HerreraBeutler is out of &lt;br&gt;touch with our district\u2019s values. You can watch &lt;br&gt;my @KGWStraighttalk interview with @LauralPorter &lt;br&gt;online or at 7pm tonight on @KGW","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Shaffer KNOWS his extremist views on choice are &lt;br&gt;hurting him. Now he's lying to save his sinking &lt;br&gt;campaign. The Pro-Life Alliance is 100% behind &lt;br&gt;him, even sending mail to help, bc they know &lt;br&gt;he'll always vote for Life at Conception bills &lt;br&gt;(outlawing abortion, with no exceptions.)&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;he supports a constitutional amendment to ban &lt;br&gt;abortion and is on record wanting rape and incest &lt;br&gt;victims (children even) subject to his abortion &lt;br&gt;bans. Folks in #PA17 want nothing to do with this &lt;br&gt;extremism.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;My right-wing opponent wants to put an abortion &lt;br&gt;ban in the Constitution, and is on the record &lt;br&gt;opposing exceptions for rape and incest victims. &lt;br&gt;He and his pals Dr. Oz &amp; Doug Mastriano are too &lt;br&gt;extreme for #PA17. Let's win this @JoshShapiroPA &lt;br&gt;@JohnFetterman","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;I\u2019m running to unite people from across the aisle &lt;br&gt;to stop inflation, create family wage jobs &amp;amp; &lt;br&gt;bring federal $$ home for vital infrastructure. &lt;br&gt;Joe Kent has focused his campaign on divisive &lt;br&gt;culture wars to distract from his lack of real &lt;br&gt;solutions for the people of #WA03.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;(1/3) We asked, and you answered! The results &lt;br&gt;from our GE Survey are in, and here\u2019s what you &lt;br&gt;chose as your top priorities: -Affordable &lt;br&gt;childcare -Jobs &amp; wage growth -Climate action &lt;br&gt;-Abortion rights -Affordable healthcare &lt;br&gt;-Supporting manufacturing -Supporting small &lt;br&gt;business&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Had a great time talking about American &lt;br&gt;manufacturing, homeownership and fighting for &lt;br&gt;Middle America with Kyle on @KXRONews this &lt;br&gt;morning! Here\u2019s the link to listen:","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Jobs! Jobs! Jobs! Our region is getting huge &lt;br&gt;investments in our AI &amp; robotics sector thanks to &lt;br&gt;the American Rescue Plan (ya know, the bill that &lt;br&gt;every single Republican voted against)&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;We're lucky to have the Beaver Valley Power &lt;br&gt;Station right here in #PA17. It's a major source &lt;br&gt;of both solid union jobs and reliable clean &lt;br&gt;energy powering our region. &lt;br&gt;https://nytimes.com/2022/07/05/bus&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;We're getting millions in healthcare savings and &lt;br&gt;huge investments in western PA manufacturing. &lt;br&gt;This bill is a big win for #PA17.","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;I'm committed to taking the actions necessary to &lt;br&gt;confront the climate crisis. My opponent thinks &lt;br&gt;climate change is a hoax invented by the Chinese &lt;br&gt;government to make money. #WA03 needs a Member of &lt;br&gt;Congress, not an extreme conspiracy theorist. &lt;br&gt;Thank you @LCVoters for your support. &lt;br&gt;https://t.co/BvvQu35wKN","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Patriotism means fighting to bring back jobs &amp; &lt;br&gt;make more stuff here to lower costs. #TeamDeluzio &lt;br&gt;is proud to be the pro-worker &amp; pro-family &lt;br&gt;campaign in #PA17. But I'm sure the greedy &lt;br&gt;corporations gouging us and shipping our jobs &lt;br&gt;overseas love the corporate exec we're up &lt;br&gt;against.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Lousy trade deals + corporate greed have killed &lt;br&gt;our jobs for far too long. We see it here in &lt;br&gt;#PA17, from the Alle-Kiski Valley to Aliquippa, &lt;br&gt;and I'm not going to let our jobs go overseas. We &lt;br&gt;need to bring back manufacturing, make stuff here &lt;br&gt;again, and do it with union workers.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;More of this. \ud83c\uddfa\ud83c\uddf8 We're in the economic fight of &lt;br&gt;our lives with China, and it's time we start &lt;br&gt;taking our jobs back starting right here in &lt;br&gt;#PA17.","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;I don't take corporate PAC money. I rely on &lt;br&gt;grassroots contributors to fund this campaign. &lt;br&gt;Please chip in now to keep our new TV ad on the &lt;br&gt;air, defeat Joe Kent, and flip #WA03 blue. &lt;br&gt;https://t.co/iQsEmAa5kB&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Ask any economist, this kind of market &lt;br&gt;consolidation is dangerous for consumers &lt;br&gt;everywhere, it just happens that these &lt;br&gt;\u201cconsumers\u201d are our most vulnerable children.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Last year, I paid more in taxes than Jeff Bezos, &lt;br&gt;the second-richest person in the country \u2013 you &lt;br&gt;read that right. This system doesn\u2019t work for &lt;br&gt;everyday folks. Stand with me to demand a fair &lt;br&gt;tax code that forces the ultra-wealthy to pay &lt;br&gt;their fair share: &lt;br&gt;https://secure.ngpvan.com/rpEHsxFarEiS39oH9kgHlQ2","&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Want to know why your burger has gotten so &lt;br&gt;expensive? Four giant meatpacking companies &lt;br&gt;control 85% of the market, and they're setting &lt;br&gt;prices, ripping us off, and making fat profits. &lt;br&gt;[1/2]&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;We should make stuff here again. Make it with &lt;br&gt;union workers, grow the middle class, and give &lt;br&gt;small businesses a chance against corporate &lt;br&gt;giants who are gouging us.&lt;br&gt;&lt;br&gt;&lt;strong&gt;Ex. Tweet:&lt;/strong&gt;&lt;br&gt;Our biggest fundraising deadline of the year is &lt;br&gt;midnight tonight \u2014 can you chip in and help us &lt;br&gt;beat the huge corporate jag-offs? &lt;br&gt;https://secure.actblue.com/donate/chris-deluzio-for-congress-social"],"shape":[20],"dtype":"object","order":"little"}],["Quoted_Topic",{"type":"ndarray","array":["\"extreme\"","\"extreme\"","\"volunteer\"","\"volunteer\"","\"unions\"","\"unions\"","\"endorsement\"","\"endorsement\"","\"protect\"","\"protect\"","\"folks\"","\"folks\"","\"abortion\"","\"abortion\"","\"manufacture\"","\"manufacture\"","\"china\"","\"china\"","\"corporations\"","\"corporations\""],"shape":[20],"dtype":"object","order":"little"}],["factors",{"type":"ndarray","array":[["\"extreme\"","MGP"],["\"extreme\"","Deluzio"],["\"volunteer\"","MGP"],["\"volunteer\"","Deluzio"],["\"unions\"","MGP"],["\"unions\"","Deluzio"],["\"endorsement\"","MGP"],["\"endorsement\"","Deluzio"],["\"protect\"","MGP"],["\"protect\"","Deluzio"],["\"folks\"","MGP"],["\"folks\"","Deluzio"],["\"abortion\"","MGP"],["\"abortion\"","Deluzio"],["\"manufacture\"","MGP"],["\"manufacture\"","Deluzio"],["\"china\"","MGP"],["\"china\"","Deluzio"],["\"corporations\"","MGP"],["\"corporations\"","Deluzio"]],"shape":[20],"dtype":"object","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p1201","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p1202"}}},"glyph":{"type":"object","name":"VBar","id":"p1197","attributes":{"x":{"type":"field","field":"factors"},"width":{"type":"value","value":0.9},"top":{"type":"field","field":"Proportion"},"line_color":{"type":"value","value":"#1f77b4"},"fill_color":{"type":"field","field":"Candidate","transform":{"type":"object","name":"CategoricalColorMapper","id":"p1193","attributes":{"palette":["#008080","#7BAFD4"],"factors":["MGP","Deluzio"]}}}}},"nonselection_glyph":{"type":"object","name":"VBar","id":"p1198","attributes":{"x":{"type":"field","field":"factors"},"width":{"type":"value","value":0.9},"top":{"type":"field","field":"Proportion"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"field","field":"Candidate","transform":{"id":"p1193"}},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"VBar","id":"p1199","attributes":{"x":{"type":"field","field":"factors"},"width":{"type":"value","value":0.9},"top":{"type":"field","field":"Proportion"},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"field","field":"Candidate","transform":{"id":"p1193"}},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p1180","attributes":{"tools":[{"type":"object","name":"HoverTool","id":"p1205","attributes":{"renderers":"auto","tooltips":[["Candidate","@Candidate"],["Topic","@Quoted_Topic"],["Proportion","@Proportion{0.0%}"],["Words Used in Search","@Semantically_Similar_Words{safe}"],["Sample Tweets","@Sample_Tweets{safe}"]]}}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p1188","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p1189","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"CustomJSTickFormatter","id":"p1206","attributes":{"code":"return (tick*100).toFixed(0) + '%'"}},"major_label_policy":{"type":"object","name":"AllLabels","id":"p1191"}}}],"below":[{"type":"object","name":"CategoricalAxis","id":"p1183","attributes":{"ticker":{"type":"object","name":"CategoricalTicker","id":"p1184"},"formatter":{"type":"object","name":"CategoricalTickFormatter","id":"p1185"},"major_label_orientation":1,"major_label_policy":{"type":"object","name":"AllLabels","id":"p1186"},"major_label_text_font_size":"12pt"}}],"center":[{"type":"object","name":"Grid","id":"p1187","attributes":{"axis":{"id":"p1183"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p1192","attributes":{"dimension":1,"axis":{"id":"p1188"}}},{"type":"object","name":"Legend","id":"p1203","attributes":{"items":[{"type":"object","name":"LegendItem","id":"p1204","attributes":{"label":{"type":"field","field":"Candidate"},"renderers":[{"id":"p1200"}]}}]}}]}}]}}
    </script>
    <script type="text/javascript">
      (function() {
        const fn = function() {
          Bokeh.safely(function() {
            (function(root) {
              function embed_document(root) {
              const docs_json = document.getElementById('d21c16d9-0a79-4e96-9a3e-52413aabc49b').textContent;
              const render_items = [{"docid":"58198d52-01a3-4336-9311-35f3487b3d7e","roots":{"p1171":"cc7b3220-e3d3-416a-a1d2-4474519bbab1"},"root_ids":["p1171"]}];
              root.Bokeh.embed.embed_items(docs_json, render_items);
              }
              if (root.Bokeh !== undefined) {
                embed_document(root);
              } else {
                let attempts = 0;
                const timer = setInterval(function(root) {
                  if (root.Bokeh !== undefined) {
                    clearInterval(timer);
                    embed_document(root);
                  } else {
                    attempts++;
                    if (attempts > 100) {
                      clearInterval(timer);
                      console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                    }
                  }
                }, 10, root)
              }
            })(window);
          });
        };
        if (document.readyState != "loading") fn();
        else document.addEventListener("DOMContentLoaded", fn);
      })();
    </script>
  </body>
</html>
</div>


# Insights and Conclusions

## 1. Abortion
In the wake of the *Dobbs v. Jackson Women's Health* decision, abortion rights became a huge topic in the 2022 midterms. The results above show that nearly 10% of all tweets from **both** Marie Gluesenkamp Perez and Chris Deluzio touched on abortion rights and reproductive health generally.

#### Messaging Differences:
- **Marie Gluesenkamp Perez:** Her tweets on this topic usually offered a more personal perspective to connect with voters in Washington's 3rd.
  > "Like many moms, I've suffered through the heartbreak of miscarriage – imagine the horror of compounding that with being thrown in JAIL. Mothers deserve autonomy, not a police state."
- **Chris Deluzio:** Deluzio's messaging on the topic focused more on the broader themes of rights and freedoms.  
  > "I think you should have the right to make your own decisions about your pregnancy and health care, and I'll vote in Congress to protect abortion rights."

Despite the low risk of losing abortion access in Washington and Pennsylvania, nearly 1 in 10 tweets from both campaigns touched on the topic. It would be interesting to observe a candidate from a state which increased restrictions, but overall, whether through personal stories or broader rights discussions, abortion was a central topic to these campaigns and the 2022 midterms overall.

## 2. Extremist Opponent

Both campaigns hammered the narrative of "extremism". Marie Gluesenkamp Perez (12% of all tweets) did so slightly more than Chris Deluzio (9.5%), but that is probably due to the fount of source material, given her opponent, Joe Kent's, genuinely insane and extreme positions:

> "Joe Kent says the attack on #January6th 'reeks of an intelligence operation' done by the police. Even today he continues defending the violent mob that ransacked the Capitol. [Link](https://t.co/AyP0ipQEqW)"

> "I'm committed to taking the actions necessary to confront the climate crisis. My opponent thinks climate change is a hoax invented by the Chinese government to make money. #WA03 needs a Member of Congress, not an extreme conspiracy theorist. [Link](https://t.co/BvvQu35wKN)"

> "Joe Kent’s QAnon rants are desperate, weird, and do nothing to improve the lives of people in our district. Anyone who’s tired of this is welcome to join my campaign. I’m running because Congress could use someone who actually knows how to fix things. [Link](https://t.co/hrMAnVXytH)"

Chris Deluzio, running against his moderate opponent with less controversial views, incorporated 'extremism' in a brilliant dual-pronged strategy. First, he highlighted Jeremy Shaffer's silence on things like January6, to imply tacit consent. Shaffer couldn't denounce the extreme views of the far right without alienating those base voters, so Deluzio's campaign used this silence to associate Shaffer with broader extremism effectively:

> "Jeremy Shaffer refuses to denounce the radical right's attack on our elections. His silence speaks volumes."

> "Jeremy, why do you refuse to denounce the insurrection? Why won’t you denounce the assault on our democracy???"

Further, Deluzio frequently connected Shaffer to other more extreme political figures, to paint him with the same brush:

> "My opponent campaigns alongside extremists like Doug Mastriano and Kevin McCarthy, who tried to overthrow our democracy."

> "Jeremy Shaffer just opened a joint campaign office with Doug Mastriano. These extremists are a threat to our freedom."
> Jeremy Shaffer is showing you exactly who he is: Campaigning with insurrectionists, courting endorsements from extremists, and begging formoney from the radical right.'

This method of linking Shaffer with known extremists, despite his moderate stances, and highlighting his silence on extreme issues was a great strategy in the political context of 2022.


## 3. Unions vs. Corporations


Deluzio's campaign emphasized unions and criticized corporate outsourcing, aligning with his district's industrial heritage and union-heavy electorate. He connected his opponent to corporate interests and used the "China" narrative to highlight the need for domestic manufacturing.  

> "Corporate execs have been stiffing folks, crushing unions, & outsourcing jobs to China & all over the planet for way too long. [1/2]"

> "'Corporate executive Jeremy Shaffer really can't stand to be asked about his business in China (or Saudi Arabia!). Corporations like the one that made him rich are raking in millions building up China's infrastructure & selling out folks from #PA17. Need proof? Here:'"

This strategy was creative because recently, China as a cudgel has been used mostly by Republicans. Take a similar district like IN-05, Indianapolis suburbs, where Republican Rep. Victoria Spartz is associating her Republican primary opponent as "China Chuck."


[![China Chuck](images/china_chuck.png)](https://s3.amazonaws.com/pdfweb/videos/da49429b-f004-4a65-bd35-48ccfd8ebe54.mp4)

## 4. Ground Game

Marie Gluesenkamp Perez's campaign prioritized ground efforts, with 1 in every 4 of her tweets promoting volunteering, canvassing, and fundraising. She often mentioned Vancouver, the largest metro area in her district, underscoring her strategy to maximize urban turnout and minimize rural conservative opposition. Since so much of her district is rural, solidifying turnout in metro/suburban areas was critical

>'Join me this Saturday for our Longview and <br>Vancouver canvass kick-offs! First we’ll rally, next I’ll say a few words, and then we’ll all go knock on doors to tell our neighbors about this <br>important race. RSVP and learn about other upcoming events here: <br>https://mobilize.us/marieforcongress/',
 
>'Hello from Pacific County, where we’re having fun in the rain at the South Bend Labor Day parade! Our campaign is powered by volunteers like you. Join us at our next event ➡️ <br>http://marieforcongress.com/volunteer/',


**Summary:** Both campaigns effectively leveraged the national issues of 2022, such as abortion rights and anti-extremism, in their strategies. Deluzio capitalized on his district-specific issues, particularly unions, to resonate with his electorate. In contrast, MGP used more of her messaging capital on structuring her ground game, using her voice and reach to mobilize volunteers and supporters to turn out the right votes in the right places. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!---## ACKNOWLEDGMENTS --
-My partner Felipe who helped me manually copy and paste tweets and format them for hours
-Not Paul Kim
