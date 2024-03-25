  
<h2 align="center">           </h2>
<a name="readme-top"></a>
<h2 align="center">Winning in Trump Country</h2>


<div align="center">
  <a href="https://github.com/samforwill/2024Strategies">
    <img src="images/twitterpolitics.png" alt="tweeters chirping" style="width: 70%; max-width: 900px;">
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
          <ul>
        <li><a href="#tweet-collection">Tweet Collection</a></li>
              </ul>
    <li><a href="#114th-congress-tweet-sentiment-classification">114th Congress Tweet Sentiment Classification</a></li>
    <li><a href="#classifying-mgp-and-deluzio-tweets">Classifying MGP and Deluzio Tweets</a></li>
                  <!--ul>
         <li><a href="#regressors-and-analysis-notebook">Regressors and Analysis Notebook</a></li>
              </ul> -->
   <ul> <li><a href="#classifying-MGP-&-Deluzio-Tweets-via-114th-Congress-Model">
   Classifying MGP & Deluzio Tweets via 114th Congress Model</a></li></ul>
      <ul> <li><a href="#unsupervised-topic-modeling">Unsupervised Topic Modeling</a></li></ul>
                        <!-- <ul>
        <li><a href="#top-democratic-features-for-predicting-pvi">Top Democratic Features for Predicting PVI</a></li>
        <li><a href="#top-republican-features-for-predicting-pvi">Top Republican Features for Predicting PVI</a></li>
        <li><a href="#top-2022-midterm-democratic-features">Top 2022 Midterm Democratic Features</a></li>
        <li><a href="#top-2022-midterm-republican-features">Top 2022 Midterm Republican Features</a></li>
                      </ul> -->
    <li><a href="#recommendations-and-conclusions">Recommendations and Conclusions</a></li>
    <li><a href="#future-work">Future Work</a></li>
  </ol>
  <!--li><a href="#acknowledgments">Acknowledgments</a></li>
<!--/details-->


## Introduction

 This project applies Natural Language Processing (NLP) to analyze the twitter messaging strategies of Marie Glusenkamp Perez (WA-03) and Chris Deluzio (PA-17), Democratic newcomers competing in two of the most challenging districts for Democrats in 2022 midterm cycle.<br />
 
 Given the 2022 midterms were marked by the defeats of many election deniers and January 6th apologists, a secondary focus of this study is to assess the difference in our candidates' messaging strategies against distinct types of opponents— one faced Joe Kent in WA, a 'Kooky' nominee who fully embraced the 2020 election conspiracies, and the other faced Jeremy Shaffer in PA, a mainstream Republican who acknowledged Joe Biden's 2020 victory. 
 
## Methodology
 
 First, I trained classification models on a dataset of 5000 Twitter/Facebook posts of members of the 114th Congress to characterize each message's: bias (neutral vs. partisan), nature of the message (e.g., informational, personal, policy, mobilization, attack), and political affiliation of the author (Democrat or Republican).
 
 I then used the best performing trained models to classify my two candidates' tweets over a 6-month period immediately leading up to the 2022 midterm elections. This classification helped me analyze overall digital strategies, and also allowed me to compare Marie Glusenkamp Perez and Chris Deluzio's messaging against each other.

Finally, I used unsupervised Topic Modeling to determine and compare the predominant themes in the candidates’ tweetsets. The topic model analyzed the collection of peak campaign season tweets for MGP and Deluzio to find patterns based on word frequency, order, distance, and meaning and then group various items into relevant categories. 




<!--
Initial analysis involved unsupervised learning Topic Modeling on a corpus of our candidates' tweets from the six month period leading up to election day on November 8, 2022, and the predominant campaign themes were assessed and compared between the two candidates. <br />

Subsequently, classification models were trained on a dataset of 5000 tweets and Facebook posts from the 114th Congress, annotated with each message's bias (neutral vs. partisan), political affiliation of the author (Democrat or Republican), and the content of the message (e.g., informational, an attack on another candidate, critiques). intent, partisanship, and target audience and augmented with political party data, was subjected to classification. Models were developed to differentiate messages based on bias (neutral/bipartisan vs. partisan), the author's political affiliation, and the nature of the content (e.g., informational, personal, policy, mobilization, attack). <br />

Finally, I used the best performing models to classify the candidates' 6-month campaign messaging tweetset based on the criteria established by the 114th Congress dataset. This classification helped me analyze overall campaign strategies, but also allowed me to compare Marie Glusenkamp Perez and Chris Deluzio's messaging against each other and against the broader dataset.

The purpose of this project is to utilize Natural Language Processing (NLP) to dissect the digital campaign strategies of Marie Glusenkamp Perez (WA-03) and Chris Deluzio (PA-17), two Democratic newcomers who each overperformed the partisan lean of their districts to win in the 2022 midterm elections. By analyzing their tweets from a crucial six-month pre-election period, the study identifies key messaging themes using Topic Modeling. It further extends to classify a dataset of 5000 tweets and Facebook posts from the 114th Congress, annotated for message intent, partisanship, and audience, with an added dimension of political affiliation.

The research trains models to categorize communications by bias (neutral/bipartisan vs. partisan), author's party affiliation, and message type (informational, announcements, attacks, etc.). The most effective models are then applied to the candidates' tweets to decode their campaign messaging, offering insights into their strategies and broader political communication trends.


I determined which candidates to focus on through comparing margins of victory in the 2022 midterm elections to their district's Partisan Voter Index scores (PVI). For my 2 selected candidates of focus, I created a corpus of their tweets going back 6 months before election day, November 8th, 2022, covering the heart of their campaign season.<br />
 
Next, I used Natural Language Processing (NLP) methods to train models to classify a 5000-tweet/facebook post dataset from members of the 114th Congress. The messaging dataset was addended with human judgments about the purpose, partisanship, and audience of the messages, and I feature engineered party onto this dataset, as well. Natural Language Processing methods were used to train models to classify on the previous m<br />


I hope that my analysis and insights can help inform Democratic strategy to help win back the US house and keep it for the decade ahead. -->
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Data Used
1. PVI score data was sourced from the [Cook Political Report](https://www.cookpolitical.com/cook-pvi/2023-partisan-voting-index/118-district-map-and-list).
2. 2022 Midterm Results were sourced from [The Daily Kos](https://www.dailykos.com/stories/2022/9/21/1742660/-The-ultimate-Daily-Kos-Elections-guide-to-all-of-our-data-sets).
3. The campaign tweets from Marie Glusenkamp Perez and Chris Deluzio were hand-copied from their twitter accounts [@MGPforCongress](https://twitter.com/mgpforcongress) and [@ChrisforPA](https://twitter.com/chrisforPA)

4. The 114th Congress tweets addended with characterization inputs was sourced from Crowdflower's Data For Everyone Library via [Kaggle](https://www.kaggle.com/datasets/crowdflower/political-social-media-posts/data).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Selecting the Candidates

I determined which candidates to focus on through comparing their 2022 electoral margins with their district's Partisan Voter Index scores (PVI). PVI measures how partisan the district is compared to the nation as a whole, based on how the constituents of those districts voted in previous presidential elections. <br />

To identify standout candidates, I devised a 'Performance' metric by calculating the difference between each district's Partisan Voter Index (PVI) and the candidate's electoral margin in 2022. This approach highlighted those who significantly outperformed their district's typical partisan alignment.


![Overperformance](images/Overperformance.png)

Next, my focus shifted to first-time candidates to avoid the complexities of incumbency. Mary Peltola from Alaska was also excluded due to the state's use of Ranked Choice Voting, which, while innovative, complicates direct comparison in this context. <br />

That left me with 6 candidates to consider, all having overperformed their districts' partisan lean by at least 5 points.  The following 4 candidates greatly overperformed in their districts, but were eliminated from consideration for various reasons:
![Candidates](images/Candidates.png)
Most of these candidates were ruled out due to their opposition by 'Kooky/Extreme' candidates or the deletion of their campaign's Twitter accounts post-midterms.
Adam Frisch, who narrowly missed victory in CO-03, was initially a candidate of interest. However, his exclusion was due to the sheer volume of his tweets, which,thanks to Elon Musk's recent termination of free API access by Twitter, made data collection too labor-intensive. The next deepest red district to pull out the win was...

![MGP](images/MGP.png)

Marie Glusenkamp Perez faced cuckoo Joe Kent, who expressed some extreme views like supporting the arrest of Dr. Anthony Fauci and endorsing the claims of a stolen 2020 election. In fact, he became the candidate for WA-03 after successfully primarying the serving Republican Congressperson, Jaime Herrera Beutler, one of only 10 republicans who voted to impeach Donald Trump following the events of January 6th.<br />


The next candidate I wanted to assess took a little more research to come to a decision, but I wanted to find a Democrat who overperformed in his district, while contending against an opponent who was a more mainstream. I landed on...

![Deluzio](images/Deluzio.png)

Chris Deluzio, competing in a toss-up district, significantly outperformed against Jeremy Shaffer, who notably sidestepped affirming or denying the 2020 election fraud claims, and even released an ad promising to "protect women's healthcare" <br />

### Tweet Collection
As mentioned before, the termination of free API access meant manually compiling tweets for Chris Deluzio and Marie Glusenkamp Perez, and then using a custom parsing script to organize and format these tweets into a structured dataset for analysis. Tweets were manually copied, separated by a '|' delimiter, and then organized into a corpus of around 900 tweets per candidate. [candidate notebook](MGP and Delozio.ipynb).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 114th Congress Tweet Sentiment Classification

So the first thing I notice in observing this tweetset of 5000 tweets from 114th Congressmembers, is that, while each tweet is labeled with "partisan" or "neutral", these is no indication of WHICH party the member belongs to. It literally has the following format:

| label | 
| :----------- | 
| From: Mitch McConnell (Senator from Kentucky) |
| From: Kurt Schrader (Representative from Oregon) |
| From: Michael Crapo (Senator from Idaho) |

So in order to assign each tweet with the political affiliation of each tweeter, I copied all of the text on this [C-span Summary Page including pictures](https://www.c-span.org/congress/members/?chamber=house&congress=114&all) of every member of the 114th Congress, and then developed some code to parse through every member, delete suffixes, and deal with duplicated member last names (like Rob Bishop UT-01, and Mike Bishop MI-08) and assign their political party. 


The dataset of 5000 tweets from 114th Congress members immediately presented a challenge: each tweet was tagged as "partisan" or "neutral," but provided no information on the political party of the tweeter. The data was presented like this:

| label |
| :----------- |
| From: Mitch McConnell (Senator from Kentucky) |
| From: Kurt Schrader (Representative from Oregon) |
| From: Michael Crapo (Senator from Idaho) |

To address this, I used this comprehensive member list from the [C-span 114th Summary Page](https://www.c-span.org/congress/members/?chamber=house&congress=114&all), including images, to match members' names and extract their partisan affiliation. The parsing code developed involved removal of name suffixes and resolving ambiguities in cases of shared last names, such as distinguishing between Rob Bishop (UT-01) and Mike Bishop (MI-08).<br />


After the data was addended to include each tweeter's political affilitation, several machine learning models were trained to classify tweets based on partisanship and content. This process encorporated Natural Language Processing (NLP) techniques, including TF-IDF vectorization for feature extraction and the application of multiple classifiers such as RandomForest, Naive Bayes, SVM, and Neural Networks within a pipeline structure optimized through GridSearchCV for hyperparameter tuning.<br />

Accuracy was significantly improved after integrating Word2Vec embeddings, enhancing the models' ability to capture semantic nuances within the text data. The final models were decent at discerning party affiliation, distinguishing between neutral and partisan messages, and categorizing the underlying message themes.<br />


The optimal models for each classification target—Party, Bias, and Message—were determined based on F1 scores and accuracy. Below are the best-performing models for each category, along with their respective confusion matrices:

- **Bias Prediction**: The Gaussian Naive Bayes Model emerged as the top performer for distinguishing between partisan and neutral tweets.
- **Party Affiliation Prediction**: The Multi-Layer Perceptron Classifier, a type of basic feedforward artificial neural network, was most effective in identifying the political party.
- **Message Category Prediction**: Gaussian Naive Bayes also proved to be the best model for categorizing the content of the tweets.

![bias_and_party](images/bias_and_party.png)

![message](images/message.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>








<!--
Well, that certainly doesn’t look normal! (in the statistical distribution sense of “normal”, but also in the common sense department). So let's dive in! <br />

The median district in the United States is around -1, which is a Republican +1 district, meaning half of the districts in the US are more Republican and half are more Democratic than that point. Pretty close to 0, so I’m not mad at that.<br />


But, right off the bat, we see a huge imbalance and large concentration of districts between the R+10 to R+20 range,(-10 to -20 in my conversion). For context,  Cook considers everything beyond the 10-point range to be non-competitive “Solid” districts.

Of course, my initial thought on seeing this is:

> **"Who could possibly be responsible for creating this huge concentration of districts that are *just* out of competitive reach? 🤔 And why is the answer almost certainly 'Republicans gerrymandering'🤨???**

So, I set about to confirm my priors. The Brennan Center for Justice [broke down ](https://www.brennancenter.org/our-work/research-reports/who-controlled-redistricting-every-state)redistricting in every state into the following categories: 
* GOP-Controlled Redistricting (177 seats)
* Democratic-Controlled Redistricting (49 seats)
* Split-Control (2 seats)
* Court-Ordered Maps (91 seats)
* Independent Commissions [non-partisan] (82 seats)
* Political Commissions [partisan appointees from both parties](28 seats)
* and At-Large Districts (6 seats/states)

Now, when we look at the same distribution graph color-coded by type of redistricting, I have a feeling we should see something pretty notable in that -10 to -20 range.

![Redistricting Type Composite](images/Redistricting_Type_Composite.png)
<p align="center">
  <img src="images/surprise.gif" alt="surprise" />
</p>

### It's a little hard to focus with all those colors going on in the same graph, so let's break it down into its component parts:

![Facet Grid](images/FacetGrid_PVI.png)

### Observations:
**Courts & Commissions**: As far as distribution goes, courts and commissions have the most natural spread, which makes sense given their priority to create fair districts. The median of the commissions is between 6-8, but this also makes sense given that states with commissions tend to be more democratic overall (CA, CO, MI, AZ, HI, ID, MO, NJ, WA)<br />

**GOP-Controlled Redistricting**: Republicans had the opportunity to draw the district lines in an astonishing 41% of all seats in Congress (177 total). As awful as that is for democracy and discourse overall, it makes it pretty easy to visually see the manipulation of district-drawing to create electoral advantages. 
- The median district is R+12, so, solidly safe districts for Republicans, but even more interesting is the immediate dropoff of districts exceeding R+20, indicating **"cracking"**, where, once a Republican district is safe enough, they can crack into more democratic areas like urban centers.
- And then notice the second hump of GOP-drawn democratic districts in the D+20 region, indicating **"packing"**, where democratic voters are packed into one district to dilute their voting power <br />

**Democratic-Controlled Redistricting**: The median district is D+7, within the competitive zone. Perhaps Democrats would see more GOP-type drawing behavior if we had more opportunitis to hold the pen, but at 49 seats total, our ability to push back on the GOP's advantage is severely limited.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Heatmapping
Because I only eliminated in the data wrangling/cleaning process those features that were *exactly* duplicated within and between data profiles, I wanted to get a sense of the extent of multicollinearity of my features and chose to do that through heatmapping.<br />

Since my data also has high dimensionality (450 unique features), I will only share here one of my heatmaps corresponding to the demographic profile highlighting highly correlated features. 
![Demo Heatmap](images/demo_heatmap_hc.png)
Here it's easy to see many of the features are highly correlated, and many of those relationships make sense intuitively. So let's look at only the top 50 highly correlated pairs of features in the graph below (open in new window, Census Bureau characteristic titles can be very long):
![Top50 Correlations](images/top_50_correlations.png)

Some of the features have an almost exact negative correlation, such as "Place of Birth - Foreign Born" and "Place of Birth - Native Born", but there are also some highly correlated pairs that don't exactly hit perfect correlation such as "Total Population" and "Total Population 1 year and over".

#### EDA Notebook
To take a more detailed look at the EDA process, especially a deeper look into all the heatmapping, follow along in my [EDA Notebook](https://github.com/samforwill/District-Insights/blob/main/02_Exploratory_Data_Analysis.ipynb).


**EDA Conclusions**:
1) The distribution of districts is skewed by Republican gerrymandering.
2) My data has high dimensionality (450 features), high multicollinearity (closely related features), and low observations (only 435 districts in the U.S).
3) To handle the high dimensionality and multicollinearity of my data, I am choosing to focus on regressors that are adept at handling these challenges. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Models and Analysis
### Regressors <br />
Given the high dimensionality and multicolinearity of my data, I chose to only focus on regressors that could handle these challenges, such as L1 and L2 regularization regressors, and Ensemble Learning models. <br />
    
Although I had some decent accuracy with Neural Networks and Principal Component Regression, eventually I sacked those methods because it's not possible to extract the features and associated weights for analysis. 

Models Assessed:
- Ridge
- Lasso
- ElasticNet
- RandomForestRegressor
- ExtraTreesRegressor
- GradientBoostingRegressor

1. **Modeling on PVI**: <br />
After several rounds of parameter tuning on all of the above models, I sorted on highest performance R²-Score on the test set. In all instances, the top 5 performing models were Ridge Regression, signaling its efficacy in managing multicollinearity within the high dimensional demographic feature set.<br />
![PVI Models](images/PVI_models.png)

Specifically, Ridge with MinMax Scaling and L2-regularization alpha of 1 performed best. <br />

The leading model also has the lowest RMSE at 4.365, meaning our model can predict the PVI of a district within ±4.365 points. That's pretty great! <br />

The collective performance of these models underscores Ridge Regression's suitability for demographic-based PVI score prediction, with significant implications for targeted campaign strategies.

2. **Modeling on 2022 Midterm Margins**: <br />
Using the same process and models as in the PVI modeling, after several iterations of parameter tuning, my highest performing models were:<br />
![22 Models](images/22_Models.png) <br />

Although ElasticNet with MinMax Scaler had the highest R²-Score on the test set, I chose to use the Lasso with MinMax scaler alpha 0.1 as my best model based on the other metrics. <br />

The ElasticNet model seems to be significantly overfitting on its training data, while the Lasso model shows more consistency between Mean CV Score, R2 Test Score, and R-squared Training data. Also, the difference in my most important metric (RMSE) is negligible. <br />

### Analysis: 

The R²-Scores on the test set for predicting midterm voter behavior were much worse than for predicting on PVI, and further, the RMSE for midterm margin came in at 10.8, meaning that with this model we could only predict the marginal outcome of the race within ±11 points! <br />

Surely there are many reasons for these models' poor predictive performance in the midterms, but I think there are a few main reasons: <br />

First, participation in midterm elections is much lower than in Presidential years, giving way to large fluctuations in results across the U.S. in expected outcomes. <br /> 

Second, acknowledging that there are fewer split-ticket voters than there used to be, it still exists; PVI is a reflection of how a district voted for the president, *not necessarily* how they voted for their congressional representative. <br />

Finally, all politics is local! Candidate quality matters, especially in midterm cycles. Check out the local situation in every district in the U.S. on [my streamlit app](https://2022midterms.streamlit.app).
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results
**PVI**: It turned out that only modeling on the demographic features of a congressional district proved remarkably accurate in predicting its partisan lean, PVI. Without considering ground game tactics like GOTV efforts, percent eligible voters registered, etc. it turns out that demographics alone can account within ±4.365 points for previous voting behavior in the past 2 presidential cycles (which PVI is a measure of). <br />

**2022 Results**: However, demographics were far less predictive of midterm voter behavior, only coming within ±10.8 points of predicted margins in the 2022 congressional matchups. <br />

But despite the mediocre performance of my model for predicting midterm voter behavior, my overall goal was to find insights into the most important demographic features that were influential in 2022. On that metric, the features my model identified as most influential should speak to and help identify overall midterm trends. <br />

Let's take a look at some of these features:
### Top Democratic Features for Predicting PVI
![Dem Features PVI](images/Dem_Features_PVI.png)

### Top Republican Features for Predicting PVI
![Rep Features PVI](images/Rep_Features_PVI.png)

### Notes on PVI Features

**Things we already knew about Democrats:**
The #1 indicator of previous Democratic presidential voting behavior was "Race - One Race - Black or African American"<br />

**Other Features We Expected**: "Industry: Professional, Scientific, and Management", "Female Householder", "Educational Attainment: Graduate or Professional Degree"<br />

**Surprising Features**: "Ancestry: Swiss" - who knew 🤷? & "House Heating Fuel: Wood"<br />

Though these features may seem surprising at first glance, what it likely reflects is a very Democratic district with an outsized portion of the population exhibiting that characteristic (Looking at you, Vermont, with all that wood-burning liberalism)<br />

**Things we already knew about Republicans:**
The #1 indicator of previous Republican presidential voting behavior was "Race - One Race - White"<br />

**Other Features We Expected**: "Industry: Agriculture, Forestry, Fishing, Hunting, and Mining", "Commuting to Work - Car Truck or Van - Drove Alone" (forget carpooling, libs!), "Educational Attainment: High School Graduate"<br />

I honestly don't find too much surprising about about this feature-set, but I do find it a little funny that one of the big indicators of Republican voting behavior is "Race - Two or More Races - White and American Indian". Anecdotally, I grew up in an extremely Republican and "country" Southern household and my dad always said we were part Cherokee somewhere down the line... According to 23&Me, this was in no way true. <br />



### Top 2022 Midterm Democratic Features
**Most Influential Democratic Features for Predicting 2022 Midterm Voting Behavior**
![Dem Features 22](images/Dem_Features_22.png)

### Top 2022 Midterm Republican Features
**Most Influential Republican Features for Predicting 2022 Midterm Voting Behavior**
![Rep Features 22](images/Rep_Features_22.png)

### Notes on 2022 Midterm Features <br />

By the very nature of how Lasso Regression works, L1-regularization combats overfitting by shrinking the parameters towards 0, eliminating altogether most features, especially in a high-dimensional training set like mine. So the results we see for the midterm characterization features are only the most significant features to the model. <br />

Observing the changes in feature-importance between PVI (previous two presidential cycles) and 2022 midterm behavior, some significant results can be gleaned, which I will go over in the next section.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Recommendations and Conclusions

#### There are some extremely significant results that could impact Democratic strategy going into 2024: <br />

1. "Female Householder, No Spouse Present" shot up to the 2nd-most significant factor in predicting midterm Democratic Voting Behavior. I believe this highlights the impact of the *Dobbs* decision in activating turnout amongst women. <br />
2. "Race - One Race - Black or African American" dropped significantly from the #1 slot in predicting PVI, to only the 9th-most significant factor in predicting 2022 midterm voter behavior (!!!). This could be due to several reasons, like the instability in predicting 2022 Congressional results, lower turnout in midterm years, or higher activation of other voters, e.g. Female NSP in the wake of *Dobbs*. However, it is worth putting extra attention into this heading into the next cycle. <br />
3. "Married-Couple Households" were the 2nd most important feature in predicting midterm Republican-voting behavior. Considering the Alabama Supreme Court ruling involving In Vitro Fertilization, this presents a significant opportunity to pry away some of these voters. 42% of U.S. adults have had or know someone who has undergone fertility treatment [(Pew)](https://www.pewresearch.org/short-reads/2023/09/14/a-growing-share-of-americans-say-theyve-had-fertility-treatments-or-know-someone-who-has/), and although Alabama has tried to walk back the damage in the IVF-ruling, other states continue to advance fetal personhood bills across the country. <br />
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Future Work
* **2022 ACS Results**: When the Census Bureau releases their 2022 ACS Results, I will be able to wrangle the data again to see any changes in feature weights, but also get a glimpse into how districts are changing year-over-year.<br />

 This can provide important strategic insight to congressional candidates when considering characteristics of the district that are increasing or decreasing. 

* **State Legislative Geographies**: The ACS demographic characteristics for state legislative districts have already been released. This could potentially drive huge increases in performance to my modeling given the low number of observations in my current dataset. While there are only 435 U.S. Congressional Districts, there are thousands of state legislative districts. <br />

The only challenge here will be the tedious process of tracking down every election result in decentralized State election result reporting systems.

* **Expand Feature Engineering to Enhance Predictive Performance**: While I only focused on purely demographic features when modeling in this project, there are tons of ways to feature engineer information into the dataset that we already know. Such things as: Regionality, Redistricting Control, Likely Voters, Previous Turnout Metrics, and Using PVI as a Feature for the Midterm Predictions.<br />

(I actually already did that last one, and PVI as a feature greatly enhanced the predictive ability of the models for the midterms. However, it overpowered the demographic features by a lot and I wanted to focus my analysis solely on "demographics as destiny").
<p align="right">(<a href="#readme-top">back to top</a>)</p>
