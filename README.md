

```python
# Observations:
# 1. At least 40 out of 100 tweets shows no sentiment.
# 2. CNN overall polarity is higher than the other Media outlets.
# 3. BBC World News and New York Times overall sentiment is slightly negative whereas CBS News, CNN, Fox News show is slightly 
#    positive
```


```python
# Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tweepy
import time
import seaborn as sns

# Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = 'PVxOe2UtKZJqWM923NhsBmzdH'
consumer_secret = 'mYfJ9uw8qUReMqnD1dMyP94djCNX83Y9vb0Nl6UyIQ7wIlTLef'
access_token = '561838533-0z3rbvwSDl4LX1cwQqc9ujxi5XSxvnCzCvF0Unrl'
access_token_secret = '4uviuIDbzKJe9DZpJIv7kCnj9DEMZTwDnpe796eQXaTcg'

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Account
target_users = ("@BBCWorld","@CBSNews","@CNN","@FoxNews","@nytimes")

# Counter
counter = 1

# Variables for holding sentiments
sentiments = []

# Loop through all target users
for user in target_users:
    
    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(5):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, page=x)

        # Loop through all tweets 
        for tweet in public_tweets:

            # Run Vader Analysis on each tweet
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
                   
            # Add sentiments for each tweet into an array
            sentiments.append({"Date": tweet["created_at"], 
                               "Compound": compound,
                               "Positive": pos,
                               "Negative": neu,
                               "Neutral": neg,
                               "Text": tweet["text"],
                               "Tweets Ago": counter,
                               "User": user
                               })
            # Add to counter 
            counter = counter + 1

```


```python
# Convert sentiments to DataFrame
sentiments_pd = pd.DataFrame.from_dict(sentiments)

# Count Tweets by User
def rolling_count(val):
    if val == rolling_count.previous:
        rolling_count.count +=1
    else:
        rolling_count.previous = val
        rolling_count.count = 1
    return rolling_count.count

rolling_count.count = 0 #static variable
rolling_count.previous = None #static variable

# Upate Tweets Ago with correct number of tweets
sentiments_pd['Tweets Ago'] = sentiments_pd['User'].apply(rolling_count) #new column in dataframe

# Export sentiments to News Mood CSV
sentiments_pd.to_csv("News Mood.csv")
sentiments_pd
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweets Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.4019</td>
      <td>Tue Jan 09 02:27:07 +0000 2018</td>
      <td>0.838</td>
      <td>0.000</td>
      <td>0.162</td>
      <td>At the stroke of midnight, same-sex couples in...</td>
      <td>1</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.4215</td>
      <td>Tue Jan 09 02:13:25 +0000 2018</td>
      <td>0.682</td>
      <td>0.318</td>
      <td>0.000</td>
      <td>The people protesting Trump's immigration dead...</td>
      <td>2</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.2960</td>
      <td>Tue Jan 09 02:13:25 +0000 2018</td>
      <td>0.804</td>
      <td>0.196</td>
      <td>0.000</td>
      <td>Calais Jungle: Police try to stop new camp for...</td>
      <td>3</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>Tue Jan 09 01:59:26 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>RT @BBCNewsAsia: 2018 Winter Olympics: North a...</td>
      <td>4</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>Tue Jan 09 01:13:21 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Romania's changing face https://t.co/CVruZyesAm</td>
      <td>5</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0000</td>
      <td>Tue Jan 09 00:51:10 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Living with Brian's 'ghost' https://t.co/jToxb...</td>
      <td>6</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.4215</td>
      <td>Tue Jan 09 00:25:27 +0000 2018</td>
      <td>0.763</td>
      <td>0.237</td>
      <td>0.000</td>
      <td>Caruana Galizia case: Malta ex-corruption inve...</td>
      <td>7</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.4767</td>
      <td>Tue Jan 09 00:14:58 +0000 2018</td>
      <td>0.744</td>
      <td>0.256</td>
      <td>0.000</td>
      <td>US police chief 'kutecop4you' arrested in chil...</td>
      <td>8</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.1280</td>
      <td>Mon Jan 08 23:39:18 +0000 2018</td>
      <td>0.620</td>
      <td>0.209</td>
      <td>0.171</td>
      <td>Peru Pasamayo: Lorry driver admits causing dea...</td>
      <td>9</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:07:05 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Apple investigated by France for 'planned obso...</td>
      <td>10</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:07:05 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Gay marriage: Couple among first to wed in Aus...</td>
      <td>11</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0000</td>
      <td>Mon Jan 08 22:45:45 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Russia probe: Trump lawyers 'in talks over Mue...</td>
      <td>12</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.5574</td>
      <td>Mon Jan 08 22:31:10 +0000 2018</td>
      <td>0.882</td>
      <td>0.118</td>
      <td>0.000</td>
      <td>RT @BBCSport: Widnes Vikings' Kato Ottio has d...</td>
      <td>13</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0000</td>
      <td>Mon Jan 08 22:21:04 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>North and South Korea to begin high-level talk...</td>
      <td>14</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0000</td>
      <td>Mon Jan 08 22:07:47 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Google sued over 'male discrimination' https:/...</td>
      <td>15</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0000</td>
      <td>Mon Jan 08 21:22:53 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Rugby League player Ottio dies aged 23 https:/...</td>
      <td>16</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.3923</td>
      <td>Mon Jan 08 21:20:13 +0000 2018</td>
      <td>0.744</td>
      <td>0.164</td>
      <td>0.092</td>
      <td>RT @BBCNorthAmerica: - Sad!\n- Bigly? Or big l...</td>
      <td>17</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.7717</td>
      <td>Mon Jan 08 21:09:40 +0000 2018</td>
      <td>0.758</td>
      <td>0.242</td>
      <td>0.000</td>
      <td>RT @BBCSport: Widnes Vikings centre Kato Ottio...</td>
      <td>18</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.5719</td>
      <td>Mon Jan 08 19:26:48 +0000 2018</td>
      <td>0.515</td>
      <td>0.485</td>
      <td>0.000</td>
      <td>'Raw water': A dangerous new health craze? htt...</td>
      <td>19</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0000</td>
      <td>Mon Jan 08 18:44:08 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Ghana bars recruits over stretch marks and ble...</td>
      <td>20</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.4019</td>
      <td>Tue Jan 09 02:27:07 +0000 2018</td>
      <td>0.838</td>
      <td>0.000</td>
      <td>0.162</td>
      <td>At the stroke of midnight, same-sex couples in...</td>
      <td>21</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.4215</td>
      <td>Tue Jan 09 02:13:25 +0000 2018</td>
      <td>0.682</td>
      <td>0.318</td>
      <td>0.000</td>
      <td>The people protesting Trump's immigration dead...</td>
      <td>22</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.2960</td>
      <td>Tue Jan 09 02:13:25 +0000 2018</td>
      <td>0.804</td>
      <td>0.196</td>
      <td>0.000</td>
      <td>Calais Jungle: Police try to stop new camp for...</td>
      <td>23</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0000</td>
      <td>Tue Jan 09 01:59:26 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>RT @BBCNewsAsia: 2018 Winter Olympics: North a...</td>
      <td>24</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0000</td>
      <td>Tue Jan 09 01:13:21 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Romania's changing face https://t.co/CVruZyesAm</td>
      <td>25</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0000</td>
      <td>Tue Jan 09 00:51:10 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Living with Brian's 'ghost' https://t.co/jToxb...</td>
      <td>26</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.4215</td>
      <td>Tue Jan 09 00:25:27 +0000 2018</td>
      <td>0.763</td>
      <td>0.237</td>
      <td>0.000</td>
      <td>Caruana Galizia case: Malta ex-corruption inve...</td>
      <td>27</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.4767</td>
      <td>Tue Jan 09 00:14:58 +0000 2018</td>
      <td>0.744</td>
      <td>0.256</td>
      <td>0.000</td>
      <td>US police chief 'kutecop4you' arrested in chil...</td>
      <td>28</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.1280</td>
      <td>Mon Jan 08 23:39:18 +0000 2018</td>
      <td>0.620</td>
      <td>0.209</td>
      <td>0.171</td>
      <td>Peru Pasamayo: Lorry driver admits causing dea...</td>
      <td>29</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:07:05 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Apple investigated by France for 'planned obso...</td>
      <td>30</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>-0.4215</td>
      <td>Mon Jan 08 16:10:04 +0000 2018</td>
      <td>0.725</td>
      <td>0.188</td>
      <td>0.087</td>
      <td>RT @marcatracy: Tonight, Georgia's band will o...</td>
      <td>71</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>471</th>
      <td>0.0000</td>
      <td>Mon Jan 08 16:00:34 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Do parents make kids fat? https://t.co/4YjB4jynMC</td>
      <td>72</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>472</th>
      <td>0.0000</td>
      <td>Mon Jan 08 15:55:07 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>RT @nytvideo: The Trump administration will en...</td>
      <td>73</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>473</th>
      <td>0.0000</td>
      <td>Mon Jan 08 15:51:07 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Some of the economic policies Trump's administ...</td>
      <td>74</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>474</th>
      <td>0.5994</td>
      <td>Mon Jan 08 15:41:05 +0000 2018</td>
      <td>0.754</td>
      <td>0.000</td>
      <td>0.246</td>
      <td>RT @nytopinion: Trump's defenders have learned...</td>
      <td>75</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>475</th>
      <td>-0.8360</td>
      <td>Mon Jan 08 15:33:07 +0000 2018</td>
      <td>0.706</td>
      <td>0.294</td>
      <td>0.000</td>
      <td>Recy Taylor, who Oprah Winfrey mentioned in he...</td>
      <td>76</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>476</th>
      <td>0.0000</td>
      <td>Mon Jan 08 15:20:13 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>President Hassan Rouhani of Iran lashed out at...</td>
      <td>77</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>477</th>
      <td>0.0000</td>
      <td>Mon Jan 08 15:20:05 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Breaking News: Nearly 200,000 Salvadorans allo...</td>
      <td>78</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>478</th>
      <td>0.0258</td>
      <td>Mon Jan 08 15:11:08 +0000 2018</td>
      <td>0.794</td>
      <td>0.100</td>
      <td>0.105</td>
      <td>RT @NYTScience: “There’s this hollowness yet t...</td>
      <td>79</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>479</th>
      <td>-0.1280</td>
      <td>Mon Jan 08 15:10:07 +0000 2018</td>
      <td>0.933</td>
      <td>0.067</td>
      <td>0.000</td>
      <td>Seth Meyers wisecracked about the notion that ...</td>
      <td>80</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>480</th>
      <td>0.7125</td>
      <td>Mon Jan 08 15:04:55 +0000 2018</td>
      <td>0.762</td>
      <td>0.000</td>
      <td>0.238</td>
      <td>RT @melbournecoal: BREAK: GLAMOUR has chosen i...</td>
      <td>81</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>481</th>
      <td>0.0000</td>
      <td>Mon Jan 08 15:00:35 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>82</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>482</th>
      <td>0.0000</td>
      <td>Mon Jan 08 14:46:05 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Video: #MeToo leads at the Golden Globes https...</td>
      <td>83</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>483</th>
      <td>0.0000</td>
      <td>Mon Jan 08 14:41:02 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Last May, Jared Kushner accompanied President ...</td>
      <td>84</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>484</th>
      <td>-0.2732</td>
      <td>Mon Jan 08 14:36:33 +0000 2018</td>
      <td>0.684</td>
      <td>0.198</td>
      <td>0.118</td>
      <td>RT @UpshotNYT: The jobless rate is low; so are...</td>
      <td>85</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>485</th>
      <td>-0.8126</td>
      <td>Mon Jan 08 14:31:09 +0000 2018</td>
      <td>0.571</td>
      <td>0.364</td>
      <td>0.064</td>
      <td>A fire broke out near the top of Trump Tower i...</td>
      <td>86</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>486</th>
      <td>0.0000</td>
      <td>Mon Jan 08 14:15:14 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Eight pairs of actresses and activists attende...</td>
      <td>87</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>487</th>
      <td>0.0000</td>
      <td>Mon Jan 08 14:00:22 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Bannon needs Breitbart. Does Breitbart need Ba...</td>
      <td>88</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>488</th>
      <td>0.0000</td>
      <td>Mon Jan 08 13:46:07 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Natalie Portman delivered a line that instantl...</td>
      <td>89</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>489</th>
      <td>0.0258</td>
      <td>Mon Jan 08 13:32:06 +0000 2018</td>
      <td>0.687</td>
      <td>0.155</td>
      <td>0.159</td>
      <td>"The Globes still celebrated the power players...</td>
      <td>90</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>490</th>
      <td>0.0000</td>
      <td>Mon Jan 08 13:00:28 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>91</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>491</th>
      <td>-0.7096</td>
      <td>Mon Jan 08 12:45:15 +0000 2018</td>
      <td>0.704</td>
      <td>0.296</td>
      <td>0.000</td>
      <td>Iran has banned the teaching of English in pri...</td>
      <td>92</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>492</th>
      <td>-0.7003</td>
      <td>Mon Jan 08 12:31:05 +0000 2018</td>
      <td>0.746</td>
      <td>0.254</td>
      <td>0.000</td>
      <td>A senior BBC News editor accused the network o...</td>
      <td>93</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0.0000</td>
      <td>Mon Jan 08 12:15:09 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>A transcript of Seth Meyers's Golden Globes mo...</td>
      <td>94</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>494</th>
      <td>0.2732</td>
      <td>Mon Jan 08 12:00:21 +0000 2018</td>
      <td>0.890</td>
      <td>0.000</td>
      <td>0.110</td>
      <td>President Trump defended his fitness for offic...</td>
      <td>95</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>495</th>
      <td>0.0258</td>
      <td>Mon Jan 08 11:50:08 +0000 2018</td>
      <td>0.942</td>
      <td>0.000</td>
      <td>0.058</td>
      <td>Stephen Bannon tried backing away from his exp...</td>
      <td>96</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>496</th>
      <td>0.0000</td>
      <td>Mon Jan 08 11:40:13 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Read speeches from Elisabeth Moss, Laura Dern,...</td>
      <td>97</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>497</th>
      <td>0.0000</td>
      <td>Mon Jan 08 11:30:06 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Morning Briefing: Here's what you need to know...</td>
      <td>98</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>498</th>
      <td>-0.2732</td>
      <td>Mon Jan 08 11:21:02 +0000 2018</td>
      <td>0.909</td>
      <td>0.091</td>
      <td>0.000</td>
      <td>An Iranian oil tanker that collided with anoth...</td>
      <td>99</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>499</th>
      <td>0.4215</td>
      <td>Mon Jan 08 11:11:02 +0000 2018</td>
      <td>0.877</td>
      <td>0.000</td>
      <td>0.123</td>
      <td>RT @nytimesworld: Remember the “new Coke” blun...</td>
      <td>100</td>
      <td>@nytimes</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>




```python
# Create Scatter plot on 100 tweets per User
sns.set(style="ticks")

fg = sns.FacetGrid(data=sentiments_pd, hue="User", size=8)
fg.map(plt.scatter, "Tweets Ago", "Compound").add_legend()
plt.title("Sentiment Analyis of Media Tweets")
plt.ylabel("Tweet Polarity")
plt.savefig("Sentiment_Scatter_Plot.png")
plt.show()
```


![png](output_4_0.png)



```python
# Overall Sentiment
overall = sentiments_pd.groupby("User", as_index=False).mean()
overall
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User</th>
      <th>Compound</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCWorld</td>
      <td>-0.024288</td>
      <td>0.81190</td>
      <td>0.10986</td>
      <td>0.07825</td>
      <td>50.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@CBSNews</td>
      <td>0.015795</td>
      <td>0.89505</td>
      <td>0.05109</td>
      <td>0.05384</td>
      <td>50.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@CNN</td>
      <td>0.075957</td>
      <td>0.89730</td>
      <td>0.03599</td>
      <td>0.06670</td>
      <td>50.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@FoxNews</td>
      <td>0.025239</td>
      <td>0.86870</td>
      <td>0.06539</td>
      <td>0.06590</td>
      <td>50.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@nytimes</td>
      <td>-0.018440</td>
      <td>0.88192</td>
      <td>0.06205</td>
      <td>0.05602</td>
      <td>50.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create bar plot on overall sentiment per User

fg2 = sns.barplot(x="User", y="Compound", data=overall)
plt.title("Overall Media Sentiment Based on Twitter")
plt.ylabel("Tweet Polarity")
plt.savefig("Overall_Sentiment_Bar_Plot.png")
plt.show()
```


![png](output_6_0.png)

