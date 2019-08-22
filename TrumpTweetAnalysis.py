#!/usr/bin/env python
# coding: utf-8

# # Trump Twitter Analysis
# This is a script that aim to analyze Mr. Trump's tweets, starting with what is the most common words he used, and which country was the most frequently mentioned
# This script is created by Yuezhe Li. 
# This script is strictly for fun, and not for other use. 
# 
# ### Before you play with this script, you need to create a twitter app to generate the keys and access tokens. Here is a detailed explanation. 
# https://cran.r-project.org/web/packages/rtweet/vignettes/auth.html

# In[1]:


import tweepy
import pandas as pd


# the following info is created to pull data from Trump's Twitter
consumer_key= 'YourKey'
consumer_secret= '***'
access_token= '*-vM0i9PvnMoEycs9MAac79SMFfTLjvo'
access_token_secret= '***'


# In[2]:


def get_all_tweets(screen_name):

    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    #initialize a list to hold all the tweepy Tweets
    alltweets = []

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1

        outtweets = [tweet.text.encode("utf-8") for tweet in alltweets]
        
    return outtweets


# In[3]:


# get all the tweets from Trump
trumptweet = get_all_tweets("realDonaldTrump")


# In[4]:


# clean up source text by removing some Twitter specific items
cleantweet1 = [str(w).replace("b'", "") for w in trumptweet]
cleantweet2 = [str(w).replace("RT", "") for w in cleantweet1]

cleantweet3 = ''.join(map(str, cleantweet2))


import re
# remove all the url in the text
tweettext = re.sub(r'http\S+', '', cleantweet3)


# In[14]:


# define a function to find all the words that are in a list
def findallwords(listofwords, tweettext):
    frequency = 0
    for words in listofwords:
        frequency += len(re.findall(words, tweettext))
    
    return frequency


# In[24]:


print(findallwords(['North Korea', 'North Korean', 'Kim'], tweettext))
print(findallwords(['Russia', 'Russian', 'Moscow', 'Kremlin', 'Putin'], tweettext))
print(findallwords(['Mexico', 'Mexican'], tweettext))
print(findallwords(['China', 'Chinese', 'Xi', 'Beijing', 'Hong Kong','Huawei', 'Sino'], tweettext))


# In[ ]:




