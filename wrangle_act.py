#!/usr/bin/env python
# coding: utf-8

# # Wrangle and Analyze Data

# **Welcome** to the Wrangle and Analyze Data project! This project aims to be wrangling, analyzing and visualizing tweets data of @dog_rates, known as WeRateDogs, to create interesting and trustworthy analyses and insights. <br>
# The project walks throughout three steps: Gathering data which is the first step in data wrangling. Then, Assessing data which is the detect and explore issues of the data and it's preparing the precursor to cleaning data. Finally, Cleaning data that is where to fix the quality and tidiness issues identified in the assessing step.<br><br>
# **Let's** start the journey!
# 
# ## Table of Contents
# <ul>
# <li><a href="#gather">Gathering Data</a></li>
# <li><a href="#assess">Assessing Data</a></li>
# <li><a href="#clean">Cleaning Data</a></li>
# <li><a href="#analyse">Analysing Data</a></li>
# </ul>

# <a id='gather'></a>
# ## Gathering Data

# In this section, the dataset that will be working on is gathering from different sources and in different formats: The tweet archive of Twitter user @dog_rates in `twitter_archive_enhanced.csv`. The tweet image predictions inside the `image_predictions.tsv` file. The tweet's retweet and favourite count that's store in a file called `tweet_json.txt`.<br>
# To get started, let's import the libraries.

# In[1]:


import requests
import os
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt


# In[2]:


df_img = pd.read_csv('image-predictions.tsv', sep = '\t')
df_img.head()


# In[3]:


df_arch = pd.read_csv('twitter-archive-enhanced.csv')
df_arch.head()


# In[4]:


with open('tweet-json copy') as file:
    data = [json.loads(line) for line in file] #list of dictionaries
data[0]


# In[5]:


#tweet ID, retweet count, and favorite count
tweets_list = []
for i in data:
    tweet_id = i['id']
    retweet_count = i['retweet_count']
    favorite_count = i['favorite_count']
    full_text = i['full_text']
    tweet_time = i['created_at']
    tweets_list.append({'tweet_id': tweet_id,
                     'retweet_count': retweet_count,
                     'favorite_count': favorite_count})
df_tweets = pd.DataFrame(tweets_list, columns = ['tweet_id', 'retweet_count', 'favorite_count'])
df_tweets.head()


# <a id='assess'></a>
# ## Assessing Data

# In this section, the dataset will be assessed for quality and tidiness issues after detecting and covering it using both ways visually and programmatically.

# ### Quality Issues

# **The Image Prediction Table**

# In[6]:


df_img.info()


# > No have null data, great!

# In[7]:


#chek the data range of img_num is (1-4)
df_img.img_num.unique()


# In[8]:


df_img.p1_conf.max(), df_img.p2_conf.max(), df_img.p3_conf.max()


# > `p1` has cell with 100% accuracey! it's true or there is over-qualified!

# In[9]:


df_img[df_img.p1_conf == 1]


# > `p1_dog` is false?? with conf 100%

# In[10]:


df_img[df_img.duplicated()]


# > There is no duplicated rows

# **The Tweet Archive Table**

# In[11]:


df_arch.info()


# > - There are columns with null values! <br>
# > - Convert `timestamp` data type from object to date <br>
# > - Convert `in_reply_to_status_id` and `in_reply_to_user_id` from float to int <br>
# > - Take the original tweets (exclude retweeted data)

# In[12]:


df_arch.source.unique()


# > Take the type from tweet's `source`

# In[13]:


df_arch['expanded_urls']


# > Filling nan in `expanded_urls` with Twitter's url + tweet id

# In[14]:


df_arch.name.describe()


# > Replace 'none' in `name` with nan value

# In[15]:


df_arch.name.unique()


# In[16]:


df_arch.name[df_arch.name.str.islower()].sample(10)


# > Replace the names which begin with a small letter such as ('a', 'my', 'his', 'an', 'all', 'by', 'life') with null values

# In[17]:


df_arch.doggo.unique(), df_arch.floofer.unique(), df_arch.pupper.unique(), df_arch.puppo.unique()


# > Replace the latest four columns from 'None' to nan value

# In[18]:


df_arch.rating_numerator.max(), df_arch.rating_denominator.max()


# In[19]:


df_arch.rating_denominator.describe()


# > There are more than 75% of data have `rating_denominator` with 10. So, coordinating the others according to it.

# In[20]:


df_arch[df_arch.rating_denominator > 10] # 20 cases have rating above than 10


# In[21]:


df_arch.rating_numerator[df_arch.rating_numerator >= df_arch.rating_denominator].count() # 1914 cases


# > All cases that have the `rating_numerator` is bigger than the `rating_denominator` that's meaning the rating is 100%. So, to coordinating data, there will be converting these to 10/10, also it's meaning 100%.

# In[22]:


df_arch[(df_arch.rating_numerator < df_arch.rating_denominator) & (df_arch.rating_denominator > 10)] # 7 cases


# > Regarding the cases that have the `rating_numerator` is less than the `rating_denominator`, also they above than 10:<br>
# By return to each tweet from this, the conclude: 
# > - Delete the first row (referring to the lunch account's date). 
#     - `id = 832088576586297345` <br>
# > - The second row referred to 11 Sep(no means rating). So, delete. 
#     - `id = 740373189193256964`<br>
# > - Replace 4/20 to 13/10. (the rating_numerator is more than the rating_denominator) so it will be 10/10. 
#     - `id = 722974582966214656` 	<br>
# > - The rating 45/50 is for five dogs in the image. So, divide 45 for 5 to rating each dog with 10. 
#     - `id = 709198395643068416` <br>
# > - Take the percentage of 4/20. So, convert to 2/10. 
#     - `id = 686035780142297088`
# > - The last one, take the percentage (7/11) to convert it from 10. [(7/11) ~ 63%] the nearest int persentege is (60%) that is 6/10. 
#     - `id = 682962037429899265`

# In[23]:


df_arch[df_arch.tweet_id.duplicated()]


# > There is no duplicates row

# **The Tweets Table**

# In[24]:


df_tweets.info()


# > Great, no null data and it's the appropriate data type

# In[25]:


df_tweets[df_tweets.duplicated()]


# > No contains duplicate rows

# In[26]:


df_tweets.retweet_count.describe()


# In[27]:


df_tweets.favorite_count.describe()


# In[28]:


df_tweets[df_tweets.favorite_count == 132810]


# In[29]:


df_arch[df_arch.tweet_id == 822872901745569793].expanded_urls


# > By referring to the tweet, the number is approximately right.

# **Quality Issues Conclusion:**
# > - There are columns with null values! <br>
# > - Convert `timestamp` data type from object to date <br>
# > - Convert `in_reply_to_status_id` and `in_reply_to_user_id` from float to int <br>
# > - Take the original tweets (exclude retweeted data)<br>
# > - Take the type from tweet's `source` <br>
# > - Filling nan in `expanded_urls` with Twitter's url + tweet id <br>
# > - Replace 'none' in `name` with nan value and replace the names which begin with a small letter such as ('a', 'my', 'his', 'an', 'all', 'by', 'life') with null values <br>
# > - Replace the latest four columns from 'None' to nan value <br>
# > - Coordinate the rating data for 10: <br>
#     - All cases that have the `rating_numerator` is bigger than the `rating_denominator` will be converting to 10/10. <br>
#     - Delete the first row (referring to the lunch account's date). 
#         - `id = 832088576586297345` <br>
#     - The second row referred to 11 Sep(no means rating). So, delete. 
#         - `id = 740373189193256964`<br>
#     - Replace 4/20 to 13/10. (the `rating_numerator` is bigger than the `rating_denominator`) so it will be 10/10. 
#         - `id = 722974582966214656` 	<br>
#     - The rating 45/50 is for five dogs in the image. So, divide 45 for 5 to rating each dog with 10. 
#         - `id = 709198395643068416` <br>
#     - Take the percentage of 4/20. So, convert to 2/10. 
#         - `id = 686035780142297088`
#     - The last one, take the percentage (7/11) to convert it from 10. [(7/11) ~ 63%] the nearest int percentage is (60%) that is 6/10. 
#         - `id = 682962037429899265`

# ### Tidiness Issues
# - Join the dog stages [`doggo`,`floofer`,`pupper`,`puppo`] into one column.
# - Join favoriate and retweeted count with archive dataset

# In[30]:


#join (retweet and favorite counts) with tweet archive, using the key (tweet ID)
df_tweets.tweet_id.dtypes == df_arch.tweet_id.dtypes # should be true


# In[31]:


len(df_tweets) == len(df_arch) # should they have the same rows??


# <a id='clean'></a>
# ## Cleaning Data

# In this section, it will be fixing the quality and tidiness issues of the dataset identified previously. Also, testing each one to verify the truth of coding.<br>
# Let's start with tidiness issues.

# **Tidiness Issues**

# ### Ddefine

# - Join the dog stages [`doggo`,`floofer`,`pupper`,`puppo`] into one column.

# ### Code

# In[32]:


# Exclude None
df_arch['doggo'].replace('None', '', inplace=True)
df_arch['floofer'].replace('None', '', inplace=True)
df_arch['pupper'].replace('None', '', inplace=True)
df_arch['puppo'].replace('None', '', inplace=True)

# Join in one column
df_arch['stage'] = df_arch.doggo.str.cat(df_arch.floofer).str.cat(df_arch.pupper).str.cat(df_arch.puppo)

# Drop the old colomns
df_arch.drop(columns = ['doggo','floofer','pupper','puppo'], inplace = True)

# Assign nan value
df_arch['stage'] = df_arch['stage'].replace('', np.nan)


# ### Test

# In[33]:


df_arch.columns


# ### Define

# - Join favoriate and retweeted count with `archive dataset`

# ### Code

# In[34]:


df = df_arch.join(df_tweets.set_index('tweet_id'), how='inner', on = 'tweet_id')


# ### Test

# In[35]:


df.head(1) # check from df shapes


# **Quality Issues**

# ### Define 

# - Filling nan in `expanded_urls` with Twitter's `url` + `tweet id`

# ### Code

# In[36]:


url = 'https://twitter.com/dog_rates/status/'
df.expanded_urls = df.expanded_urls.fillna(url + df.tweet_id.apply(str))


# ### Test

# In[37]:


sum(df.expanded_urls.isnull())


# ### Define

# - Convert `timestamp` data type from object to date

# ### Code

# In[38]:


df.timestamp = pd.to_datetime(df.timestamp)


# ### Test

# In[39]:


df.timestamp.dtypes


# ### Define

# - Dropping the columns have nan values from `df`

# ### Code

# In[40]:


df.dropna(axis='columns', inplace = True)   


# ### Test

# In[41]:


df.info()


# ### Define

# - Take the original tweets (exclude retweeted data)

# ### Code

# In[42]:


# exclude it from data, it's text begin with'RT' (181 cases)
df.drop(df.loc[df.text.str.contains('RT @'), 'text'].index, axis = 0, inplace = True)
df.reset_index(drop=True, inplace=True)


# ### Test

# In[43]:


df.shape[0] #number of dataset is 2355 - number of retweeted data is 181


# ### Define

# - Take the type from tweet's `source`

# ### Code

# In[44]:


for i in range(len(df.source)):
    df.source[i] = df.source[i][(df.source[i].index('>') + 1):-4]
#df.loc[:, 'source'] = re.findall(r'\>(.*?)\<', str(df.source))


# ### Test

# In[45]:


df.source.unique()


# ### Define

# - Replace 'None' in `name` and the names which begin with a small letter such as ('a', 'my', 'his', 'an', 'all', 'by', 'life') with null values

# ### Code

# In[46]:


df.loc[df.name.str.islower(), 'name'] = np.nan
df['name'].replace(['None'], np.nan, inplace=True)


# ### Test

# In[47]:


df.name.isnull().sum()


# ### Define

# - Coordinate the rating data for 10: <br>
#     - All cases that have the `rating_numerator` is bigger than the `rating_denominator` will be converting to 10/10. <br>
#     - Delete the first row (referring to the lunch account's date). 
#         - `id = 832088576586297345` <br>
#     - The second row referred to 11 Sep(no means rating). So, delete. 
#         - `id = 740373189193256964`<br>
#     - Replace 4/20 to 13/10. (the `rating_numerator` is bigger than the `rating_denominator`) so it will be 10/10. 
#         - `id = 722974582966214656` 	<br>
#     - The rating 45/50 is for five dogs in the image. So, divide 45 for 5 to rating each dog with 10. 
#         - `id = 709198395643068416` <br>
#     - Take the percentage of 4/20. So, convert to 2/10. 
#         - `id = 686035780142297088`
#     - The last one, take the percentage (7/11) to convert it from 10. [(7/11) ~ 63%] the nearest int percentage is (60%) that is 6/10. 
#         - `id = 682962037429899265`

# ### Code

# In[48]:


# Convert rating to 10
df.loc[df.rating_numerator >= df.rating_denominator, ['rating_numerator', 'rating_denominator']] = 10

# Delete the wronge detect rating
indexes = df[ (df.tweet_id == 832088576586297345)  | (df.tweet_id == 740373189193256964)].index
df.drop(indexes , inplace=True)

# Update the rating that above than 10 with approperate meaning
df.loc[df.tweet_id == 722974582966214656,['rating_numerator','rating_denominator']] = 10, 10
df.loc[df.tweet_id == 709198395643068416,['rating_numerator','rating_denominator']] = 9, 10
df.loc[df.tweet_id == 686035780142297088,['rating_numerator','rating_denominator']] = 2, 10
df.loc[df.tweet_id == 682962037429899265,['rating_numerator','rating_denominator']] = 6, 10


# ### Test

# In[49]:


df.rating_numerator.max(), df.rating_denominator.max() # should be 10,10


# ### Storing Data

# After gathered, assessed and cleaned data, now is the time to storing it and working on to exploring what there.

# In[50]:


df.to_csv('twitter_archive_master.csv', index=False)


# <a id='analyse'></a>
# ## Analysing Data

# In this section, the dataset will analyse the insights and displays the visualizations produced from the wrangled data. The exploration and analysing of wrangled data are supporting to make and build related decisions. 

# In[51]:


df_analyse = pd.read_csv('twitter_archive_master.csv')


# To discover what is the most source using in writing the account's tweet, we have to analyse the `source` column in the dataset.

# In[52]:


df_analyse.source.value_counts().plot(kind = 'pie', autopct= '%1.1f%%', figsize=(6,8))
plt.title('Tweet Sources');


# > The most tweets source using is 'Twitter for iPhone'

# Let's see the rating number of dogs, are the most in the median, above or below? To know that, we have to refer to `rating_numerator` and `rating_denominator` columns and analysing these values.

# In[53]:


x = df_analyse.rating_numerator
y = df_analyse.rating_denominator
(((x/y)*100).value_counts()).plot(kind = 'bar', figsize=(8, 6))
plt.title('Rating Dogs (%)')
plt.xlabel('Rating')
plt.ylabel('Frequencies');


# > 80% of these tweets they are rating with (10/10) 100% 

# Here, what about discovering the relation between a retweet and favourite count for a tweet? Is there a relation between them? Are there increasing or decreasing together? Ok, we have to analyse these columns to get these answers. 

# In[54]:


plt.figure(figsize=(8,6))
plt.scatter(df_analyse.retweet_count, df_analyse.favorite_count)
plt.title('The Correlation Between Retweet and Favourite Count')
plt.xlabel('Retweet Count')
plt.ylabel('Favourite Count');


# > The relation between retweet count and favourite count is a strong positive linear relationship.
