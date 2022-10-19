# Project 10 Bis: Sentiment Analysis and Machine learning  
Consider the Restaurant Review Dataset available at Restaurant Customer Reviews | Kaggle. You may also scrutinize the various available implementations that used the dataset available in Kaggle. The dataset includes a user’s review and a binary indicator to indicate whether the user likes it or not.

1. Use initially SentiStrength SentiStrength - sentiment strength detection in short texts - sentiment analysis, opinion mining (wlv.ac.uk) implementation of sentiment, which provides negative and positive sentiment score, compute Pearson correlation between this constructed sentiment polarity and the annotation. 
    
2. Repeat this process when considering the correlation of the positive class alone and the correlation of the negative class alone.
    
3. Now we want to test the correlation with respect to some stylistic aspects of the review. Write a script that estimate the number of personal pronouns and number of adjectives and number of adverbs using part-of-speech tagger of your choice. Compute both the cosine similarity between each of the above attributes (number of pronouns, number of adjectives, number of adverbs) and the annotation.

4. We want to test the hypothesis that the opinion of about the restaurant is constructed according to Price, Quality of food served in the restaurant, and friendly staff. Suggest a script that allows you to identify Review that are more focused on price, quality of food, friendly staff. You may consider a set of keywords that are most suitable to each category and then use simple string matching to match this effect. For each category, generate a binary vector indicating whether the given review focuses on the corresponding category.

5. Estimate the correlation using Pearson correlation between each vector category and the data annotation. 

6. We want to revisit the construction of the categories in 4). Instead of string matching, use the semantic similarity in the following way. Calculate the Wu and Palmer similarity between “price” and the Review (using the sentence-to-sentence similarity as in labs), repeat this process for the other three categories by suggestion a representative keyword (s) that will be used to calculate sentence-to-sentence similarity score.

7. We want to test another approach for computing the categories by using the empath categories embedding. For this purpose, re-visit the naming of the empath-categories in GitHub - Ejhfast/empath-client: analyze text with empath and select those that might be linked to Price, Quality, Staff friendship. Write a code that allows you to determine appropriate categories from this embedding and then calculate the correlation score.  Alternative to manual scrutinization of the Empath categories, you may also generate an empath category embedding for the keyword “price”, “food quality”, “friendly staff”, and then compute cosine similarity between the Review embedding vector and each of the above four embedding vectors, so that the one that yields the highest similarity score will be considered as the one that best represents the underlined category.    

8. We want to further emphasize on misclassified reviews. For this purpose, concatenate all reviews for which the sentiment score is positive while the annotation is zero and those for which the sentiment is zero while the annotation is 1. Construct the Wordcloud of this dataset. Write a histogram showing the 10 most common wordings in this dataset. Comment on the findings.

9. Now we would like to build a machine learning model for sentiment analysis that takes into account the ambiguous cases identified in 9). For this purpose, write and script and review the preprocessing and stopword list to not discard relevant information in the context of sentiment analysis (e.g., avoid discarding negation cues, adjectives that subsumes polarity and apostrophes, lower-case as capitalization brings emotion,..), then use TfIdfVectorizer with a maximum feature set of 500, minimum 2 repetition and no more than 60% of word repetition across sentences. Build this model for one dataset using randomly selected 70% training and 30% testing. Report the classification accuracy.   

10. Use Glove embedding instead of TfidfVectorizer, see GloVe: Global Vectors for Word Representation (stanford.edu). Use the Glove embedding as feature vectors and test the performance in the original data (30% test data) and report the classification accuracy on the other two datasets. Comment on the limitations of the approach

11. Identify appropriate literature to comment on your findings and methodology.

<br/><br/><br/>

## Old tasks as a reference

Consider the Restaurant Review Dataset available at Restaurant Customer Reviews | Kaggle. You may also
scrutinize the various available implementations that used the dataset available in Kaggle. The dataset
includes a user’s review and a binary indicator to indicate whether the user likes it or not.

  1. Use initially textblob implementation of sentiment, which provides a three-class output (positive,
negative and neutral sentiment polarity). Assuming that both neutral and negative sentiment score are
cast as part of “dislike” category or “0” and positive sentiment is cast in “1”, compute Pearson
correlation between this constructed sentiment polarity and the annotation.

  2. Repeat 1) when using the cosine similarity measure. Repeat this process when considering the
correlation of the positive class alone and the correlation of the negative class alone.

  3. Now we want to test the correlation with respect to some stylistic aspects of the review. Write a script
that estimate the length of the review in terms of number of characters. Compute both the Pearson
correlation and the cosine similarity between the Review length and the annotations.

  4. We want to test the hypothesis that the opinion of about the restaurant is constructed according to
Price, Quality of food, Quantity of the food, and location of restaurant. Suggest a script that allows you
to identify Review that are more focused on price, quality of food, quantity of food and location of
restaurant. You may consider a set of keywords that are most suitable to each category and then use
simple string matching to match this effect. For each category, generate a binary vector indicating
whether the given review focuses on the corresponding category.

  5. Estimate the correlation using Pearson correlation and cosine similarity between each vector category
and the data annotation.

  6. We want to revisit the construction of the categories in 4). Instead of string matching, use the semantic
similarity in the following way. Calculate the Wu and Palmer similarity between “price” and the Review
(using the sentence-to-sentence similarity as in labs), repeat this process for the other three categories
by suggestion a representative keyword (s) that will be used to calculate sentence-to-sentence
similarity score.

  7. We want to test another approach for computing the categories by using the empath categories
embedding. For this purpose, re-visit the naming of the empath-categories in GitHub - Ejhfast/empath-
client: analyze text with empath and select those that might be linked to Price, Quality, Quantity,
Location. Write a code that allows you to determine appropriate categories from this embedding and
then calculate the correlation score. Alternative to manual scrutinization of the Empath categories,
you may also generate an empath category embedding for the keyword “price”, “food quality”, “food
quantity”, “location” and then compute cosine similarity between the Review embedding vector and
each of the above four embedding vectors, so that the one that yields the highest similarity score will
be considered as the one that best represents the underlined category.

  8. Repeat the 1-7) when using SentiStrength sentiment analyzer instead of textblob. The package is
available from sentistrength.wlv.ac.uk.

  9. We want to further emphasize on misclassified reviews. For this purpose, concatenate all reviews for
which the sentiment score is positive while the annotation is zero and those for which the sentiment is
zero while the annotation is 1. Construct the Wordcloud of this dataset. Write a histogram showing the
10 most common wordings in this dataset. Comment on the findings.

  10. Now we would like to build a machine learning model for sentiment analysis that takes into account
the ambiguous cases identified in 9). For this purpose, write and script and review the preprocessing
and stopword list to not discard relevant information in the context of sentiment analysis (e.g., avoid
discarding negation cues, adjectives that subsumes polarity and apostrophes, lower-case as
capitalization brings emotion,..), then use TfIdfVectorizer with a maximum feature set of 1000,
minimum 2 repetition and no more than 60% of word repetition across sentences. Build this model for
one dataset using randomly selected 70% training and 30% testing. Report the classification accuracy.

  11. Use FastText encoding instead of TfidfVectorizer, see
https://github.com/facebookresearch/fastText/archive/v0.2.0.zip
You should install Fasttext and consult one of the tutorial to find out how you will run it, see, e.g.,
FastText Word Embeddings Python implementation - ThinkInfi.
Use the FastText embedding as feature vectors and test the performance in the original data (30% test
data) and report the classification accuracy on the other two datasets. Comment on the limitations of
the approach

  12. Identify appropriate literature to comment on your findings and methodology.

Install packages
```
pip install -r requirements.txt
```
