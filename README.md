# TGINE_reddit_v1

This program extracts posts and comments data from Reddit's subreddit r/AskScience and performs basic tf/idf vectorization. Four types of extraction are defined as per Reddit browing options:

- new
- top
- rising
- hot

## Execution

Requires one parameter: limit of type int 

Limit is the max number of posts retrieved from reddit.com (max allowed by library is 1000)

Usage: `TGINE_reddit_v1.py limit`
 
Only creates each file if it does not exist beforehand. Otherwise reads from existing file.

## Output

The output is:

- One json file for each mode (new, top, rising, hot) if they didn't exist before execution
- Screen output for each file. Order is new, top, rising, hot


# TGINE_twitter_v1

This program extracts posts from 4 Twitter users and performs classification and basic sentiment analysis. For classification, an SVM model with different parameters is used, and only the best parameters for the training set are used to test on new data.

## Execution

No parameters

## Output

The output is:

- One json file for each user 
- Screen output for each dataset (2 users each dataset): parameters, accuracy and confusion matrix (train and test)
- Screen output for each user: ratio of positive and negative words
