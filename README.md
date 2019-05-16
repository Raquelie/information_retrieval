# information_retrieval
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
