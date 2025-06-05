## 🧠 Goal:
**Understand what real voters are talking about and how they feel about political issues in swing states** using NLP.

---

## 🛠️ Step-by-Step Roadmap (with Code Snippets)

### ✅ Step 1: **Define the Question**
"What issues do voters in [swing state] talk about the most?"
"What is the sentiment of posts about the economy, immigration, or healthcare?"

---

### ✅ Step 2: **Collect Data**
Use Reddit or Twitter API. For Reddit, use **Pushshift** API (via `psaw` or requests).

#### Example: Reddit Posts by Swing State Subreddits

```python
# Install with: pip install psaw
from psaw import PushshiftAPI
import datetime as dt
import pandas as pd

api = PushshiftAPI()

state_subs = ['Pennsylvania', 'Georgia', 'Michigan', 'Wisconsin', 'Arizona']

posts = list(api.search_submissions(
    subreddit='pennsylvania',
    after=int(dt.datetime(2023,1,1).timestamp()),
    before=int(dt.datetime(2024,1,1).timestamp()),
    filter=['title', 'selftext', 'created_utc'],
    limit=1000
))

data = [{
    'text': post.title + ' ' + (post.selftext or ''),
    'timestamp': dt.datetime.fromtimestamp(post.created_utc)
} for post in posts]

df = pd.DataFrame(data)
```

---

### ✅ Step 3: **Clean and Preprocess Text**

```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

def clean(text):
    text = re.sub(r"http\S+|[^a-zA-Z ]", "", text.lower())
    tokens = text.split()
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if w not in stopwords.words("english")]
    return " ".join(tokens)

df['cleaned_text'] = df['text'].apply(clean)
```

---

### ✅ Step 4: **Choose NLP Tasks**
Choose your focus:
- **Topic modeling** → discover common issues (e.g., "inflation", "crime", "immigration")
- **Sentiment analysis** → assess public mood
- **NER** → extract politicians, policy terms, etc.

---

### ✅ Step 5: **Topic Modeling (to find core issues)**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])

lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(X)

# Show top 10 words per topic
words = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    print(f"Topic {i}: ", [words[i] for i in topic.argsort()[-10:]])
```

---

### ✅ Step 6: **Sentiment Analysis (using pretrained model or lexicon)**

#### Option A: Simple VADER

```python
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
```

#### Option B: Finetuned Transformer (e.g., RoBERTa, BERT)

```python
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

df['sentiment'] = df['text'].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'])
```

---

### ✅ Step 7: **Aggregate by Time or Issue**

Group sentiment or topics over time, per state:

```python
df['month'] = df['timestamp'].dt.to_period('M')
monthly_sentiment = df.groupby('month')['sentiment'].mean()
```

Or group by keywords:

```python
economy_mask = df['cleaned_text'].str.contains("economy|inflation|job|wage")
print(df[economy_mask]['sentiment'].mean())
```

---

### ✅ Step 8: **Visualize Trends**
Use matplotlib or seaborn to chart sentiment or topic frequency.

```python
import matplotlib.pyplot as plt
monthly_sentiment.plot(title="Monthly Voter Sentiment in Pennsylvania")
plt.ylabel("Average Sentiment (VADER)")
plt.show()
```

---

### 🔍 Bonus:
You can track **how sentiment on specific issues (e.g. “inflation”) changes over time or differs by state**.

---

## 📊 Example Output:
| Month | State       | Topic        | Sentiment |
|-------|-------------|--------------|-----------|
| Jan   | Pennsylvania| Economy      | +0.21     |
| Jan   | Arizona     | Immigration  | -0.13     |
| Feb   | Georgia     | Health care  | +0.04     |

---

To make a word cloud like the one you uploaded — **per state** — showing which words people mention the most in Reddit posts, follow this step-by-step using your `df` (which includes a `state` and `cleaned_text` column):

---

## ✅ Step-by-Step: Generate a Word Cloud Per State

### ✅ 1. Install the wordcloud library

If not already installed:

```bash
pip install wordcloud
```

---

### ✅ 2. Import and Set Up

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
```

---

### ✅ 3. Create and Plot a Word Cloud for One State

```python
def plot_wordcloud(state_name, df):
    text = " ".join(df[df['state'] == state_name]['cleaned_text'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Most Common Words in {state_name}", fontsize=16)
    plt.show()

# Example
plot_wordcloud("Arizona", df)
```

---

### ✅ 4. Automatically Generate One for Each State

```python
for state in df['state'].unique():
    plot_wordcloud(state, df)
```

---

### 🧠 Optional Enhancements

* Filter by **issue** (e.g., only posts labeled as “Economy”)
* Add stopwords (like "people", "thing") with `stopwords` argument
* Export each image with `plt.savefig(f"{state}_wordcloud.png")`

---

Would you like help filtering these by `top_issue` or saving them to disk as image files for a report?


Great! Here’s a **step-by-step guide** to:

1. ✅ Set up Reddit API credentials
2. ✅ Use `praw` to fetch **both posts and comments**
3. ✅ Preprocess and run **basic sentiment analysis**

---

## 🔐 STEP 1: Set Up Reddit API Credentials

1. Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Scroll down and click: **"create app"**
3. Fill out:

   * **Name:** `swing_state_scraper`
   * **Type:** `script`
   * **Redirect URI:** `http://localhost`
4. After submission, you’ll get:

   * `client_id` (below the app name)
   * `client_secret` (in the app details)

---

## 📦 STEP 2: Install `praw`

```bash
pip install praw
```

---

## 🧪 STEP 3: Use `praw` to Fetch Posts + Comments

```python
import praw
import pandas as pd
from datetime import datetime

# 🔑 Replace with your actual credentials
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="swing_state_scraper"
)

def fetch_reddit_posts_and_comments(subreddit_name, limit=50):
    subreddit = reddit.subreddit(subreddit_name)
    data = []

    for submission in subreddit.hot(limit=limit):
        if not submission.stickied:
            post = {
                'title': submission.title,
                'text': submission.selftext,
                'created_utc': datetime.utcfromtimestamp(submission.created_utc),
                'subreddit': subreddit_name,
                'comments': []
            }

            submission.comments.replace_more(limit=0)  # Load all comments
            post['comments'] = [comment.body for comment in submission.comments[:10]]  # Top 10 comments

            data.append(post)

    return pd.DataFrame(data)

# Example usage
df = fetch_reddit_posts_and_comments('PApolitics', limit=20)
print(df.head(3))
```

---

## 💬 STEP 4: Sentiment Analysis with TextBlob

```bash
pip install textblob
python -m textblob.download_corpora
```

```python
from textblob import TextBlob

def analyze_sentiment(text):
    if not text:
        return 0
    return TextBlob(text).sentiment.polarity  # [-1 to 1]

# Add sentiment scores
df['post_sentiment'] = df['text'].apply(analyze_sentiment)
df['avg_comment_sentiment'] = df['comments'].apply(lambda comments: 
    sum(analyze_sentiment(c) for c in comments) / len(comments) if comments else 0
)

print(df[['title', 'post_sentiment', 'avg_comment_sentiment']].head())
```

---

## ✅ Optional: Filter for Swing State Topics

You can apply keyword filters like:

```python
keywords = ['healthcare', 'jobs', 'abortion', 'biden', 'trump', 'inflation']
df_filtered = df[df['text'].str.contains('|'.join(keywords), case=False)]
```

---

Would you like to:

* Turn this into a live dashboard or report?
* Extract topics (LDA or BERTopic)?
* Expand to multiple swing-state subreddits (e.g. `r/WisconsinPolitics`, `r/Michigan`)?
