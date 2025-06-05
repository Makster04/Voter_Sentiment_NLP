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

Would you like me to generate this pipeline into a complete script or notebook for you to run?
