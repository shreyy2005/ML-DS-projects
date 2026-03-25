# ============================================================
#  SPAM EMAIL CLASSIFIER — Shrey Dukare
#  Copy-paste this entire file into ONE Google Colab cell
# ============================================================

# ---------- STEP 1: Install & Import ----------
import subprocess
subprocess.run(["pip", "install", "wordcloud", "--quiet"])

import re, string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')
print("✅ Libraries imported!")

# ---------- STEP 2: Load Dataset ----------
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
print(f"✅ Dataset loaded: {df.shape[0]} messages")
print(df['label'].value_counts())

# ---------- STEP 3: EDA — Message Length ----------
df['msg_length'] = df['message'].apply(len)
print("\nAverage message length:")
print(df.groupby('label')['msg_length'].mean().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ['#2ecc71', '#e74c3c']
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=colors, edgecolor='black', rot=0)
axes[0].set_title('Spam vs Ham Count', fontweight='bold')
for i, v in enumerate(df['label'].value_counts()):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
for label, color in zip(['ham', 'spam'], colors):
    axes[1].hist(df[df['label'] == label]['msg_length'], bins=40, alpha=0.6, label=label, color=color)
axes[1].set_title('Message Length Distribution', fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.show()

# ---------- STEP 4: Word Clouds ----------
spam_words = ' '.join(df[df['label'] == 'spam']['message'].tolist())
ham_words  = ' '.join(df[df['label'] == 'ham']['message'].tolist())
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].imshow(WordCloud(width=700, height=400, background_color='white', colormap='Reds').generate(spam_words))
axes[0].axis('off'); axes[0].set_title('🚨 SPAM Words', fontweight='bold', color='red')
axes[1].imshow(WordCloud(width=700, height=400, background_color='white', colormap='Greens').generate(ham_words))
axes[1].axis('off'); axes[1].set_title('✅ HAM Words', fontweight='bold', color='green')
plt.tight_layout(); plt.show()

# ---------- STEP 5: Preprocessing ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_message'] = df['message'].apply(preprocess_text)
print("✅ Text cleaned!")

# ---------- STEP 6: TF-IDF + Train/Test Split ----------
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"✅ TF-IDF done. Feature matrix shape: {X_train_tfidf.shape}")

# ---------- STEP 7: Train 3 Models ----------
models = {
    'Naive Bayes'         : MultinomialNB(),
    'Logistic Regression' : LogisticRegression(max_iter=1000),
    'LinearSVC'           : LinearSVC(max_iter=1000)
}
results = {}
print("\n" + "="*55)
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'predictions': y_pred, 'accuracy': acc}
    print(f"\n🔷 {name}  —  Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

best_name = max(results, key=lambda k: results[k]['accuracy'])
print(f"🏆 Best Model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")

# ---------- STEP 8: Visualize ----------
names = list(results.keys())
accs  = [results[n]['accuracy'] * 100 for n in names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(names, accs, color=['#3498db','#e67e22','#9b59b6'], edgecolor='black', width=0.5)
axes[0].set_ylim(90, 100); axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{acc:.2f}%', ha='center', fontweight='bold')
cm = confusion_matrix(y_test, results[best_name]['predictions'])
ConfusionMatrixDisplay(cm, display_labels=['Ham','Spam']).plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
plt.tight_layout(); plt.show()

# ---------- STEP 9: Predict Custom Messages ----------
def predict_spam(message):
    cleaned = preprocess_text(message)
    vec = tfidf.transform([cleaned])
    pred = results['Logistic Regression']['model'].predict(vec)[0]
    label = '🚨 SPAM' if pred == 1 else '✅ HAM'
    print(f'{label}  →  "{message}"')

print("\n" + "="*55)
print("       CUSTOM MESSAGE PREDICTIONS")
print("="*55)
test_msgs = [
    "Congratulations! You've won a FREE iPhone. Click here to claim your prize NOW!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account is suspended. Call 1800-XXX-XXXX immediately.",
    "Don't forget to submit your assignment before Friday.",
    "Win Rs 50,000 cash! Reply WIN to 56161 now. Limited offer!",
    "Mom, I'll be home by 8pm tonight."
]
for msg in test_msgs:
    predict_spam(msg)

print("\n✅ Project Complete! Try your own message with: predict_spam('your message here')")
