import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('spam.csv', encoding='latin-1')
df = pd.DataFrame(data)
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['Label', 'Text']
df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

x = df.drop(['Label'], axis=1)
y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, train_size = 0.9, random_state=56)

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(x_train['Text'])
x_test = vectorizer.transform(x_test['Text']) #check whether it works for x_test only as well


model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict (x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")


spam_counts = pd.Series(y_pred).value_counts()

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.bar(spam_counts.index, spam_counts.values, color=['green', 'red'])
plt.xlabel('Email Type')
plt.ylabel('Email Count')
plt.title('Spam and Non-Spam Emails')
plt.xticks([0, 1], ['Non-Spam', 'Spam'])
plt.show()