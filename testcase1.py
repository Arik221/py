import seaborn as sns
import requests
from pybrain.structure import FeedForwardNetwork
from nltk.tokenize import word_tokenize


iris = sns.load_dataset("iris")
response = requests.get('https://api.example.com/data')


network = FeedForwardNetwork()


sentence = "This is an example sentence."
tokens = word_tokenize(sentence)

print("Seaborn Iris Dataset Head:")
print(iris.head())

print("\nResponse from API:", response.text)

print("\nPybrain Feedforward Network:", network)

print("\nNLTK Tokenized Sentence:", tokens)
