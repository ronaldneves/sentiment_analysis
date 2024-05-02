import pandas as pd 
import nltk  
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')


def pre_processamento_texto(texto):
    tokens = word_tokenize(texto.lower())

    tokens_filtrados = [token for token in tokens if token not in stopwords.words('english')]
    lematizador = WordNetLemmatizer()
    tokens_lematizados = [lematizador.lemmatize(token) for token in tokens_filtrados]
    texto_processado = ' '.join(tokens_lematizados)

    return texto_processado


def analise_sentimento(texto):
    analise = SentimentIntensityAnalyzer()

    nota_sentimento = analise.polarity_scores(texto)
    sentimento = 1 if nota_sentimento['pos'] > 0 else 0

    return sentimento


df['reviewText'] = df['reviewText'].apply(pre_processamento_texto)
df['sentimento'] = df['reviewText'].apply(analise_sentimento)

print(df)