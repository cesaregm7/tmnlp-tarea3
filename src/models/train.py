import gensim
from src.features.dictionary import create_dictionary, term_document_matrix
from src.data.prepare_data import read_sample
from src.features.tokenize import tokenize_bigrams,tokenize_trigrams
from pprint import pprint
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim

#Funcion para realizar el entrenamiento
def train(bigrams:bool = True):
    print("Leyendo la muestra de datos...")
    sample = read_sample()
    
    print("Tokenizando los datos...")
    if bigrams:
        dw = tokenize_bigrams(sample)
    else:
        dw = tokenize_trigrams(sample)
    
    print("Creando el diccionario...")
    id2word = create_dictionary(dw)
    
    print("Generando el corpus...")
    corpus = term_document_matrix(dw,id2word)
    
    print("Generando el modelo LDA...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)    
    return dw,lda_model,corpus,id2word
    

    