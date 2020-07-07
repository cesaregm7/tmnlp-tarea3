import pyLDAvis
import pyLDAvis.gensim
import os

#Funcion para generar la visualización con el corpus y el modelo entrenado
def generate_visualization(lda_model,corpus,id2word):
    basePath = os.path.dirname(os.path.abspath(__file__))
    print("Generando visualización...")
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, basePath+'/../../reports/LDA_Visualization.html')