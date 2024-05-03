import spacy
nlp = spacy.load("fr_core_news_sm")
import fr_core_news_sm
import re
nlp = fr_core_news_sm.load()
import os 
import argparse
from collections import defaultdict,Counter
import json 
import csv
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl  
from matplotlib.patches import Rectangle

mpl.rcParams['figure.dpi'] = 300
#mpl.rc('font',family='Marianne', size=13)
mpl.rc('font',family='Arial', size=13)

"""
The script outputs information about the corpus and three files:

    - information_corpus.txt: a text file with information about the number of tokens and the size of the vocabulary.
    - data_subtitles.json: a dictionary with VERBs, ADJs, NOUNs, and PROPNs sorted by frequency.
    - data_subtitles.csv: a dictionary with VERBs, ADJs, NOUNs, and PROPNs sorted by frequency.
    - 5 figures (.png) Top 20 most frequent VERBs|ADJs|NOUNs|PROPs|ALL
To use it: python lexicometry.py --subtitles_folder PATH/TO/FOLDER
"""

def lister_fichiers(chemin:str)->list:
    """
    The function takes a folder name as input and returns a list of files it contains.

    Args:
        path (str): name of the folder

    Returns:
        list: list of paths of files contained in the folder

    Examples:
        list_files("subtitles") = ["subtitles/files.json",...]
    """
    try:
        os.path.isdir(chemin)
        # Si le chemin est un dossier, retourner la liste des fichiers dans le dossier
        fichiers = [os.path.join(chemin, f) for f in os.listdir(chemin) if os.path.isfile(os.path.join(chemin, f))]
        return fichiers
    except Exception as e:
        print(f"Erreur : {e}")
        return []


def json_to_text(file:str)->str:
    """
    Conversion of subtitle JSON files to text. Only the values of the 'text' attribute from the JSON are kept.

    Args:
        file (str): name of the JSON file

    Returns:
        str: textual content

    Examples:
        json_to_text("subtitles/file.json") = "Textual content. Of the entire file. ..."
    """

    text = ""
    with open(file) as f:
        content = json.load(f)
        j = 0
        while j < len(content):
            element = content[j]
            text = text + element['text'] + " "
            text=text.replace("\n"," ")
            j +=1
    return text 


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subtitles_folder', help="subtitle's folder to process")
    args = parser.parse_args()
    dossier_cible = args.subtitles_folder 

    fichiers_dans_dossier = lister_fichiers(dossier_cible)

    whole_text = ""
    for file in tqdm(fichiers_dans_dossier, desc="files processing"):
        whole_text = whole_text + json_to_text(file)

    # gérer la mémoire de spacy pour le traitement du texte
    nlp.max_length = len(whole_text)

    print("\nTOKENISATION ....")
    doc = nlp(whole_text)
    print("DONE \n\n")

    # on récupère tous les tokens hors ponctuation
    tokens = [token.text for token in doc if not token.is_punct]
    nb_token = len(tokens)

    # on récupère le vocabulaire (lemma)
    voc = [token.lemma_ for token in doc]

    # et on compte le nombre d'occurrence par lemme
    c = Counter()
    for element in voc:
        c[element] +=1

    # on récupère la taille du vocabulaire (nombre de lemme différent)
    taille_voc = len(c)

    tt = f"INFORMATIONS CORPUS : \n\n token's number = {nb_token} \n vocabulary size = {taille_voc}"
    print(tt)

    # on renvoie les informations dans un fichier text + on les affiche

    with open("information_corpus","w") as input:
        input.write(tt)
    
    # on récupère seulement les VERB, NOUN, PROPN et ADJ

    infor_lemma_pos = [(token.lemma_,token.pos_) for token in doc if token.pos_=='NOUN' or token.pos_ == 'VERB' or token.pos_=='ADJ' or token.pos_=="PROPN"]

    counter = Counter()

    for element in infor_lemma_pos:
        counter[element] +=1
    
    dict_lex = dict(counter)

    # on les trie par fréquence (ordre décroissant)
    # sorted by frequency
    dict_lex=dict(sorted(dict_lex.items(), key=lambda item: item[1], reverse=True))

    # key sous forme de texte et non de tuple pour remplir un json
    # key is string to file the json
    dict_lex_str = {str(key):value for key,value in dict_lex.items()}

    # enregistrer les informations dans un json
    # save informations in json
    with open('data_subtitles.json', 'w') as mon_fichier:
	    json.dump(dict_lex_str, mon_fichier)

    # et dans un csv
    # and in csv
    colonnes = ['Lemme','POS','frequence']
    with open('date_subtitles.csv','w',newline='') as output:
        writer = csv.DictWriter(output, fieldnames=colonnes)

        # en tête
        #header
        writer.writeheader()

        # écrire données
        # writting data
        for key,value in dict_lex.items():
            lemma,pos = key
            writer.writerow({'Lemme': lemma, 'POS': pos, 'frequence': value})

    # on créer des histogrammes : 
    # pour les NOUN, ADJ, PROPN et VERB du corpus mélangé (trié par fréquece d'occurrence)
    # We create histograms:
    # for the NOUN, ADJ, PROPN, and VERB of the mixed corpus (sorted by frequency of occurrence)

    new_dict={}
    for i, (key, value) in enumerate(dict_lex.items()):
        lemma, pos = key
        freq = value
        new_dict[lemma]=freq
        if i == 21:
            break
    
    # et un dictionnaire pour chaque POS enregistré (NOUN, VERB, ADJ, PROPN)
    # and a dictionary for each POS recorded (NOUN, VERB, ADJ, PROPN)
    new_dict_verb = {}
    new_dict_noun = {}
    new_dict_adj = {}
    new_dict_propn = {}
    new_dict = {}
    # Limite pour chaque dictionnaire
    # dictionary limits
    limite = 20

    for key, value in dict_lex.items():
        lemma, pos = key
        freq = value

        if pos == 'VERB' and len(new_dict_verb) < limite:
            new_dict_verb[lemma] = freq

        elif pos == 'NOUN' and len(new_dict_noun) < limite:
            new_dict_noun[lemma] = freq

        elif pos == 'ADJ' and len(new_dict_adj) < limite:
            new_dict_adj[lemma] = freq

        elif pos == 'PROPN' and len(new_dict_propn) < limite:
            new_dict_propn[lemma] = freq
        
        # Vérifier si toutes les limites sont atteintes
        # check the limits
        if len(new_dict_verb) == limite and len(new_dict_noun) == limite and len(new_dict_adj) == limite and len(new_dict_propn) == limite and len(new_dict)==limite:
            break

    # liste des sous-dictionnaires de 20 items : 
    # list of sub-dictionaries of 20 items:
    list_dict = [new_dict,new_dict_adj,new_dict_noun,new_dict_propn,new_dict_verb]

    # création des figures - enregistrement 
    # create Figure and save them 
    for title,dictionnaire in zip(['All','ADJ','NOUN','PROPN','VERB'],list_dict):
        data = dictionnaire
        names = list(data.keys())
        values = list(data.values())

        # Créer une seule figure avec un axe
        # Create a figure with one axe
        fig, ax = plt.subplots(figsize=(9, 3))

        # Tracer l'histogramme
        # Create histogram
        ax.bar(names, values, color='#002654')
        plt.ylabel('Occurences')
        # Ajouter un titre à la figure
        # Add title
        fig.suptitle(f'Top 20 most frequent {title.lower()}')

        # Faire pivoter les étiquettes de l'axe des abscisses
        # Pivot abscisse's axe
        ax.set_xticklabels(names, rotation=45, ha='right')

        #Enregistrer la figure
        # Save figure 
        plt.savefig(f'figure_frequence_{title}.pdf',bbox_inches='tight')

    
