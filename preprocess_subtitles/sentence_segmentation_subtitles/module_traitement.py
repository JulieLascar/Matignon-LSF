import os
import re 
from datetime import datetime
import spacy
spacy.prefer_gpu()
nlp = spacy.load("fr_dep_news_trf")
from spacy.language import Language
from datetime import datetime, timedelta

"""
Ce module regroupe les différentes fonctions utiles aux traitements des fichiers .vtt. 
"""


def lister_fichiers_with_path(dossier):
    path = dossier
    try:
        # Liste de tous les fichiers dans le dossier
        fichiers = [os.path.join(path,f) for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))]
        return fichiers
    except Exception as e:
        print(f"Erreur lors de la récupération des fichiers : {e}")
        return None
    

def lister_fichiers(dossier):
    path = dossier
    try:
        # Liste de tous les fichiers dans le dossier
        fichiers = [f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))]
        return fichiers
    except Exception as e:
        print(f"Erreur lors de la récupération des fichiers : {e}")
        return None
    
def conv_str_to_time(temps_str):
    temps_format = "%H:%M:%S.%f"
    temps_obj = datetime.strptime(temps_str, temps_format)
    delta = timedelta(seconds=temps_obj.second, microseconds=temps_obj.microsecond)
    return delta


def ajouter_secondes(temps_str, secondes_a_ajouter):
    # Convertir la chaîne en objet datetime
    temps_format = "%H:%M:%S.%f"
    temps_obj = datetime.strptime(temps_str, temps_format)

    # Ajouter le nombre de secondes spécifié
    nouveau_temps_obj = temps_obj + timedelta(seconds=secondes_a_ajouter)

    # Formater le nouveau temps en chaîne avec une précision de 3 décimales
    nouveau_temps_str = nouveau_temps_obj.strftime("%H:%M:%S.%f")[:-3]

    return nouveau_temps_str


def get_dict_vtt(input):
    with open(input,encoding="utf-8") as f:
        lines = f.readlines()

    dict_sub = {}
    i = 0
    j = 0  

    while j < len(lines): 
        element = lines[j]
        if element.startswith("00:") or element.startswith("01:") or element.startswith("02:"):
            # Extraire le temps de début et de fin
            timing_line = element.strip().split(' --> ')
            start_time, end_time = timing_line

            text = ""
            while j + 1 < len(lines) and not lines[j + 1].startswith("00:") and not lines[j+1].startswith("01:") and not lines[j+1].startswith("02:"):
                j += 1
                content = lines[j]
                text = text + " " + content.strip()
                text=text.replace("[INAUDIBLE]","")
                text=text.replace("[ INAUDIBLE ]","")
                text=text.replace("... -G. Attal : ","")
                text=text.replace("-G. Attal : ","")
                text=text.replace("G. Attal : ","")

            dict_sub[i] = {'start': start_time, 'end': end_time, 'text': text.strip()}
            i += 1

        j += 1

    return dict_sub


def create_vtt_file(dictionary, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write("WEBVTT\n\n")

        for key, v in dictionary.items():
            for kk,vv in v.items():
                if kk == 'start':
                    start_time = vv
                if kk == 'end':
                    end_time = vv
                if kk == 'text':
                    text = vv

            vtt_entry = f"{start_time} --> {end_time}\n{text}\n\n"
            file.write(vtt_entry)

def comparer_listes(liste1, liste2):
    differences = []

    # Détermine la longueur maximale entre les deux listes
    longueur_maximale = max(len(liste1), len(liste2))

    # Parcours des éléments jusqu'à la longueur maximale
    for i in range(longueur_maximale):
        # Détermine les éléments à comparer, en gérant les cas où les listes ont des longueurs différentes
        element_liste1 = liste1[i] if i < len(liste1) else None
        element_liste2 = liste2[i] if i < len(liste2) else None

        if element_liste1 != element_liste2:
            # Ajoute la position et les éléments qui diffèrent à la liste des différences
            differences.append((i, element_liste1, element_liste2))

    return differences


def get_sentences(input):
    sentences = []
    with open(input, encoding="utf-8") as f:
        line = f.readline()
        while line:
            sentences.append(line.strip())
            line = f.readline()
    return sentences