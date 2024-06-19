import os
import re 
from datetime import datetime
import spacy
spacy.prefer_gpu()
nlp = spacy.load("fr_dep_news_trf")
from spacy.language import Language
from datetime import datetime, timedelta
import shutil

"""
This module gathers various functions useful for processing .vtt files.
"""


def time_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, seconds, and milliseconds
    milliseconds = int(timestamp.split('.')[1])
    tmp = timestamp.split('.')[0]
    hours, minutes, seconds = map(int, tmp.split(':'))

    # Calculate the total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    return total_seconds

def time_to_milliseconds(timestamp):

    # Split the timestamp into hours, minutes, seconds, and milliseconds

    milliseconds = int(timestamp.split('.')[1])

    tmp = timestamp.split('.')[0]

    hours, minutes, seconds = map(int, tmp.split(':'))

    #seconds, milliseconds = map(int, seconds.split('.'))

     

    # Calculate the total milliseconds

    total_milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds

    return total_milliseconds

def time_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, seconds, and milliseconds
    milliseconds = int(timestamp.split('.')[1])
    tmp = timestamp.split('.')[0]
    hours, minutes, seconds = map(int, tmp.split(':'))

    # Calculate the total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    return total_seconds


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

def convertir_grand_nombre(nombre_texte):
    nombre_sans_points = re.sub(r'(\d)\.(\d)', r'\1\2', nombre_texte)
    return nombre_sans_points

def remplacer_points_adresses_email(texte):
    # Expression régulière pour repérer les adresses e-mail
    pattern = r'\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b'

    # Fonction de remplacement pour remplacer les points par "POINT"
    def remplacer(match):
        return match.group().replace('.', 'POINT')

    texte_modifie = re.sub(pattern, remplacer, texte)
    return texte_modifie

def remplacer_points_adresses(texte):
    # Expression régulière pour repérer différentes formes d'adresses de site internet
    pattern = r'\b(?:https?://)?(?:www\.)?[\w.-]+\.[a-zA-Z]{2,}\b'

    # Fonction de remplacement pour remplacer les points par "POINT"
    def remplacer(match):
        return match.group().replace('.', 'POINT')

    texte_modifie = re.sub(pattern, remplacer, texte)
    return texte_modifie

def normaliser_points_de_suspension(texte):
    # Ajouter un espace après les points de suspension suivis d'une lettre
    texte_modifie = re.sub(r'\.\.\.(\w)', r'... \1', texte)

    # Retirer un espace avant les points de suspension précédés d'une lettre
    texte_modifie = re.sub(r'(\w)\s*\.\.\.', r'\1...', texte_modifie)

    return texte_modifie

def remplacer_ponctuation_html(texte):
    substitutions = [
        ("(", "&#40;"),
        (")", "&#41;"),
        ("?", "&#63;"),
        ("!", "&#33;"),
        (".", "&#46;"),
        ("...", "&#8230;")
    ]

    def remplacer_match(match):
        contenu_parentheses = match.group(1)
        for recherche, remplacement in substitutions:
            contenu_parentheses = contenu_parentheses.replace(recherche, remplacement)
        return f'({contenu_parentheses})'

    texte_modifie = re.sub(r'\(([^)]*)\)', remplacer_match, texte)
    return texte_modifie

def get_dict_vtt_clean(input):
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
                text = normaliser_points_de_suspension(text)
                text = text.replace("(...)","[SUSPENSIONP]") #régler ce problème.
                text = remplacer_ponctuation_html(text)
                ### changement
                text = text.replace("…","...")
                # text = text.replace("... ...","[DOUBLE_SUSPENSION]")
                #text = text.replace(" ... ","[SUSPENSION].") ### gérer la double suspension !
                text = re.sub(r'["“”«»]', '', text)
                text = text.replace("(???)","[INTERROGATION3]")
                text = text.replace("(?)","[INTERROGATION1]")
                text = text.replace("(??)","[INTERROGATION2]")
                text = text.replace("( ?)","[INTERROGATION1]")
                text = text.replace("?,","[INTERROGATION],")
                text = text.replace("!,","[EXCLAMATION],")
                text = text.replace("etc.,","etc,")
                text = text.replace("etc.)","etc)")
                text = text.replace("Etc","etc")
                text = text.replace("PAM !","[PAM]")
                text = text.replace("Média'Pi!","[NOM_MEDIA]")
                text = text.replace("Média'Pi !","[NOM_MEDIA]")
                text = text.replace("Media'Pi !","[NOM_MEDIA]")
                text = text.replace("Média'Pi&nbsp;!","[NOM_MEDIA]")
                text = text.replace("Média' Pi !","[NOM_MEDIA]")
                text = text.replace(".e.s","")
                text = text.replace(".ne.s","")
                text = text.replace(".e.","")
                text = text.replace(".e","")
                text = text.replace("!!","!")
                text = text.replace("??","?")
                text = re.sub(r'\b(\w+)\s*\.\.\.\s*(\w+)\b', r'\1... \2', text)
                text = remplacer_points_adresses_email(text)
                text = remplacer_points_adresses(text)
                text = text.replace(".,","[POINT],")
                text = text.replace("y.a","y a")
                ### 
                text=text.replace("... -G. Attal : ","")
                text=text.replace("-G. Attal : ","")
                text=text.replace("G. Attal : ","")
                text = text.replace("-Bonjour", "Bonjour")
                text = text.replace("Bonjour.", "Bonjour,")
                text = text.replace(" M."," Monsieur")
                text = text.replace(" Mme"," Madame")
                text = text.replace("-"," ")
                text = convertir_grand_nombre(text)
                # text = text.replace("(","&#40;")
                # text = text.replace(")","&#41;")
                

            dict_sub[i] = {'start': start_time, 'end': end_time, 'text': text.strip()}
            i += 1

        j += 1

    return dict_sub



def get_dict_vtt_clean_matignon(input):
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
                text=text.replace("... -G. Attal : ","")
                text=text.replace("-G. Attal : ","")
                text=text.replace("G. Attal : ","")
                text = text.replace("-Bonjour", "Bonjour")
                text = text.replace("Bonjour.", "Bonjour,")
                text = text.replace(" M."," Monsieur")
                text = text.replace(" Mme"," Madame")
                text = text.replace("-"," ")
                text = convertir_grand_nombre(text)
                # text = text.replace("(","&#40;")
                # text = text.replace(")","&#41;")
                

            dict_sub[i] = {'start': start_time, 'end': end_time, 'text': text.strip()}
            i += 1

        j += 1

    return dict_sub


def segmenter_texte_en_phrases(texte):
    # Utilisation de l'expression régulière pour diviser le texte en phrases
    phrases = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', texte)
    return phrases

def convertir_grand_nombre(nombre_texte):
    nombre_sans_points = re.sub(r'(\d)\.(\d)', r'\1\2', nombre_texte)
    return nombre_sans_points


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



def convertir_chaine_en_temps(temps_str:str)-> datetime:
    # Formatter la chaîne de caractères en timedelta
    temps_delta = datetime.strptime(temps_str, "%H:%M:%S.%f")

    # Extraire l'heure, les minutes, les secondes et les microsecondes
    heures, minutes, secondes = temps_delta.hour, temps_delta.minute, temps_delta.second
    microsecondes = temps_delta.microsecond

    # Formater la sortie pour afficher uniquement l'heure, les minutes, les secondes et les millisecondes
    temps_formate = f"{heures:02d}:{minutes:02d}:{secondes:02d}.{microsecondes // 1000:03d}"

    return temps_formate


def deplacer_fichiers(chemins_source, dossier_destination):
    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(dossier_destination):
        print(f"Le dossier de destination '{dossier_destination}' n'existe pas. Création en cours...")
        os.makedirs(dossier_destination)

    # Boucler à travers les chemins de fichiers et les déplacer vers le dossier de destination
    for chemin_source in chemins_source:
        # Extraire le nom du fichier du chemin source
        nom_fichier = os.path.basename(chemin_source)
        chemin_destination = os.path.join(dossier_destination, nom_fichier)

        # Déplacer le fichier
        shutil.move(chemin_source, chemin_destination)
        print(f"Le fichier '{nom_fichier}' a été déplacé vers '{dossier_destination}'.")

def verifier_ou_creer_dossier(chemin):
    """
    Vérifie si un dossier existe, et s'il n'existe pas, le crée.

    Args:
    chemin (str): Le chemin du dossier à vérifier ou créer.

    Returns:
    str: Un message indiquant si le dossier existait déjà ou a été créé.
    """
    if not os.path.exists(chemin):
        os.makedirs(chemin)
        return f"Le dossier '{chemin}' a été créé."
    else:
        return f"Le dossier '{chemin}' existe déjà."