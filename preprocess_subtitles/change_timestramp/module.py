import re
from datetime import datetime, timedelta
import os 
import argparse
from collections import defaultdict,Counter
import json 
import csv
from tqdm import tqdm
from typing import List
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


"""
Fonctions utilisées pour le changement de timestamps des sous-titres
"""

def lister_fichiers(chemin: str) -> list:
    """
    La fonction prend en entrée le nom d'un dossier et renvoie la liste des fichiers qu'il contient.

    Args:
        chemin (str): nom du dossier

    Returns:
        list : liste des chemins des fichiers contenus dans le dossier

    Examples:
        lister_fichiers("subtitles") = ["subtitles/files.json",...]
    """
    try:
        if os.path.isdir(chemin):
            # Si le chemin est un dossier, retourner la liste des fichiers dans le dossier
            fichiers = [os.path.join(chemin, f) for f in os.listdir(chemin) if os.path.isfile(os.path.join(chemin, f))]
            return fichiers
    except Exception as e:
        print(f"Erreur : {e}")
        return []


def convertir_chaine_en_temps(temps_str:str)-> datetime:
    # Formatter la chaîne de caractères en timedelta
    temps_delta = datetime.strptime(temps_str, "%H:%M:%S.%f")

    # Extraire l'heure, les minutes, les secondes et les microsecondes
    heures, minutes, secondes = temps_delta.hour, temps_delta.minute, temps_delta.second
    microsecondes = temps_delta.microsecond

    # Formater la sortie pour afficher uniquement l'heure, les minutes, les secondes et les millisecondes
    temps_formate = f"{heures:02d}:{minutes:02d}:{secondes:02d}.{microsecondes // 1000:03d}"

    return temps_formate

def ajouter_une_seconde(temps_str:str)-> datetime:
    #convertir str en time : 

    temps_objet = datetime.strptime(temps_str, "%H:%M:%S.%f").time()

    # Convertir l'objet time en datetime pour pouvoir effectuer l'opération d'ajout
    temps_datetime = datetime.combine(datetime.min, temps_objet)

    # Ajouter 1 seconde à l'objet datetime
    temps_datetime = temps_datetime + timedelta(seconds=1)

    # Extraire l'objet time du résultat
    temps_objet_modifie = temps_datetime.time()

    #Formater la sortie (00:00:00.000)
    temps_formate = temps_objet_modifie.strftime("%H:%M:%S.%f")[:-3]


    return temps_formate



def convertir_temps(chaine_temps):
    # Convertir la chaîne de temps en un objet timedelta
    duree = timedelta(seconds=float(chaine_temps))

    # Formater la durée sous forme de chaîne dans le format "00:00:00.000"
    temps_formate = str(duree)

    # Ajouter des zéros pour remplir les champs manquants
    parties_temps = temps_formate.split(".")

    # Vérifier si la liste a une partie fractionnaire
    if len(parties_temps) > 1:
        heures, minutes, secondes = map(int, parties_temps[0].split(":"))
        millisecondes = parties_temps[1][:3].ljust(3, '0')
    else:
        heures, minutes, secondes = map(int, temps_formate.split(":"))
        millisecondes = "000"

    temps_final = f"{heures:02d}:{minutes:02d}:{secondes:02d}.{millisecondes}"

    return temps_final


def change_timecode(temps_str:str,decalage:float)->str:
    #convertir str en time : 
    temps_objet = datetime.strptime(temps_str, "%H:%M:%S.%f").time()

    # Convertir l'objet time en datetime pour pouvoir effectuer l'opération d'ajout
    temps_datetime = datetime.combine(datetime.min, temps_objet)

    # Ajouter décalage à l'objet timedelta
    temps_datetime = temps_datetime + timedelta(seconds=decalage)

    # Extraire l'objet time du résultat
    temps_objet_modifie = temps_datetime.time()

    #Formater la sortie (00:00:00.000)
    temps_formate = temps_objet_modifie.strftime("%H:%M:%S.%f")[:-3]


    return temps_formate