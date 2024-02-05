# Sentence segmentation subtitles

Ce dossier contient deux notebooks : 
1. `create_new_subtitles.ipynb` : ce code permet de créer des nouveaux fichiers de sous-titres segmentés en phrase (grâce aux fichiers pohrase et sursegmenté du notebook `separate_sentence.ipynb`) et de visualier les erreurs afin de faire une correction manuelle, ou de supprimer les fichiers contenant de erreurs. 
2. `separate_sentence.ipynb` : notebook permettat d'obtenir les deux types de fichiers nécessaires pour créer de nouveaux fichiers :
    - un fichier contenant une phrase par ligne
    - un fichier avefc des sous-titres sursegmenté pour qu'il y ait uniquement une portion de phrase ou une phrase entière par unité de sous-titre. 

Ce dossier contient également le module python `module_traitement.py` contenant l'ensemble des fonctions utiles pour les deux notebooks. 