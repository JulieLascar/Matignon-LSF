# Sentence segmentation subtitles

***french version below*** 

This folder contains the notebook and the module used to get the subtitles sentences based :

1. `sent_based_subtitles.ipynb`
2. `module_traitemet.py`

This folder contains an archived folder :

1. `create_new_subtitles.ipynb`: This code allows creating new subtitle files segmented into sentences (using the sentence and oversubtitle files from the `separate_sentence.ipynb` and `normalization.ipynb` notebooks) and visualizing errors for manual correction or deleting files containing errors.
2. `separate_sentence.ipynb`: A notebook for obtaining an oversubtitlted file (.vtt). It will be usefull in order to get a .vtt file with one sentence per subtitle.
-  <s> a file containing one sentence per line </s>
- <s> a file with oversubtitles to ensure only a portion of a sentence or a whole sentence per subtitle unit. </s>
3. `normalization.ipynb`: A notebook for normalizing subtitles and obtaining a file with one sentence per line, extracted from the subtitle file (.vtt).
4. `check_timestamps.ipynb`: A notebook for checking if subtitle timestamps do not present anomalies (i.e., the start is before the end timecode).

This folder also contains the Python module `module_processing.py` containing all the useful functions for both notebooks."




<br/>
<br/>

___

<br/>
<br/>

Ce dossier content le notebook et le module pour otebnir des fichiers de sous-titres segmentés en phrase : 

1. `sent_based_subtitles.ipynb`
2. `module_traitemet.py`

Ce dossier contient une archive : 
1. `create_new_subtitles.ipynb` : ce code permet de créer des nouveaux fichiers de sous-titres segmentés en phrase (grâce aux fichiers phrase et sursegmenté du notebook `separate_sentence.ipynb` et `normalisation.ipynb`) et de visualier les erreurs afin de faire une correction manuelle, ou de supprimer les fichiers contenant de erreurs. 
2. `separate_sentence.ipynb` : notebook permett d'obtenir les deux types de fichiers nécessaires pour créer de nouveaux fichiers :
    - un fichier contenant une phrase par ligne
    - <s> un fichier avec des sous-titres sursegmenté pour qu'il y ait uniquement une portion de phrase ou une phrase entière par unité de sous-titre. </s>
3. `normalisation.ipynb`: notebook permettant de normaliser les sous-titres et d'avoir un fichier avec une phrase par ligne, extrait du fichier de sous-titre (.vtt).
4. `check_timestamps.ipynb`: notebook permettant de vérifier si les timestamps des sous-titres ne présente pas d'anomali (càd : le début est antérieur au timecode de fin)

Ce dossier contient également le module python `module_traitement.py` contenant l'ensemble des fonctions utiles pour les deux notebooks. 