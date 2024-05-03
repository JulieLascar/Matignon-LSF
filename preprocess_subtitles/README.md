# Preprocess subtitles


This subfolder contains three folders:

1. `change_timestamp/`: a folder containing notebooks and modules used to modify the timestamps of subtitles, either to add time (to mitigate any potential delay in French to LSF interpretation) or to add noise to subtitles (used for the Mediapi subtitle corpus to misalign the subtitles with the sign for use with Hannah Bull's Transformers SAT for fine-tuning for LSF).
2. `sentence_segmentation_subtitles/`: a folder containing scripts and modules to generate sentence-segmented subtitles.
3. `data/`: a folder containing data. Please note, this data has been manually modified (manual correction step) - it is also available in the data_orginal.zip folder.

<br/>
<br/>

___

<br/>
<br/>

*french version*

Ce sous dossier contient trois dossiers : 
1. `change_timestamp/` : dossier contenant les notebooks et les modules utilisés pour modifier les timestamps des sous-titres, soit pour ajotuer du temps (afin d'atténuer un éventuel décalage dns l'interprétation du français en LSF) soit pour bruiter des sous-titres (utilisé pour le corpus de sous-titre de Mediapi afin de désaligner les sous-titres avec le signe pour utiliser le Transformers SAT d'Hannah Bull pour le finetuner pour la LSF)
2. `sentence_segmentation_subtitles/` : dossier contenant les scripts et les modules pour générer des sous-titres segmentés en phrase
3. `data/` : dossier contenant les données. Attention, ces données ont été modifiés à la main (étape de correction manuelle) - elles se trouvent également dans le dossier `data_orginal.zip`