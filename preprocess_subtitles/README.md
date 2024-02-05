# Preprocess subtitles

Ce sous dossier contient trois dossiers : 
1. `change_timestamp/` : dossier contenant les notebooks et les modules utilisés pour modifier les timestamps des sous-titres, soit pour ajotuer du temps (afin d'atténuer un éventuel décalage dns l'interprétation du français en LSF) soit pour bruiter des sous-titres (utilisé pour le corpus de sous-titre de Mediapi afin de désaligner les sous-titres avec le signe pour utiliser le Transformers SAT d'Hannah Bull pour le finetuner pour la LSF)
2. `sentence_segmentation_subtitles/` : dossier contenant les scripts et les modules pour générer des sous-titres segmentés en phrase
3. `data/` : dossier contenant les données. Attention, ces données ont été modifiés à la main (étape de correction manuelle) - elles se trouvent également dans le dossier `data_orginal.zip`