# Change timestamps

Ce dossier contient le notebook `change_timecode.ipynb` et le module `module.py` permettant de créer de nouveaux sous-titres ayant des timestamps différents :
- Soit on ajoute du temps aux sous-titres afin d'estomber le décalage entre le signeur et le sous-titre
- Soit on bruite l'alignement des sous-titre (Mediapi) pour avoir une version non aligné avec le signeur (et utiliser le Transformers SAT de Hannah Bull)

Ce dossier contient également les données issues des notebooks : 
- `non_aligned_mediapi` : sous-titre original Mediapi dont l'alignement a été bruité
- `non_aligned_mediapi_sent_seg` : sous-titre Mediapi segmenté en phrase dont l'alignement a été bruité
- `cr_plus_x_sec` : sous-titre Matignon-LSF dont le temsp des sous-titre a été augmenté d'une seconde.

Les données originales utilisés pour l'article du 22/02/24 sont dans l'archive `original_timechange.zip`