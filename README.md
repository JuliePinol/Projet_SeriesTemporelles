# Projet_SeriesTemporelles
Analyse de données de température issues de capteurs dans des chambres froides - Transformée en rondelettes + clustering

Analyse_jour.py permet la transformation des datasets avec une segmentation par jour complet (on exclue les jours avec des données manquantes)

Transfo_ondelettes_v2.py prend en entrée les datasets normalisés par analyse_jour.py et renvoie le clustering de la composante basale issue de la transformée en ondelettes discrètes

Affichage_cluster.py affiche la température moyenne de chaque jour et la colore de la couleur du cluster (outil de visualisation)
