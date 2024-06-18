import numpy as np



data_retrieved = np.array((147807.62806815607, 6839347.824555168, 89.94688461538462))
data_og = np.array((147807.631, 6839347.831, 89.91))

diff = data_retrieved[:2] - data_og[:2]
# Calculer la racine carrée de la somme des carrés des éléments de diff
result = np.sqrt(np.sum(diff**2))

diff2 = data_retrieved - data_og
# Calculer la racine carrée de la somme des carrés des éléments de diff
result2 = np.sqrt(np.sum(diff2**2))

# Afficher le résultat
print("Résultats différence point numéro 5 et données récupérées par serial")
print(f"Erreur (sans prendre en compte altitude): {result*100} cm")
print(f"Erreur (avec altitude): {result2*100} cm")