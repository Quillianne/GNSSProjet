from osgeo import gdal

# Ouvrir l'image en utilisant GDAL
im = gdal.Open('ensta_2015.jpg')

# Vérifier la géotransformation
geo_transform = im.GetGeoTransform()
if geo_transform:
    print("Géotransformation:", geo_transform)
else:
    print("Aucune géotransformation trouvée.")

# Vérifier le système de coordonnées
projection = im.GetProjection()
if projection:
    print("Système de coordonnées:", projection)
else:
    print("Aucun système de coordonnées trouvé.")