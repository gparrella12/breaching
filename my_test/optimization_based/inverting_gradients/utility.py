import os
import shutil

# Percorso della cartella padre contenente le immagini
cartella_padre = '/user/gparrella/data/flickr_images'

# Ottieni la lista di file presenti nella cartella padre
immagini = os.listdir(cartella_padre)

# Crea le sottodirectory per le immagini
for idx, immagine in enumerate(immagini):
    # Crea un nome univoco per la sottodirectory
    nome_sottodir = f'immagine_{idx}'  # Modifica se vuoi usare un altro formato per il nome

    # Percorso completo della nuova sottodirectory
    percorso_sottodir = os.path.join(cartella_padre, nome_sottodir)

    # Crea la sottodirectory
    os.makedirs(percorso_sottodir)

    # Sposta l'immagine nella nuova sottodirectory
    percorso_immagine_originale = os.path.join(cartella_padre, immagine)
    shutil.move(percorso_immagine_originale, percorso_sottodir)
