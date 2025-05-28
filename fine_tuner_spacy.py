import spacy
import random # Pour mélanger les données d'entraînement
import json # Pour charger les données préparées
from spacy.training.example import Example # Nécessaire pour SpaCy v3.x

# --- Configuration ---
MODELE_BASE = "fr_core_news_md"  # Modèle SpaCy pré-entraîné à fine-tuner
CHEMIN_DONNEES_ENTRAINEMENT = "donnees_entrainement_spacy.json" # Fichier généré par preparer_donnees.py
CHEMIN_MODELE_FINETUNE = "./modele_pseudonymisation_finetune" # Dossier où sauvegarder le modèle fine-tuné
NOMBRE_ITERATIONS = 10 # Nombre de passages sur l'ensemble des données d'entraînement
DROPOUT = 0.35 # Taux de dropout pour la régularisation (aide à prévenir le surapprentissage)

def charger_donnees_entrainement(chemin_fichier):
    """Charge les données d'entraînement depuis un fichier JSON."""
    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        donnees = json.load(f)
    # SpaCy s'attend à des tuples, mais JSON sauvegarde des listes.
    # Convertir les listes d'entités en tuples.
    # Exemple: ["texte", {"entities": [[0, 4, "PER"]]}] -> ("texte", {"entities": [(0, 4, "PER")]})
    donnees_formatees = []
    for texte, annotations in donnees:
        entites_formatees = []
        if "entities" in annotations:
            for debut, fin, label in annotations["entities"]:
                entites_formatees.append((debut, fin, label))
            donnees_formatees.append((texte, {"entities": entites_formatees}))
        else:
            # Gérer les cas où il n'y a pas d'entités (ne devrait pas arriver avec nos données générées)
            donnees_formatees.append((texte, annotations))
            
    return donnees_formatees

def fine_tuner_modele_spacy(donnees_entrainement):
    """
    Fine-tune un modèle SpaCy pour la reconnaissance d'entités nommées.
    """
    try:
        # Charger le modèle SpaCy pré-entraîné
        nlp = spacy.load(MODELE_BASE)
        print(f"Modèle de base '{MODELE_BASE}' chargé.")
    except OSError:
        print(f"ERREUR: Le modèle de base '{MODELE_BASE}' n'a pas été trouvé.")
        print(f"Veuillez le télécharger en utilisant la commande : python -m spacy download {MODELE_BASE}")
        return

    # Obtenir le composant NER (Reconnaissance d'Entités Nommées)
    if "ner" not in nlp.pipe_names:
        # Créer le composant NER s'il n'existe pas (peu probable pour fr_core_news_md)
        ner = nlp.add_pipe("ner", last=True)
        print("Composant NER ajouté au pipeline.")
    else:
        ner = nlp.get_pipe("ner")
        print("Composant NER existant obtenu du pipeline.")

    # Ajouter la nouvelle étiquette (label) au composant NER si elle n'existe pas.
    # Pour "PER", elle devrait déjà exister dans fr_core_news_md.
    # Cette boucle s'assure que toutes les étiquettes présentes dans nos données sont connues du NER.
    for _, annotations in donnees_entrainement:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2]) # ent[2] est le label, par exemple "PER"

    # Commencer l'entraînement
    # Désactiver les autres composants du pipeline qui ne sont pas nécessaires pour le fine-tuning du NER
    # Cela accélère l'entraînement.
    pipes_a_desactiver = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    with nlp.select_pipes(disable=pipes_a_desactiver): # Désactive temporairement les autres pipes
        print(f"Début du fine-tuning du composant NER pour {NOMBRE_ITERATIONS} itérations...")
        optimizer = nlp.begin_training() # Crée un optimiseur
        
        for iteration in range(NOMBRE_ITERATIONS):
            random.shuffle(donnees_entrainement) # Mélanger les données à chaque itération
            pertes = {} # Pour suivre les erreurs (pertes)
            
            # Diviser les données en lots (batches) peut être plus efficace pour de grands datasets
            # Ici, pour simplifier, nous traitons les exemples un par un, mais spacy.util.minibatch peut être utilisé.
            for texte, annotations in donnees_entrainement:
                try:
                    doc = nlp.make_doc(texte) # Crée un objet Doc à partir du texte
                    example = Example.from_dict(doc, annotations) # Crée un objet Example
                    # Mettre à jour le modèle
                    nlp.update([example], sgd=optimizer, drop=DROPOUT, losses=pertes)
                except Exception as e:
                    print(f"Erreur lors de la mise à jour avec le texte : '{texte[:50]}...'")
                    print(f"Annotations : {annotations}")
                    print(f"Erreur : {e}")
                    continue # Passe à l'exemple suivant

            print(f"Itération {iteration + 1}/{NOMBRE_ITERATIONS} - Pertes : {pertes.get('ner', 'N/A')}")

    # Sauvegarder le modèle fine-tuné dans le dossier spécifié
    nlp.to_disk(CHEMIN_MODELE_FINETUNE)
    print(f"\nModèle fine-tuné sauvegardé avec succès dans : '{CHEMIN_MODELE_FINETUNE}'")
    print("Vous pouvez maintenant charger ce modèle en utilisant spacy.load(CHEMIN_MODELE_FINETUNE)")

# --- Programme Principal ---
if __name__ == "__main__":
    print("Chargement des données d'entraînement...")
    TRAIN_DATA = charger_donnees_entrainement(CHEMIN_DONNEES_ENTRAINEMENT)
    
    if TRAIN_DATA:
        print(f"{len(TRAIN_DATA)} exemples d'entraînement chargés.")
        fine_tuner_modele_spacy(TRAIN_DATA)
    else:
        print("Aucune donnée d'entraînement trouvée. Veuillez d'abord exécuter le script de préparation des données.")