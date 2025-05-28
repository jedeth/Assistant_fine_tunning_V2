import spacy
import json
import argparse # Pour gérer les arguments de la ligne de commande
import os

# Chemin par défaut vers ton modèle fine-tuné
CHEMIN_MODELE_PAR_DEFAUT = "./modele_pseudonymisation_finetune" 

def charger_modele_spacy(chemin_modele):
    """Charge le modèle SpaCy fine-tuné."""
    if not os.path.exists(chemin_modele):
        print(f"ERREUR : Le dossier du modèle '{chemin_modele}' n'a pas été trouvé.")
        print("Veuillez vérifier le chemin ou entraîner le modèle d'abord.")
        return None
    try:
        nlp = spacy.load(chemin_modele)
        print(f"Modèle SpaCy chargé depuis '{chemin_modele}'")
        return nlp
    except Exception as e:
        print(f"Erreur lors du chargement du modèle SpaCy depuis '{chemin_modele}': {e}")
        return None

def lire_fichier_texte(chemin_fichier):
    """Lit le contenu d'un fichier texte."""
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERREUR : Le fichier d'entrée '{chemin_fichier}' n'a pas été trouvé.")
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier '{chemin_fichier}': {e}")
        return None

def ecrire_fichier_texte(texte, chemin_fichier):
    """Écrit du texte dans un fichier."""
    try:
        with open(chemin_fichier, 'w', encoding='utf-8') as f:
            f.write(texte)
        print(f"Texte pseudonymisé sauvegardé dans : '{chemin_fichier}'")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier '{chemin_fichier}': {e}")

def ecrire_fichier_json(dictionnaire, chemin_fichier):
    """Écrit un dictionnaire dans un fichier JSON."""
    try:
        with open(chemin_fichier, 'w', encoding='utf-8') as f:
            json.dump(dictionnaire, f, ensure_ascii=False, indent=4)
        print(f"Table de correspondance sauvegardée dans : '{chemin_fichier}'")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier JSON '{chemin_fichier}': {e}")

def pseudonymiser_texte(nlp, texte_original):
    """
    Pseudonymise le texte en utilisant le modèle SpaCy et retourne le texte modifié
    ainsi que la table de correspondance.
    """
    doc = nlp(texte_original)
    
    texte_modifie = list(texte_original) # Convertit en liste de caractères pour modification facile
    correspondances = {}
    pseudonyme_compteur = 1
    
    entites_a_remplacer = []

    for entite in doc.ents:
        if entite.label_ == "PER": # Si l'entité est une personne
            nom_original = entite.text
            
            if nom_original not in correspondances:
                pseudonyme = f"[PERSONNE_{pseudonyme_compteur}]"
                correspondances[nom_original] = pseudonyme
                pseudonyme_compteur += 1
            else:
                pseudonyme = correspondances[nom_original]
            
            # On stocke l'entité et son pseudonyme pour le remplacement
            # (start_char, end_char, pseudonyme)
            entites_a_remplacer.append((entite.start_char, entite.end_char, pseudonyme))

    # Trier les entités par position de début en ordre inverse
    # pour éviter les problèmes d'indices lors du remplacement
    entites_a_remplacer.sort(key=lambda x: x[0], reverse=True)
    
    nouveau_texte_parts = []
    dernier_index_traite = len(texte_original)

    for start_char, end_char, pseudonyme in entites_a_remplacer:
        # Ajouter la partie du texte après la dernière entité remplacée (ou depuis la fin)
        if end_char < dernier_index_traite:
            nouveau_texte_parts.append(texte_original[end_char:dernier_index_traite])
        # Ajouter le pseudonyme
        nouveau_texte_parts.append(pseudonyme)
        dernier_index_traite = start_char
    
    # Ajouter la partie initiale du texte (avant la première entité)
    nouveau_texte_parts.append(texte_original[0:dernier_index_traite])
    
    # Reconstruire le texte final en inversant l'ordre des parties
    texte_pseudonymise_final = "".join(reversed(nouveau_texte_parts))
            
    return texte_pseudonymise_final, correspondances

# --- Programme Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudonymise un fichier texte en utilisant un modèle SpaCy fine-tuné.")
    parser.add_argument("--input", required=True, help="Chemin vers le fichier .txt à pseudonymiser.")
    parser.add_argument("--output_txt", required=True, help="Chemin vers le fichier .txt de sortie (texte pseudonymisé).")
    parser.add_argument("--output_json", required=True, help="Chemin vers le fichier .json de sortie (table de correspondance).")
    parser.add_argument("--modele", default=CHEMIN_MODELE_PAR_DEFAUT, help=f"Chemin vers le dossier du modèle SpaCy fine-tuné (défaut: {CHEMIN_MODELE_PAR_DEFAUT}).")
    
    args = parser.parse_args()
    
    # 1. Charger le modèle
    nlp_modele = charger_modele_spacy(args.modele)
    if not nlp_modele:
        exit() # Arrête le script si le modèle ne peut pas être chargé
        
    # 2. Lire le fichier d'entrée
    texte_a_traiter = lire_fichier_texte(args.input)
    if not texte_a_traiter:
        exit()
        
    # 3. Pseudonymiser
    print("Pseudonymisation en cours...")
    texte_resultat, table_correspondance = pseudonymiser_texte(nlp_modele, texte_a_traiter)
    
    # 4. Écrire les fichiers de sortie
    ecrire_fichier_texte(texte_resultat, args.output_txt)
    ecrire_fichier_json(table_correspondance, args.output_json)
    
    print("\nPseudonymisation terminée !")