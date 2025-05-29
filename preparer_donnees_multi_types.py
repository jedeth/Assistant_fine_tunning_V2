import re
import json
import os
# from tkinter import messagebox # Retiré car c'est un script CLI pour l'instant. On utilise print.

# --- Fonctions de base ---

def lire_entites_depuis_fichier(chemin_fichier):
    """
    Lit une liste d'entités (une par ligne) depuis un fichier.
    """
    entites = []
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            for ligne in f:
                entite_texte = ligne.strip()
                if entite_texte:
                    entites.append(entite_texte)
        if not entites:
            print(f"Attention : Aucun contenu trouvé dans le fichier d'entités : {chemin_fichier}")
        return entites
    except FileNotFoundError:
        print(f"ERREUR : Le fichier d'entités '{chemin_fichier}' est introuvable.")
        return None
    except Exception as e:
        print(f"ERREUR : Erreur lors de la lecture du fichier d'entités '{chemin_fichier}': {e}")
        return None

def lire_phrases_modeles_specifiques(chemin_fichier_phrases, placeholder_attendu):
    """
    Lit les phrases modèles depuis un fichier.
    Vérifie la présence d'un placeholder spécifique (ex: {LOC} ou {ORG}).
    """
    phrases = []
    try:
        with open(chemin_fichier_phrases, 'r', encoding='utf-8') as f:
            for ligne_num, ligne in enumerate(f, 1):
                phrase_modele = ligne.strip()
                
                # --- DÉBUT DE LA CORRECTION ---
                # Nettoyage simple des guillemets et virgules (si copiées d'une liste Python)
                if phrase_modele.startswith('"') and phrase_modele.endswith('",'):
                    phrase_modele = phrase_modele[1:-2]
                elif phrase_modele.startswith("'") and phrase_modele.endswith("',"):
                    phrase_modele = phrase_modele[1:-2]
                elif phrase_modele.startswith('"') and phrase_modele.endswith('"'):
                    phrase_modele = phrase_modele[1:-1]
                elif phrase_modele.startswith("'") and phrase_modele.endswith("'"):
                    phrase_modele = phrase_modele[1:-1]
                # --- FIN DE LA CORRECTION ---

                if phrase_modele and placeholder_attendu in phrase_modele:
                    phrases.append(phrase_modele)
                elif phrase_modele: # Si la ligne n'est pas vide après nettoyage mais manque le placeholder
                    print(f"Attention (ligne {ligne_num} de {chemin_fichier_phrases}): La phrase ne contient pas le placeholder '{placeholder_attendu}' ou est vide après nettoyage : '{ligne.strip()}'")
        if not phrases:
            print(f"Attention : Aucun modèle de phrase valide trouvé dans : {chemin_fichier_phrases} pour le placeholder '{placeholder_attendu}'")
        return phrases
    except FileNotFoundError:
        print(f"ERREUR : Le fichier de modèles de phrases '{chemin_fichier_phrases}' est introuvable.")
        return None
    except Exception as e:
        print(f"ERREUR : Erreur lors de la lecture du fichier de modèles de phrases '{chemin_fichier_phrases}': {e}")
        return None

def generer_donnees_pour_type(liste_entites, liste_phrases_modeles, label_entite, placeholder):
    """
    Génère les données d'entraînement au format SpaCy pour un type d'entité spécifique.
    """
    donnees_entrainement = []
    if not liste_entites or not liste_phrases_modeles:
        print(f"Impossible de générer des données pour le label '{label_entite}' car la liste d'entités ou de phrases modèles est vide/invalide.")
        return donnees_entrainement

    for entite_texte in liste_entites:
        for phrase_modele in liste_phrases_modeles:
            phrase_formatee = phrase_modele.replace(placeholder, entite_texte)
            
            match = re.search(re.escape(entite_texte), phrase_formatee)
            if match:
                debut, fin = match.span()
                entite_spacy = (debut, fin, label_entite)
                donnees_entrainement.append((phrase_formatee, {"entities": [entite_spacy]}))
            else:
                print(f"Attention : Impossible de trouver l'entité '{entite_texte}' (label: {label_entite}) dans la phrase générée '{phrase_formatee}' avec le placeholder '{placeholder}'.")
    
    return donnees_entrainement

def sauvegarder_donnees_json(donnees, chemin_fichier_sortie):
    """
    Sauvegarde les données générées dans un fichier JSON.
    """
    if not donnees:
        print(f"Aucune donnée à sauvegarder pour '{chemin_fichier_sortie}'.")
        return False
    try:
        with open(chemin_fichier_sortie, "w", encoding="utf-8") as outfile:
            json.dump(donnees, outfile, ensure_ascii=False, indent=4)
        print(f"{len(donnees)} exemples d'entraînement sauvegardés dans '{chemin_fichier_sortie}'")
        return True
    except Exception as e:
        print(f"ERREUR : Impossible de sauvegarder les données dans '{chemin_fichier_sortie}': {e}")
        return False

# --- Programme Principal ---
if __name__ == "__main__":
    print("Outil de génération de données d'entraînement SpaCy pour types d'entités multiples.\n")

    # Définition des types d'entités à traiter
    # Assurez-vous que les fichiers listés ici existent ou modifiez les noms de fichiers.
    types_a_generer = [
        {
            "label": "LOC",
            "placeholder": "{LOC}",
            "fichier_entites": "lieux.txt", 
            "fichier_phrases": "phrases_modeles_lieux.txt", 
            "fichier_sortie_json": "donnees_entrainement_loc.json"
        },
        {
            "label": "ORG",
            "placeholder": "{ORG}",
            "fichier_entites": "organisations.txt", 
            "fichier_phrases": "phrases_modeles_org.txt", 
            "fichier_sortie_json": "donnees_entrainement_org.json"
        },
        # Vous pouvez ajouter d'autres types ici. Par exemple, pour les noms de personnes :
        # {
        #     "label": "PER",
        #     "placeholder": "{NOM}", # ou {PER} si vous préférez
        #     "fichier_entites": "annuaire_pour_finetuning.txt", # Fichier que vous utilisiez avant
        #     "fichier_phrases": "phrases_modeles.txt", # Fichier que vous utilisiez avant
        #     "fichier_sortie_json": "donnees_entrainement_per.json"
        # }
    ]

    donnees_combinees_pour_fine_tuning = []

    # Charger d'abord les données PER existantes si elles ont été générées par l'ancien script
    # et que vous ne les régénérez pas avec celui-ci.
    # Exemple :
    # if os.path.exists("donnees_entrainement_per.json"): # Remplacez par le nom de votre fichier PER
    #     print("Chargement des données PER pré-existantes...")
    #     with open("donnees_entrainement_per.json", "r", encoding="utf-8") as f_per:
    #         donnees_per = json.load(f_per)
    #         donnees_combinees_pour_fine_tuning.extend(donnees_per)
    #         print(f"{len(donnees_per)} exemples PER chargés.")


    for config_type in types_a_generer:
        print(f"\n--- Traitement du type d'entité : {config_type['label']} ---")
        
        entites = lire_entites_depuis_fichier(config_type["fichier_entites"])
        if entites is None: # Si le fichier n'existe pas ou erreur de lecture
            print(f"Échec de la lecture des entités pour {config_type['label']}. Ce type sera ignoré.")
            continue # Passe au type d'entité suivant
            
        phrases_modeles = lire_phrases_modeles_specifiques(config_type["fichier_phrases"], config_type["placeholder"])
        if phrases_modeles is None: # Si le fichier n'existe pas ou erreur de lecture
            print(f"Échec de la lecture des phrases modèles pour {config_type['label']}. Ce type sera ignoré.")
            continue # Passe au type d'entité suivant

        if not entites or not phrases_modeles: # Si les listes sont vides après lecture
            print(f"Pas assez de données (entités ou phrases valides) pour générer les exemples pour {config_type['label']}.")
            continue
            
        print(f"Génération des données pour {config_type['label']}...")
        donnees_generees = generer_donnees_pour_type(entites, phrases_modeles, config_type["label"], config_type["placeholder"])
        
        if donnees_generees:
            sauvegarder_donnees_json(donnees_generees, config_type["fichier_sortie_json"])
            donnees_combinees_pour_fine_tuning.extend(donnees_generees)
        else:
            print(f"Aucune donnée n'a été générée pour {config_type['label']}.")

    # Sauvegarder toutes les données combinées dans un unique fichier pour le fine-tuning
    if donnees_combinees_pour_fine_tuning:
        chemin_fichier_combine = "donnees_entrainement_combinees.json"
        print(f"\nSauvegarde de toutes les données combinées ({len(donnees_combinees_pour_fine_tuning)} exemples)...")
        sauvegarder_donnees_json(donnees_combinees_pour_fine_tuning, chemin_fichier_combine)
    else:
        print("\nAucune donnée combinée à sauvegarder. Vérifiez vos fichiers d'entrée et les logs.")
        
    print("\nTerminé.")