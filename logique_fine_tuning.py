# logique_fine_tuning.py

import spacy
from spacy.training.example import Example
import random
import os
import json
import traceback # Pour un logging d'erreur plus détaillé

def charger_donnees_entrainement_json(chemin_fichier_json, log_callback=None):
    """
    Charge les données d'entraînement depuis un fichier JSON.
    Retourne les données formatées ou None en cas d'erreur.
    """
    try:
        with open(chemin_fichier_json, 'r', encoding='utf-8') as f:
            donnees_brutes = json.load(f)
        
        if not isinstance(donnees_brutes, list):
            if log_callback: log_callback(f"ERREUR: Le fichier JSON '{chemin_fichier_json}' doit contenir une liste d'exemples.")
            return None

        donnees_formatees = []
        for i, item in enumerate(donnees_brutes):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                if log_callback: log_callback(f"ERREUR: Exemple {i} dans '{chemin_fichier_json}' n'est pas une liste/tuple de 2 éléments.")
                return None 
            
            texte, annotations_dict = item
            
            if not isinstance(texte, str):
                if log_callback: log_callback(f"ERREUR: Texte de l'exemple {i} n'est pas une chaîne.")
                return None
            if not isinstance(annotations_dict, dict):
                if log_callback: log_callback(f"ERREUR: Annotations de l'exemple {i} ne sont pas un dictionnaire.")
                return None

            entites_tuples = []
            if "entities" in annotations_dict:
                if not isinstance(annotations_dict["entities"], list):
                    if log_callback: log_callback(f"ERREUR: 'entities' de l'exemple {i} doit être une liste.")
                    return None
                
                for j, ent_item in enumerate(annotations_dict["entities"]):
                    if not (isinstance(ent_item, (list, tuple)) and len(ent_item) == 3):
                        if log_callback: log_callback(f"ERREUR: Entité {j} de l'exemple {i} n'est pas une liste/tuple de 3 éléments.")
                        return None
                    if not (isinstance(ent_item[0], int) and isinstance(ent_item[1], int)):
                        if log_callback: log_callback(f"ERREUR: Indices de l'entité {j} (exemple {i}) ne sont pas des entiers.")
                        return None
                    if not isinstance(ent_item[2], str):
                        if log_callback: log_callback(f"ERREUR: Label de l'entité {j} (exemple {i}) n'est pas une chaîne.")
                        return None
                    entites_tuples.append(tuple(ent_item))
            
            donnees_formatees.append((texte, {"entities": entites_tuples}))

        if not donnees_formatees and donnees_brutes:
             if log_callback: log_callback(f"Attention : Aucune donnée d'entraînement valide formatée depuis '{chemin_fichier_json}'. Vérifiez la structure.")
             return None
        elif not donnees_formatees:
             if log_callback: log_callback(f"Attention : Le fichier '{chemin_fichier_json}' est vide ou ne contient pas de données.")
             return None
        return donnees_formatees
        
    except FileNotFoundError:
        if log_callback: log_callback(f"ERREUR : Fichier de données d'entraînement '{chemin_fichier_json}' introuvable.")
        return None
    except json.JSONDecodeError:
        if log_callback: log_callback(f"ERREUR : Le fichier '{chemin_fichier_json}' n'est pas un JSON valide.")
        return None
    except Exception as e:
        if log_callback: 
            log_callback(f"ERREUR critique lors du chargement/formatage des données depuis '{chemin_fichier_json}': {e}")
            log_callback(traceback.format_exc())
        return None


def executer_fine_tuning(nom_modele_base, chemin_donnees_entrainement, chemin_sauvegarde_modele, 
                         iterations, dropout_rate, log_callback):
    """
    Exécute le processus de fine-tuning de SpaCy.
    Retourne (True, chemin_sauvegarde_modele) en cas de succès, (False, None) sinon.
    """
    try:
        if not log_callback: # Fallback si aucun callback n'est fourni (pourrait être utilisé en CLI pur)
            def print_log(msg): print(msg)
            log_callback = print_log

        log_callback(f"Chargement des données d'entraînement depuis : {os.path.basename(chemin_donnees_entrainement)}")
        TRAIN_DATA = charger_donnees_entrainement_json(chemin_donnees_entrainement, log_callback)
        if not TRAIN_DATA:
            # Le message d'erreur est déjà géré par charger_donnees_entrainement_json via log_callback
            return False, None

        log_callback(f"Chargement du modèle de base SpaCy : {nom_modele_base}...")
        nlp = spacy.load(nom_modele_base)
        log_callback(f"Modèle '{nom_modele_base}' chargé.")

        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
            log_callback("Composant NER ajouté au pipeline.")
        else:
            ner = nlp.get_pipe("ner")
            log_callback("Composant NER existant obtenu du pipeline.")

        labels_dans_donnees = set()
        for _, annotations in TRAIN_DATA:
            for ent in annotations.get("entities", []): # S'assurer que entities existe
                if len(ent) == 3: # S'assurer que l'entité a bien 3 composantes
                    labels_dans_donnees.add(ent[2])
        
        for label in labels_dans_donnees:
            ner.add_label(label)
        log_callback(f"Labels pour NER (vérifiés/ajoutés) : {labels_dans_donnees if labels_dans_donnees else 'Aucun'}")

        log_callback(f"Début du fine-tuning pour {iterations} itérations (dropout={dropout_rate})...")
        
        # Désactiver les autres pipes pour l'entraînement du NER uniquement
        pipes_a_desactiver = [pipe_name for pipe_name in nlp.pipe_names if pipe_name != "ner"]
        
        with nlp.select_pipes(disable=pipes_a_desactiver):
            optimizer = nlp.begin_training()
            for iteration in range(iterations):
                random.shuffle(TRAIN_DATA)
                pertes = {}
                examples = [] # Construire les objets Example pour SpaCy 3+
                for texte, annotations in TRAIN_DATA:
                    try:
                        doc = nlp.make_doc(texte)
                        examples.append(Example.from_dict(doc, annotations))
                    except Exception as e_example:
                        log_callback(f"AVERTISSEMENT: Erreur création Example (it {iteration + 1}, texte: '{texte[:30]}...'): {e_example}")
                        continue
                
                if not examples:
                    log_callback(f"AVERTISSEMENT: Aucun exemple valide pour l'itération {iteration + 1}.")
                    continue

                try:
                    nlp.update(examples, sgd=optimizer, drop=dropout_rate, losses=pertes)
                except Exception as e_update:
                    log_callback(f"AVERTISSEMENT: Erreur pendant nlp.update (iteration {iteration + 1}): {e_update}")
                    # Il pourrait être judicieux de stopper l'entraînement ici si les erreurs sont trop nombreuses
                    continue 

                loss_value = pertes.get('ner', 0.0)
                log_callback(f"Itération {iteration + 1}/{iterations} - Perte NER : {loss_value:.4f}")
        
        if not os.path.exists(chemin_sauvegarde_modele):
            os.makedirs(chemin_sauvegarde_modele)
            log_callback(f"Dossier de sauvegarde créé : {chemin_sauvegarde_modele}")
            
        nlp.to_disk(chemin_sauvegarde_modele)
        log_callback(f"\nModèle fine-tuné sauvegardé avec succès dans : '{chemin_sauvegarde_modele}'")
        return True, chemin_sauvegarde_modele
        
    except Exception as e:
        log_callback(f"\nERREUR MAJEURE pendant le fine-tuning : {e}")
        log_callback(traceback.format_exc())
        return False, None