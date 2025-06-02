# logique_preparation_donnees.py

import re
import json
import os

def lire_entites_depuis_fichier(chemin_fichier, log_callback=None):
    """
    Lit une liste d'entités (une par ligne) depuis un fichier.
    Retourne une liste d'entités, ou None en cas d'erreur.
    """
    entites = []
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            for ligne in f:
                entite_texte = ligne.strip()
                if entite_texte:
                    entites.append(entite_texte)
        if not entites:
            if log_callback: log_callback(f"Attention : Aucun contenu trouvé dans le fichier d'entités : {chemin_fichier}")
        return entites
    except FileNotFoundError:
        msg = f"ERREUR : Le fichier d'entités '{chemin_fichier}' est introuvable."
        if log_callback: log_callback(msg)
        return None
    except Exception as e:
        msg = f"ERREUR : Erreur lors de la lecture du fichier d'entités '{chemin_fichier}': {e}"
        if log_callback: log_callback(msg)
        return None

def lire_phrases_modeles_specifiques(chemin_fichier_phrases, placeholder_attendu, log_callback=None):
    """
    Lit les phrases modèles depuis un fichier.
    Vérifie la présence d'un placeholder spécifique.
    Retourne une liste de phrases, ou None en cas d'erreur.
    """
    phrases = []
    try:
        with open(chemin_fichier_phrases, 'r', encoding='utf-8') as f:
            for ligne_num, ligne in enumerate(f, 1):
                phrase_modele = ligne.strip()
                if phrase_modele.startswith('"') and phrase_modele.endswith('",'): phrase_modele = phrase_modele[1:-2]
                elif phrase_modele.startswith("'") and phrase_modele.endswith("',"): phrase_modele = phrase_modele[1:-2]
                elif phrase_modele.startswith('"') and phrase_modele.endswith('"'): phrase_modele = phrase_modele[1:-1]
                elif phrase_modele.startswith("'") and phrase_modele.endswith("'"): phrase_modele = phrase_modele[1:-1]
                
                if phrase_modele and placeholder_attendu in phrase_modele:
                    phrases.append(phrase_modele)
                elif phrase_modele and log_callback: 
                    log_callback(f"Ligne {ligne_num} ({os.path.basename(chemin_fichier_phrases)}): Placeholder '{placeholder_attendu}' manquant ou phrase vide.")
        if not phrases:
            if log_callback: log_callback(f"Attention : Aucun modèle de phrase valide pour '{placeholder_attendu}' dans : {os.path.basename(chemin_fichier_phrases)}")
        return phrases
    except FileNotFoundError:
        msg = f"ERREUR : Fichier modèles phrases '{chemin_fichier_phrases}' introuvable."
        if log_callback: log_callback(msg)
        return None
    except Exception as e:
        msg = f"ERREUR lecture phrases '{chemin_fichier_phrases}': {e}"
        if log_callback: log_callback(msg)
        return None

def generer_donnees_pour_type(liste_entites, liste_phrases_modeles, label_entite, placeholder, log_callback=None):
    """
    Génère les données d'entraînement au format SpaCy pour un type d'entité spécifique.
    """
    donnees_entrainement = []
    if not liste_entites or not liste_phrases_modeles:
        if log_callback: log_callback(f"Données sources vides pour label '{label_entite}'. Impossible de générer.")
        return donnees_entrainement # Retourne une liste vide

    for entite_texte in liste_entites:
        for phrase_modele in liste_phrases_modeles:
            phrase_formatee = phrase_modele.replace(placeholder, entite_texte)
            match = re.search(re.escape(entite_texte), phrase_formatee)
            if match:
                debut, fin = match.span()
                entite_spacy = (debut, fin, label_entite)
                donnees_entrainement.append((phrase_formatee, {"entities": [entite_spacy]}))
            elif log_callback:
                log_callback(f"Attention : Entité '{entite_texte}' ({label_entite}) non trouvée dans la phrase générée avec placeholder '{placeholder}'.")
    return donnees_entrainement

def sauvegarder_donnees_json(donnees, chemin_fichier_sortie, log_callback=None):
    """
    Sauvegarde les données générées dans un fichier JSON.
    Retourne True en cas de succès, False sinon.
    """
    if not donnees:
        if log_callback: log_callback(f"Aucune donnée à sauvegarder pour '{chemin_fichier_sortie}'.")
        return False
    try:
        with open(chemin_fichier_sortie, "w", encoding="utf-8") as outfile:
            json.dump(donnees, outfile, ensure_ascii=False, indent=4)
        if log_callback: log_callback(f"{len(donnees)} exemples sauvegardés dans '{os.path.basename(chemin_fichier_sortie)}'")
        return True
    except Exception as e:
        if log_callback: log_callback(f"ERREUR sauvegarde dans '{chemin_fichier_sortie}': {e}")
        return False