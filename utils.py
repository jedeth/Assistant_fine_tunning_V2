# utils.py

import traceback
import spacy
import subprocess
import os # Seulement si vous avez besoin de manipuler des chemins ici, sinon spacy.load suffit.

def verifier_existence_modele_spacy(nom_modele):
    """
    Vérifie si un modèle SpaCy est déjà installé et chargeable.
    Retourne:
        {"status": "existe", "model_name": nom_modele} si le modèle existe.
        {"status": "non_trouve", "model_name": nom_modele} si le modèle n'est pas trouvé (OSError).
        {"status": "erreur_verification", "model_name": nom_modele, "error": str(e)} pour d'autres erreurs.
    """
    try:
        spacy.load(nom_modele) # Essaye de charger le modèle
        return {"status": "existe", "model_name": nom_modele}
    except OSError:
        # OSError est typiquement levée si le modèle n'est pas trouvé/installé
        return {"status": "non_trouve", "model_name": nom_modele}
    except Exception as e:
        # Capturer d'autres exceptions potentielles lors du spacy.load
        return {"status": "erreur_verification", "model_name": nom_modele, "error": str(e)}

def telecharger_modele_spacy(nom_modele, log_callback=None):
    """
    Tente de télécharger un modèle SpaCy en utilisant subprocess.
    Le log_callback peut être utilisé pour afficher la progression dans une GUI.
    Retourne:
        {"status": "telechargement_reussi", "model_name": nom_modele}
        {"status": "telechargement_echec", "model_name": nom_modele, "error": str(e)}
        {"status": "python_non_trouve", "model_name": nom_modele} si python n'est pas dans le PATH
    """
    if log_callback:
        log_callback(f"Tentative de téléchargement du modèle '{nom_modele}'...")
    
    try:
        # Utilise subprocess pour appeler la commande de téléchargement de SpaCy
        # stderr=subprocess.PIPE pour capturer les erreurs de la commande spacy download
        process = subprocess.run(
            ['python', '-m', 'spacy', 'download', nom_modele], 
            stdout=subprocess.PIPE,  # Capturer la sortie standard
            stderr=subprocess.PIPE,  # Capturer la sortie d'erreur
            text=True,               # Décoder stdout/stderr en texte
            check=False              # Ne pas lever d'exception pour les codes de retour non nuls
        )
        
        if process.returncode == 0:
            if log_callback: log_callback(f"Modèle '{nom_modele}' téléchargé avec succès.")
            return {"status": "telechargement_reussi", "model_name": nom_modele}
        else:
            error_message = process.stderr if process.stderr else "Erreur inconnue lors du téléchargement."
            if log_callback: log_callback(f"ERREUR lors du téléchargement du modèle '{nom_modele}': {error_message.strip()}")
            return {"status": "telechargement_echec", "model_name": nom_modele, "error": error_message.strip()}
            
    except FileNotFoundError: # Si la commande 'python' elle-même n'est pas trouvée
        if log_callback: log_callback("ERREUR: La commande 'python' n'a pas été trouvée. Assurez-vous que Python est dans votre PATH.")
        return {"status": "python_non_trouve", "model_name": nom_modele}
    except Exception as e: # Autres erreurs potentielles de subprocess
        if log_callback: 
            log_callback(f"ERREUR subprocess inattendue lors du téléchargement de '{nom_modele}': {e}")
            log_callback(traceback.format_exc())
        return {"status": "telechargement_echec", "model_name": nom_modele, "error": str(e)}