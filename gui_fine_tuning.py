import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import spacy # Pour la vérification du modèle, spacy.load
import os
import re # Importé depuis preparer_donnees.py
import json # Pour sauvegarder les données d'entraînement

# --- Logique copiée et adaptée de preparer_donnees.py ---
def lire_noms(chemin_fichier_noms):
    noms = []
    try:
        with open(chemin_fichier_noms, 'r', encoding='utf-8') as f:
            for ligne in f:
                nom_propre = ligne.strip()
                if nom_propre:
                    noms.append(nom_propre)
    except FileNotFoundError:
        messagebox.showerror("Erreur Fichier", f"Le fichier annuaire '{chemin_fichier_noms}' est introuvable.")
        return None
    except Exception as e:
        messagebox.showerror("Erreur Lecture Annuaire", f"Erreur lors de la lecture de '{chemin_fichier_noms}': {e}")
        return None
    return noms

def lire_phrases_modeles(chemin_fichier_phrases):
    phrases = []
    try:
        with open(chemin_fichier_phrases, 'r', encoding='utf-8') as f:
            for ligne in f:
                phrase_modele = ligne.strip()
                if phrase_modele.startswith('"') and phrase_modele.endswith('",'):
                    phrase_modele = phrase_modele[1:-2]
                elif phrase_modele.startswith("'") and phrase_modele.endswith("',"):
                    phrase_modele = phrase_modele[1:-2]
                elif phrase_modele.startswith('"') and phrase_modele.endswith('"'):
                    phrase_modele = phrase_modele[1:-1]
                elif phrase_modele.startswith("'") and phrase_modele.endswith("'"):
                    phrase_modele = phrase_modele[1:-1]

                if phrase_modele and "{NOM}" in phrase_modele:
                    phrases.append(phrase_modele)
                elif phrase_modele:
                    print(f"Attention : La phrase modèle suivante ne contient pas {{NOM}} ou est vide après nettoyage : '{ligne.strip()}'") # Log console
    except FileNotFoundError:
        messagebox.showerror("Erreur Fichier", f"Le fichier de modèles de phrases '{chemin_fichier_phrases}' est introuvable.")
        return None
    except Exception as e:
        messagebox.showerror("Erreur Lecture Modèles Phrases", f"Erreur lors de la lecture de '{chemin_fichier_phrases}': {e}")
        return None
    return phrases

def generer_donnees_entrainement_interne(noms, phrases_modeles): # Renommée pour éviter conflit
    donnees_entrainement = []
    label_entite = "PER"
    if not noms or not phrases_modeles:
        return [] # Retourne une liste vide si les entrées sont invalides

    for nom in noms:
        for phrase_modele in phrases_modeles:
            phrase_formatee = phrase_modele.replace("{NOM}", nom)
            match = re.search(re.escape(nom), phrase_formatee)
            if match:
                debut, fin = match.span()
                entite = (debut, fin, label_entite)
                donnees_entrainement.append((phrase_formatee, {"entities": [entite]}))
            else:
                print(f"Attention : Impossible de trouver le nom '{nom}' dans la phrase générée '{phrase_formatee}'") # Log console
    return donnees_entrainement

# --- Fin de la logique de preparer_donnees.py ---

# Liste des modèles SpaCy français disponibles
MODELES_SPACY_FR = {
    "Petit (sm)": "fr_core_news_sm",
    "Moyen (md)": "fr_core_news_md",
    "Grand (lg)": "fr_core_news_lg"
}

modele_spacy_selectionne = None
chemin_fichier_annuaire = None
chemin_fichier_modeles_phrases = None
chemin_output_donnees_spacy = "donnees_entrainement_spacy_gui.json" # Nom de fichier par défaut

def valider_choix_modele():
    global modele_spacy_selectionne
    choix_utilisateur_label = choix_modele_var.get()
    modele_spacy_selectionne = MODELES_SPACY_FR.get(choix_utilisateur_label)
    
    if modele_spacy_selectionne:
        if verifier_et_telecharger_modele(modele_spacy_selectionne):
            messagebox.showinfo("Modèle Prêt", f"Modèle sélectionné : {modele_spacy_selectionne}\nVous pouvez maintenant préparer les données.")
            # Activer le cadre de préparation des données
            activer_cadre_preparation_donnees(True)
            bouton_valider_modele.config(state="disabled") # Désactiver le bouton de validation du modèle
        else:
            # Le téléchargement a échoué ou a été annulé, ne pas activer la suite
            activer_cadre_preparation_donnees(False)
    else:
        messagebox.showwarning("Aucun Modèle", "Veuillez sélectionner un modèle.")

def verifier_et_telecharger_modele(nom_modele):
    try:
        spacy.load(nom_modele)
        print(f"Le modèle '{nom_modele}' est déjà disponible.")
        return True
    except OSError:
        reponse = messagebox.askyesno("Modèle Non Trouvé", 
                                      f"Le modèle '{nom_modele}' n'est pas trouvé. Voulez-vous le télécharger ?")
        if reponse:
            print(f"Tentative de téléchargement du modèle '{nom_modele}'...")
            try:
                # Afficher un message d'attente
                label_statut_telechargement = ttk.Label(cadre_choix_modele, text=f"Téléchargement de {nom_modele} en cours...")
                label_statut_telechargement.pack(pady=5)
                fenetre.update_idletasks() # Mettre à jour l'interface pour afficher le message

                subprocess.check_call(['python', '-m', 'spacy', 'download', nom_modele], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                
                label_statut_telechargement.destroy() # Enlever le message d'attente
                messagebox.showinfo("Téléchargement Réussi", f"Le modèle '{nom_modele}' a été téléchargé avec succès.")
                print(f"Modèle '{nom_modele}' téléchargé avec succès.")
                return True
            except subprocess.CalledProcessError as e:
                label_statut_telechargement.destroy()
                stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Pas de détails d'erreur."
                messagebox.showerror("Erreur de Téléchargement", f"Impossible de télécharger '{nom_modele}'.\n{stderr_output}\nEssayez: python -m spacy download {nom_modele}")
                print(f"Erreur lors du téléchargement du modèle '{nom_modele}': {e}\n{stderr_output}")
                return False
            except FileNotFoundError:
                label_statut_telechargement.destroy()
                messagebox.showerror("Erreur Python", "La commande 'python' n'a pas été trouvée. Assurez-vous que Python est dans votre PATH.")
                return False
        else:
            messagebox.showinfo("Téléchargement Annulé", "Le téléchargement a été annulé. Vous ne pourrez pas continuer sans le modèle.")
            return False

def choisir_fichier_pour_variable(variable_chemin_tk, titre_dialogue):
    """Ouvre une boîte de dialogue pour choisir un fichier et met à jour la variable Tkinter."""
    chemin_fichier = filedialog.askopenfilename(title=titre_dialogue, filetypes=(("Fichiers Texte", "*.txt"), ("Tous les fichiers", "*.*")))
    if chemin_fichier:
        variable_chemin_tk.set(chemin_fichier)
        # Met à jour les variables globales aussi si besoin (alternative à l'utilisation directe de var.get())
        if variable_chemin_tk == var_chemin_annuaire:
            global chemin_fichier_annuaire
            chemin_fichier_annuaire = chemin_fichier
        elif variable_chemin_tk == var_chemin_modeles_phrases:
            global chemin_fichier_modeles_phrases
            chemin_fichier_modeles_phrases = chemin_fichier

def lancer_generation_donnees():
    """Lance la génération des données d'entraînement SpaCy."""
    global chemin_output_donnees_spacy # Utilise le nom de fichier par défaut ou un nom choisi par l'utilisateur plus tard
    
    # Récupérer les chemins depuis les variables Tkinter
    path_annuaire = var_chemin_annuaire.get()
    path_modeles = var_chemin_modeles_phrases.get()

    if not path_annuaire or not path_modeles:
        messagebox.showerror("Erreur", "Veuillez sélectionner les deux fichiers (annuaire et modèles de phrases).")
        return

    noms = lire_noms(path_annuaire)
    if noms is None: return # Erreur gérée dans lire_noms

    phrases_modeles = lire_phrases_modeles(path_modeles)
    if phrases_modeles is None: return # Erreur gérée dans lire_phrases_modeles

    if not noms or not phrases_modeles:
        messagebox.showwarning("Données Vides", "L'un des fichiers (ou les deux) n'a pas fourni de données valides (noms ou phrases modèles).")
        label_statut_generation.config(text="Échec : Données d'entrée vides ou invalides.")
        return

    label_statut_generation.config(text="Génération des données en cours...")
    fenetre.update_idletasks()

    donnees_spacy = generer_donnees_entrainement_interne(noms, phrases_modeles)

    if not donnees_spacy:
        messagebox.showwarning("Aucune Donnée Générée", "Aucune donnée d'entraînement n'a pu être générée. Vérifiez vos fichiers et les logs console.")
        label_statut_generation.config(text="Échec : Aucune donnée générée.")
        return

    try:
        # Demander où sauvegarder le fichier de données d'entraînement
        chemin_sauvegarde = filedialog.asksaveasfilename(
            title="Sauvegarder les données d'entraînement SpaCy",
            defaultextension=".json",
            initialfile="donnees_entrainement_spacy.json",
            filetypes=(("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*"))
        )
        if not chemin_sauvegarde: # L'utilisateur a annulé
            label_statut_generation.config(text="Sauvegarde annulée.")
            return

        chemin_output_donnees_spacy = chemin_sauvegarde # Mettre à jour le chemin global

        with open(chemin_output_donnees_spacy, "w", encoding="utf-8") as outfile:
            json.dump(donnees_spacy, outfile, ensure_ascii=False, indent=4)
        
        message = f"{len(donnees_spacy)} exemples d'entraînement générés et sauvegardés dans :\n{chemin_output_donnees_spacy}"
        messagebox.showinfo("Succès", message)
        label_statut_generation.config(text=f"Succès : {len(donnees_spacy)} exemples générés.")
        # Ici, on pourrait activer l'étape suivante (fine-tuning)
        bouton_generer_donnees.config(state="disabled")
    except Exception as e:
        messagebox.showerror("Erreur de Sauvegarde", f"Impossible de sauvegarder les données d'entraînement : {e}")
        label_statut_generation.config(text="Échec de la sauvegarde.")

def activer_cadre_preparation_donnees(activer):
    """Active ou désactive les widgets dans le cadre de préparation des données."""
    etat = "normal" if activer else "disabled"
    bouton_choisir_annuaire.config(state=etat)
    entry_chemin_annuaire.config(state="readonly" if activer else "disabled") # readonly pour afficher mais pas éditer
    bouton_choisir_modeles_phrases.config(state=etat)
    entry_chemin_modeles_phrases.config(state="readonly" if activer else "disabled")
    bouton_generer_donnees.config(state=etat)
    if not activer:
        var_chemin_annuaire.set("")
        var_chemin_modeles_phrases.set("")
        label_statut_generation.config(text="")


# --- Création de la fenêtre principale ---
fenetre = tk.Tk()
fenetre.title("Assistant de Fine-tuning LLM SpaCy")
fenetre.geometry("600x500") 

# --- Cadre pour le choix du modèle ---
cadre_choix_modele = ttk.LabelFrame(fenetre, text="1. Choix du Modèle SpaCy de Base", padding=(10, 10))
cadre_choix_modele.pack(padx=10, pady=10, fill="x")

label_instruction_modele = ttk.Label(cadre_choix_modele, text="Sélectionnez le modèle français à fine-tuner :")
label_instruction_modele.pack(pady=(0,10), anchor="w")
choix_modele_var = tk.StringVar(fenetre)
liste_labels_modeles = list(MODELES_SPACY_FR.keys())
menu_deroulant_modeles = ttk.Combobox(cadre_choix_modele, textvariable=choix_modele_var, values=liste_labels_modeles, state="readonly", width=30)
if liste_labels_modeles: menu_deroulant_modeles.current(1) # 'Moyen (md)' par défaut
menu_deroulant_modeles.pack(pady=5)
bouton_valider_modele = ttk.Button(cadre_choix_modele, text="Valider Modèle et Continuer", command=valider_choix_modele)
bouton_valider_modele.pack(pady=10)

# --- Cadre pour la préparation des données d'entraînement ---
cadre_preparation_donnees = ttk.LabelFrame(fenetre, text="2. Préparation des Données d'Entraînement", padding=(10, 10))
cadre_preparation_donnees.pack(padx=10, pady=10, fill="x")

# Variables Tkinter pour les chemins de fichiers
var_chemin_annuaire = tk.StringVar()
var_chemin_modeles_phrases = tk.StringVar()

# Fichier Annuaire
frame_annuaire = ttk.Frame(cadre_preparation_donnees)
frame_annuaire.pack(fill="x", pady=5)
label_annuaire = ttk.Label(frame_annuaire, text="Fichier Annuaire (.txt):", width=25)
label_annuaire.pack(side=tk.LEFT, padx=(0,5))
entry_chemin_annuaire = ttk.Entry(frame_annuaire, textvariable=var_chemin_annuaire, state="readonly", width=40)
entry_chemin_annuaire.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_annuaire = ttk.Button(frame_annuaire, text="Parcourir...", command=lambda: choisir_fichier_pour_variable(var_chemin_annuaire, "Sélectionner l'annuaire des noms"))
bouton_choisir_annuaire.pack(side=tk.LEFT)

# Fichier Modèles de Phrases
frame_modeles = ttk.Frame(cadre_preparation_donnees)
frame_modeles.pack(fill="x", pady=5)
label_modeles_phrases = ttk.Label(frame_modeles, text="Fichier Modèles Phrases (.txt):", width=25)
label_modeles_phrases.pack(side=tk.LEFT, padx=(0,5))
entry_chemin_modeles_phrases = ttk.Entry(frame_modeles, textvariable=var_chemin_modeles_phrases, state="readonly", width=40)
entry_chemin_modeles_phrases.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_modeles_phrases = ttk.Button(frame_modeles, text="Parcourir...", command=lambda: choisir_fichier_pour_variable(var_chemin_modeles_phrases, "Sélectionner les modèles de phrases"))
bouton_choisir_modeles_phrases.pack(side=tk.LEFT)

# Bouton pour lancer la génération
bouton_generer_donnees = ttk.Button(cadre_preparation_donnees, text="Générer les Données d'Entraînement", command=lancer_generation_donnees)
bouton_generer_donnees.pack(pady=15)

# Label de statut pour la génération des données
label_statut_generation = ttk.Label(cadre_preparation_donnees, text="")
label_statut_generation.pack(pady=5)

# Initialement, désactiver le cadre de préparation des données
activer_cadre_preparation_donnees(False)

fenetre.mainloop()