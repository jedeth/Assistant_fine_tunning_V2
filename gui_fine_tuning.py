import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext # scrolledtext pour la sortie du fine-tuning
import subprocess
import spacy # Pour spacy.load, spacy.training.Example
from spacy.training.example import Example # Nécessaire pour SpaCy v3.x
import os
import re
import json
import random # Pour mélanger les données d'entraînement lors du fine-tuning

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
                    print(f"Attention : La phrase modèle suivante ne contient pas {{NOM}} ou est vide après nettoyage : '{ligne.strip()}'")
    except FileNotFoundError:
        messagebox.showerror("Erreur Fichier", f"Le fichier de modèles de phrases '{chemin_fichier_phrases}' est introuvable.")
        return None
    except Exception as e:
        messagebox.showerror("Erreur Lecture Modèles Phrases", f"Erreur lors de la lecture de '{chemin_fichier_phrases}': {e}")
        return None
    return phrases

def generer_donnees_entrainement_interne(noms, phrases_modeles):
    donnees_entrainement = []
    label_entite = "PER"
    if not noms or not phrases_modeles: return []
    for nom in noms:
        for phrase_modele in phrases_modeles:
            phrase_formatee = phrase_modele.replace("{NOM}", nom)
            match = re.search(re.escape(nom), phrase_formatee)
            if match:
                debut, fin = match.span()
                entite = (debut, fin, label_entite)
                donnees_entrainement.append((phrase_formatee, {"entities": [entite]}))
            else:
                print(f"Attention : Impossible de trouver le nom '{nom}' dans la phrase générée '{phrase_formatee}'")
    return donnees_entrainement
# --- Fin de la logique de preparer_donnees.py ---

# --- Logique adaptée de fine_tuner_spacy.py ---
def charger_donnees_entrainement_json(chemin_fichier_json):
    try:
        with open(chemin_fichier_json, 'r', encoding='utf-8') as f:
            donnees = json.load(f)
        donnees_formatees = []
        for texte, annotations in donnees:
            entites_formatees = []
            if "entities" in annotations:
                for debut, fin, label in annotations["entities"]:
                    entites_formatees.append((debut, fin, label))
                donnees_formatees.append((texte, {"entities": entites_formatees}))
            else:
                donnees_formatees.append((texte, annotations))
        return donnees_formatees
    except FileNotFoundError:
        messagebox.showerror("Erreur Fichier JSON", f"Fichier de données d'entraînement '{chemin_fichier_json}' introuvable.")
        return None
    except Exception as e:
        messagebox.showerror("Erreur Chargement JSON", f"Erreur lors du chargement des données depuis '{chemin_fichier_json}': {e}")
        return None
# --- Fin de la logique de fine_tuner_spacy.py ---


# Variables globales
MODELES_SPACY_FR = {"Petit (sm)": "fr_core_news_sm", "Moyen (md)": "fr_core_news_md", "Grand (lg)": "fr_core_news_lg"}
modele_spacy_selectionne = None # Nom du modèle de base (ex: "fr_core_news_md")
chemin_output_donnees_spacy = None # Chemin vers le fichier JSON de données d'entraînement généré

# --- Fonctions GUI ---
def valider_choix_modele():
    global modele_spacy_selectionne
    choix_utilisateur_label = choix_modele_var.get()
    modele_spacy_selectionne = MODELES_SPACY_FR.get(choix_utilisateur_label)
    if modele_spacy_selectionne:
        if verifier_et_telecharger_modele(modele_spacy_selectionne):
            messagebox.showinfo("Modèle Prêt", f"Modèle sélectionné : {modele_spacy_selectionne}\nVous pouvez maintenant préparer les données.")
            activer_cadre_preparation_donnees(True)
            bouton_valider_modele.config(state="disabled")
            activer_cadre_fine_tuning(False) # S'assurer que le fine-tuning est désactivé
        else:
            activer_cadre_preparation_donnees(False)
    else:
        messagebox.showwarning("Aucun Modèle", "Veuillez sélectionner un modèle.")

def verifier_et_telecharger_modele(nom_modele):
    # ... (fonction inchangée de la réponse précédente)
    try:
        spacy.load(nom_modele)
        print(f"Le modèle '{nom_modele}' est déjà disponible.")
        return True
    except OSError:
        reponse = messagebox.askyesno("Modèle Non Trouvé", 
                                      f"Le modèle '{nom_modele}' n'est pas trouvé. Voulez-vous le télécharger ?")
        if reponse:
            print(f"Tentative de téléchargement du modèle '{nom_modele}'...")
            # (gestion de l'affichage du statut de téléchargement)
            status_label_dl = ttk.Label(cadre_choix_modele, text=f"Téléchargement de {nom_modele}...")
            status_label_dl.pack(pady=2)
            fenetre.update_idletasks()
            try:
                subprocess.check_call(['python', '-m', 'spacy', 'download', nom_modele], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                status_label_dl.destroy()
                messagebox.showinfo("Téléchargement Réussi", f"Le modèle '{nom_modele}' a été téléchargé avec succès.")
                return True
            except subprocess.CalledProcessError as e:
                status_label_dl.destroy()
                stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Pas de détails d'erreur."
                messagebox.showerror("Erreur de Téléchargement", f"Impossible de télécharger '{nom_modele}'.\n{stderr_output}\nEssayez: python -m spacy download {nom_modele}")
                return False
            except FileNotFoundError:
                status_label_dl.destroy()
                messagebox.showerror("Erreur Python", "La commande 'python' n'a pas été trouvée.")
                return False
        else: # L'utilisateur a refusé le téléchargement
            return False


def choisir_fichier_pour_variable(variable_chemin_tk, titre_dialogue):
    # ... (fonction inchangée)
    chemin_fichier = filedialog.askopenfilename(title=titre_dialogue, filetypes=(("Fichiers Texte", "*.txt"), ("Tous les fichiers", "*.*")))
    if chemin_fichier:
        variable_chemin_tk.set(chemin_fichier)

def lancer_generation_donnees():
    global chemin_output_donnees_spacy
    path_annuaire = var_chemin_annuaire.get()
    path_modeles = var_chemin_modeles_phrases.get()
    if not path_annuaire or not path_modeles:
        messagebox.showerror("Erreur", "Veuillez sélectionner les deux fichiers (annuaire et modèles de phrases).")
        return

    noms = lire_noms(path_annuaire)
    if noms is None: return
    phrases_modeles = lire_phrases_modeles(path_modeles)
    if phrases_modeles is None: return
    if not noms or not phrases_modeles:
        messagebox.showwarning("Données Vides", "L'un des fichiers n'a pas fourni de données valides.")
        label_statut_generation.config(text="Échec : Données d'entrée vides ou invalides.")
        return

    label_statut_generation.config(text="Génération des données en cours...")
    fenetre.update_idletasks()
    donnees_spacy = generer_donnees_entrainement_interne(noms, phrases_modeles)
    if not donnees_spacy:
        messagebox.showwarning("Aucune Donnée Générée", "Aucune donnée d'entraînement n'a pu être générée.")
        label_statut_generation.config(text="Échec : Aucune donnée générée.")
        return
    
    chemin_sauvegarde = filedialog.asksaveasfilename(title="Sauvegarder les données d'entraînement SpaCy", defaultextension=".json", initialfile="donnees_entrainement_spacy.json", filetypes=(("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")))
    if not chemin_sauvegarde:
        label_statut_generation.config(text="Sauvegarde annulée.")
        return
    chemin_output_donnees_spacy = chemin_sauvegarde
    try:
        with open(chemin_output_donnees_spacy, "w", encoding="utf-8") as outfile:
            json.dump(donnees_spacy, outfile, ensure_ascii=False, indent=4)
        message = f"{len(donnees_spacy)} exemples générés et sauvegardés dans :\n{chemin_output_donnees_spacy}"
        messagebox.showinfo("Succès", message)
        label_statut_generation.config(text=f"Succès : {len(donnees_spacy)} exemples générés.")
        bouton_generer_donnees.config(state="disabled")
        activer_cadre_fine_tuning(True) # Activer l'étape suivante
    except Exception as e:
        messagebox.showerror("Erreur de Sauvegarde", f"Impossible de sauvegarder les données : {e}")
        label_statut_generation.config(text="Échec de la sauvegarde.")

def choisir_dossier_sauvegarde_modele():
    chemin_dossier = filedialog.askdirectory(title="Sélectionner le dossier pour sauvegarder le modèle fine-tuné")
    if chemin_dossier:
        var_chemin_sauvegarde_modele.set(chemin_dossier)

def lancer_fine_tuning_gui():
    global modele_spacy_selectionne, chemin_output_donnees_spacy

    if not modele_spacy_selectionne:
        messagebox.showerror("Erreur", "Aucun modèle SpaCy de base n'a été sélectionné.")
        return
    if not chemin_output_donnees_spacy or not os.path.exists(chemin_output_donnees_spacy):
        messagebox.showerror("Erreur", "Le fichier de données d'entraînement JSON n'a pas été généré ou est introuvable.")
        return

    try:
        iterations = var_iterations.get()
        dropout = var_dropout.get()
        chemin_sauvegarde = var_chemin_sauvegarde_modele.get()
        if iterations <= 0:
            messagebox.showerror("Erreur de Configuration", "Le nombre d'itérations doit être supérieur à 0.")
            return
        if not (0.0 <= dropout <= 1.0):
            messagebox.showerror("Erreur de Configuration", "Le taux de dropout doit être entre 0.0 et 1.0.")
            return
        if not chemin_sauvegarde:
            messagebox.showerror("Erreur de Configuration", "Veuillez spécifier un dossier pour sauvegarder le modèle fine-tuné.")
            return
    except tk.TclError: # Erreur si les champs ne sont pas des nombres valides
        messagebox.showerror("Erreur de Configuration", "Veuillez entrer des valeurs numériques valides pour les itérations et le dropout.")
        return
        
    TRAIN_DATA = charger_donnees_entrainement_json(chemin_output_donnees_spacy)
    if not TRAIN_DATA:
        return # Erreur déjà affichée par la fonction de chargement

    log_fine_tuning("Fine-tuning démarré...\n")
    log_fine_tuning(f"Modèle de base: {modele_spacy_selectionne}")
    log_fine_tuning(f"Données: {chemin_output_donnees_spacy} ({len(TRAIN_DATA)} exemples)")
    log_fine_tuning(f"Itérations: {iterations}, Dropout: {dropout}")
    log_fine_tuning(f"Sauvegarde vers: {chemin_sauvegarde}\n")
    
    bouton_lancer_fine_tuning.config(state="disabled")
    fenetre.update_idletasks()

    try:
        nlp = spacy.load(modele_spacy_selectionne)
        log_fine_tuning(f"Modèle de base '{modele_spacy_selectionne}' chargé.")
        
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
        
        for _, annotations in TRAIN_DATA:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])
        
        pipes_a_desactiver = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.select_pipes(disable=pipes_a_desactiver):
            optimizer = nlp.begin_training()
            for iteration in range(iterations):
                random.shuffle(TRAIN_DATA)
                pertes = {}
                for texte, annotations in TRAIN_DATA:
                    try:
                        doc = nlp.make_doc(texte)
                        example = Example.from_dict(doc, annotations)
                        nlp.update([example], sgd=optimizer, drop=dropout, losses=pertes)
                    except Exception as e_update:
                        log_fine_tuning(f"Erreur pendant nlp.update (ex. ignoré): {e_update}")
                        continue

                loss_value = pertes.get('ner', 0.0)
                log_fine_tuning(f"Itération {iteration + 1}/{iterations} - Perte NER : {loss_value:.4f}")
                fenetre.update_idletasks() # Mettre à jour l'UI pour voir le log
        
        if not os.path.exists(chemin_sauvegarde):
            os.makedirs(chemin_sauvegarde) # Créer le dossier s'il n'existe pas
            
        nlp.to_disk(chemin_sauvegarde)
        log_fine_tuning(f"\nModèle fine-tuné sauvegardé avec succès dans : '{chemin_sauvegarde}'")
        messagebox.showinfo("Fine-tuning Terminé", f"Le modèle a été fine-tuné et sauvegardé dans\n{chemin_sauvegarde}")

    except Exception as e:
        log_fine_tuning(f"\nErreur majeure pendant le fine-tuning : {e}")
        messagebox.showerror("Erreur Fine-tuning", f"Une erreur est survenue : {e}")
    finally:
        bouton_lancer_fine_tuning.config(state="normal")


def log_fine_tuning(message):
    """Ajoute un message à la zone de log du fine-tuning."""
    text_log_fine_tuning.config(state="normal")
    text_log_fine_tuning.insert(tk.END, message + "\n")
    text_log_fine_tuning.see(tk.END) # Scroll vers le bas
    text_log_fine_tuning.config(state="disabled")
    fenetre.update_idletasks()


def activer_cadre_preparation_donnees(activer):
    # ... (fonction inchangée)
    etat = "normal" if activer else "disabled"
    bouton_choisir_annuaire.config(state=etat)
    entry_chemin_annuaire.config(state="readonly" if activer else "disabled")
    bouton_choisir_modeles_phrases.config(state=etat)
    entry_chemin_modeles_phrases.config(state="readonly" if activer else "disabled")
    bouton_generer_donnees.config(state=etat)
    if not activer:
        var_chemin_annuaire.set("")
        var_chemin_modeles_phrases.set("")
        label_statut_generation.config(text="")

def activer_cadre_fine_tuning(activer):
    """Active ou désactive les widgets dans le cadre de fine-tuning."""
    etat = "normal" if activer else "disabled"
    # Widgets du cadre fine-tuning à (dés)activer
    entry_iterations.config(state=etat)
    entry_dropout.config(state=etat)
    entry_chemin_sauvegarde_modele.config(state="readonly" if activer else "disabled")
    bouton_choisir_dossier_modele.config(state=etat)
    bouton_lancer_fine_tuning.config(state=etat)
    text_log_fine_tuning.config(state="normal" if activer else "disabled")
    if not activer:
        var_iterations.set(10) # Valeur par défaut
        var_dropout.set(0.3)  # Valeur par défaut
        var_chemin_sauvegarde_modele.set("")
        text_log_fine_tuning.config(state="normal")
        text_log_fine_tuning.delete(1.0, tk.END) # Effacer le log
        text_log_fine_tuning.config(state="disabled")


# --- Création de la fenêtre principale ---
fenetre = tk.Tk()
fenetre.title("Assistant de Fine-tuning LLM SpaCy")
fenetre.geometry("700x750") # Agrandir un peu pour la nouvelle section

# --- Cadre 1: Choix du Modèle ---
# ... (inchangé)
cadre_choix_modele = ttk.LabelFrame(fenetre, text="1. Choix du Modèle SpaCy de Base", padding=(10, 10))
cadre_choix_modele.pack(padx=10, pady=10, fill="x")
label_instruction_modele = ttk.Label(cadre_choix_modele, text="Sélectionnez le modèle français à fine-tuner :")
label_instruction_modele.pack(pady=(0,10), anchor="w")
choix_modele_var = tk.StringVar(fenetre)
liste_labels_modeles = list(MODELES_SPACY_FR.keys())
menu_deroulant_modeles = ttk.Combobox(cadre_choix_modele, textvariable=choix_modele_var, values=liste_labels_modeles, state="readonly", width=30)
if liste_labels_modeles: menu_deroulant_modeles.current(1) 
menu_deroulant_modeles.pack(pady=5)
bouton_valider_modele = ttk.Button(cadre_choix_modele, text="Valider Modèle et Continuer", command=valider_choix_modele)
bouton_valider_modele.pack(pady=10)

# --- Cadre 2: Préparation des Données ---
# ... (inchangé en termes de structure, mais l'appel à activer_cadre_fine_tuning a été ajouté à lancer_generation_donnees)
cadre_preparation_donnees = ttk.LabelFrame(fenetre, text="2. Préparation des Données d'Entraînement", padding=(10, 10))
cadre_preparation_donnees.pack(padx=10, pady=10, fill="x")
var_chemin_annuaire = tk.StringVar()
var_chemin_modeles_phrases = tk.StringVar()
# ... (widgets pour annuaire et modeles phrases comme avant) ...
frame_annuaire = ttk.Frame(cadre_preparation_donnees)
frame_annuaire.pack(fill="x", pady=2)
ttk.Label(frame_annuaire, text="Fichier Annuaire (.txt):", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_chemin_annuaire = ttk.Entry(frame_annuaire, textvariable=var_chemin_annuaire, state="readonly", width=40)
entry_chemin_annuaire.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_annuaire = ttk.Button(frame_annuaire, text="Parcourir...", command=lambda: choisir_fichier_pour_variable(var_chemin_annuaire, "Sélectionner l'annuaire"))
bouton_choisir_annuaire.pack(side=tk.LEFT)

frame_modeles = ttk.Frame(cadre_preparation_donnees)
frame_modeles.pack(fill="x", pady=2)
ttk.Label(frame_modeles, text="Fichier Modèles Phrases (.txt):", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_chemin_modeles_phrases = ttk.Entry(frame_modeles, textvariable=var_chemin_modeles_phrases, state="readonly", width=40)
entry_chemin_modeles_phrases.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_modeles_phrases = ttk.Button(frame_modeles, text="Parcourir...", command=lambda: choisir_fichier_pour_variable(var_chemin_modeles_phrases, "Sélectionner les modèles de phrases"))
bouton_choisir_modeles_phrases.pack(side=tk.LEFT)

bouton_generer_donnees = ttk.Button(cadre_preparation_donnees, text="Générer les Données d'Entraînement", command=lancer_generation_donnees)
bouton_generer_donnees.pack(pady=10)
label_statut_generation = ttk.Label(cadre_preparation_donnees, text="")
label_statut_generation.pack(pady=2)


# --- Cadre 3: Fine-tuning du Modèle ---
cadre_fine_tuning = ttk.LabelFrame(fenetre, text="3. Fine-tuning du Modèle", padding=(10, 10))
cadre_fine_tuning.pack(padx=10, pady=10, fill="both", expand=True)

# Variables Tkinter pour les paramètres de fine-tuning
var_iterations = tk.IntVar(value=10) # Valeur par défaut
var_dropout = tk.DoubleVar(value=0.3) # Valeur par défaut
var_chemin_sauvegarde_modele = tk.StringVar()

# Itérations
frame_iter = ttk.Frame(cadre_fine_tuning)
frame_iter.pack(fill="x", pady=2)
ttk.Label(frame_iter, text="Nombre d'itérations:", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_iterations = ttk.Spinbox(frame_iter, from_=1, to=1000, textvariable=var_iterations, width=10) # Spinbox pour les itérations
entry_iterations.pack(side=tk.LEFT)

# Dropout
frame_drop = ttk.Frame(cadre_fine_tuning)
frame_drop.pack(fill="x", pady=2)
ttk.Label(frame_drop, text="Taux de Dropout (0.0-1.0):", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_dropout = ttk.Spinbox(frame_drop, from_=0.0, to=1.0, increment=0.05, textvariable=var_dropout, width=10, format="%.2f") # Spinbox pour le dropout
entry_dropout.pack(side=tk.LEFT)

# Chemin de sauvegarde du modèle fine-tuné
frame_sauvegarde_modele = ttk.Frame(cadre_fine_tuning)
frame_sauvegarde_modele.pack(fill="x", pady=2)
ttk.Label(frame_sauvegarde_modele, text="Dossier de sauvegarde du modèle:", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_chemin_sauvegarde_modele = ttk.Entry(frame_sauvegarde_modele, textvariable=var_chemin_sauvegarde_modele, state="readonly", width=40)
entry_chemin_sauvegarde_modele.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_dossier_modele = ttk.Button(frame_sauvegarde_modele, text="Parcourir...", command=choisir_dossier_sauvegarde_modele)
bouton_choisir_dossier_modele.pack(side=tk.LEFT)

# Bouton pour lancer le fine-tuning
bouton_lancer_fine_tuning = ttk.Button(cadre_fine_tuning, text="Lancer le Fine-tuning", command=lancer_fine_tuning_gui)
bouton_lancer_fine_tuning.pack(pady=10)

# Zone de log pour le fine-tuning
ttk.Label(cadre_fine_tuning, text="Log du Fine-tuning:").pack(anchor="w", pady=(5,0))
text_log_fine_tuning = scrolledtext.ScrolledText(cadre_fine_tuning, height=10, width=80, state="disabled", wrap=tk.WORD)
text_log_fine_tuning.pack(pady=5, fill="both", expand=True)


# Initialisation des états des cadres
activer_cadre_preparation_donnees(False)
activer_cadre_fine_tuning(False)

fenetre.mainloop()