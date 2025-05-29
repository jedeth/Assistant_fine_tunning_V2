import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import spacy
from spacy.training.example import Example
import os
import json # Retiré 're' car generer_donnees_entrainement_interne est supprimé
import random

# --- Fonctions de chargement de données (pour le fine-tuning) ---
def charger_donnees_entrainement_json(chemin_fichier_json):
    try:
        with open(chemin_fichier_json, 'r', encoding='utf-8') as f:
            donnees = json.load(f)
        # S'assurer que le format est correct pour SpaCy
        donnees_formatees = []
        for item in donnees:
            if isinstance(item, list) and len(item) == 2: # Ancien format sauvegardé par le GUI
                 texte, annotations = item
            elif isinstance(item, tuple) and len(item) == 2: # Format correct
                 texte, annotations = item
            else:
                # Tenter de lire le format où les entités sont déjà des tuples (comme généré par preparer_donnees_multi_types.py)
                # Exemple: ("texte", {"entities": [(0, 4, "PER")]})
                # Cette partie suppose que le JSON externe est déjà bien formaté.
                # Si le JSON contient des listes au lieu de tuples pour les entités,
                # une conversion plus profonde serait nécessaire ici.
                # Pour l'instant, on assume que le JSON est compatible.
                texte, annotations = item[0], item[1]


            entites_formatees = []
            if "entities" in annotations and isinstance(annotations["entities"], list):
                for ent_item in annotations["entities"]:
                    if isinstance(ent_item, list) and len(ent_item) == 3: # [[0,4,"PER"]]
                        entites_formatees.append(tuple(ent_item))
                    elif isinstance(ent_item, tuple) and len(ent_item) == 3: # [(0,4,"PER")]
                        entites_formatees.append(ent_item)
                    else:
                        raise ValueError(f"Format d'entité incorrect dans les annotations: {ent_item}")
                donnees_formatees.append((texte, {"entities": entites_formatees}))
            elif "entities" not in annotations : # Cas sans entités (pourrait être utile pour des exemples négatifs globaux)
                donnees_formatees.append((texte, annotations))
            else: # Si "entities" est déjà au bon format (liste de tuples) ou structure inconnue
                donnees_formatees.append((texte, annotations))


        if not donnees_formatees:
             messagebox.showwarning("Données Vides", f"Aucune donnée d'entraînement valide trouvée dans '{chemin_fichier_json}'.")
             return None
        return donnees_formatees
    except FileNotFoundError:
        messagebox.showerror("Erreur Fichier JSON", f"Fichier de données d'entraînement '{chemin_fichier_json}' introuvable.")
        return None
    except Exception as e:
        messagebox.showerror("Erreur Chargement JSON", f"Erreur lors du chargement ou du formatage des données depuis '{chemin_fichier_json}': {e}")
        return None

# Variables globales
MODELES_SPACY_FR = {"Petit (sm)": "fr_core_news_sm", "Moyen (md)": "fr_core_news_md", "Grand (lg)": "fr_core_news_lg"}
modele_spacy_selectionne = None
chemin_output_donnees_spacy = None # Sera le chemin du fichier JSON combiné sélectionné par l'utilisateur

# --- Fonctions GUI ---
def valider_choix_modele():
    global modele_spacy_selectionne
    choix_utilisateur_label = choix_modele_var.get()
    modele_spacy_selectionne = MODELES_SPACY_FR.get(choix_utilisateur_label)
    if modele_spacy_selectionne:
        if verifier_et_telecharger_modele(modele_spacy_selectionne):
            messagebox.showinfo("Modèle Prêt", f"Modèle sélectionné : {modele_spacy_selectionne}\nVous pouvez maintenant sélectionner vos données d'entraînement.")
            activer_cadre_selection_donnees(True) # Modifié pour refléter le nouveau nom du cadre/fonction
            bouton_valider_modele.config(state="disabled")
            activer_cadre_fine_tuning(False)
        else:
            activer_cadre_selection_donnees(False)
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
        else:
            return False

def choisir_fichier_json_donnees():
    """Ouvre une boîte de dialogue pour choisir le fichier JSON de données d'entraînement."""
    global chemin_output_donnees_spacy
    chemin_fichier = filedialog.askopenfilename(
        title="Sélectionner le fichier de données d'entraînement JSON",
        filetypes=(("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*"))
    )
    if chemin_fichier:
        var_chemin_donnees_json.set(chemin_fichier)
        chemin_output_donnees_spacy = chemin_fichier # Stocker pour l'étape de fine-tuning
        label_statut_selection_donnees.config(text=f"Fichier sélectionné : {os.path.basename(chemin_fichier)}")

def valider_fichier_donnees():
    """Valide la sélection du fichier de données et active l'étape de fine-tuning."""
    if not chemin_output_donnees_spacy or not os.path.exists(chemin_output_donnees_spacy):
        messagebox.showerror("Erreur", "Veuillez sélectionner un fichier de données d'entraînement JSON valide.")
        label_statut_selection_donnees.config(text="Aucun fichier valide sélectionné.")
        return
    
    # Test simple de chargement pour vérifier si le fichier est un JSON grossièrement valide (optionnel)
    try:
        with open(chemin_output_donnees_spacy, 'r', encoding='utf-8') as f:
            json.load(f) # Tente de parser le JSON
        messagebox.showinfo("Données Prêtes", "Fichier de données d'entraînement validé.\nVous pouvez maintenant configurer le fine-tuning.")
        label_statut_selection_donnees.config(text="Fichier de données prêt.")
        bouton_valider_fichier_donnees.config(state="disabled")
        activer_cadre_fine_tuning(True)
    except json.JSONDecodeError:
        messagebox.showerror("Erreur JSON", "Le fichier sélectionné n'est pas un fichier JSON valide.")
        label_statut_selection_donnees.config(text="Erreur : Fichier JSON invalide.")
    except Exception as e:
        messagebox.showerror("Erreur Fichier", f"Impossible de lire le fichier : {e}")
        label_statut_selection_donnees.config(text="Erreur lecture fichier.")


def choisir_dossier_sauvegarde_modele():
    chemin_dossier = filedialog.askdirectory(title="Sélectionner le dossier pour sauvegarder le modèle fine-tuné")
    if chemin_dossier:
        var_chemin_sauvegarde_modele.set(chemin_dossier)

def lancer_fine_tuning_gui():
    global modele_spacy_selectionne, chemin_output_donnees_spacy

    if not modele_spacy_selectionne: # ... (logique inchangée)
        messagebox.showerror("Erreur", "Aucun modèle SpaCy de base n'a été sélectionné.")
        return
    if not chemin_output_donnees_spacy or not os.path.exists(chemin_output_donnees_spacy): # ... (logique inchangée)
        messagebox.showerror("Erreur", "Le fichier de données d'entraînement JSON n'a pas été sélectionné ou est introuvable.")
        return

    try: # ... (logique de récupération des paramètres inchangée)
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
    except tk.TclError: 
        messagebox.showerror("Erreur de Configuration", "Veuillez entrer des valeurs numériques valides pour les itérations et le dropout.")
        return
        
    TRAIN_DATA = charger_donnees_entrainement_json(chemin_output_donnees_spacy)
    if not TRAIN_DATA:
        log_fine_tuning("Échec du chargement des données d'entraînement. Vérifiez le fichier JSON.")
        return

    log_fine_tuning("Fine-tuning démarré...\n") # ... (logique de fine-tuning principale inchangée)
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
        
        # S'assurer que tous les labels présents dans les données sont connus du composant NER
        labels_dans_donnees = set()
        for _, annotations in TRAIN_DATA:
            for ent in annotations.get("entities", []):
                labels_dans_donnees.add(ent[2]) # ent[2] est le label (ex: "PER", "LOC")
        for label in labels_dans_donnees:
            ner.add_label(label)
        log_fine_tuning(f"Labels présents dans les données et ajoutés au NER : {labels_dans_donnees}")

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
                        log_fine_tuning(f"Erreur pendant nlp.update avec texte: '{texte[:30]}...' (ex. ignoré): {e_update}")
                        continue
                loss_value = pertes.get('ner', 0.0)
                log_fine_tuning(f"Itération {iteration + 1}/{iterations} - Perte NER : {loss_value:.4f}")
                fenetre.update_idletasks() 
        
        if not os.path.exists(chemin_sauvegarde):
            os.makedirs(chemin_sauvegarde)
        nlp.to_disk(chemin_sauvegarde)
        log_fine_tuning(f"\nModèle fine-tuné sauvegardé avec succès dans : '{chemin_sauvegarde}'")
        messagebox.showinfo("Fine-tuning Terminé", f"Le modèle a été fine-tuné et sauvegardé dans\n{chemin_sauvegarde}")

    except Exception as e:
        log_fine_tuning(f"\nErreur majeure pendant le fine-tuning : {e}")
        messagebox.showerror("Erreur Fine-tuning", f"Une erreur est survenue : {e}")
    finally:
        bouton_lancer_fine_tuning.config(state="normal")


def log_fine_tuning(message): # ... (inchangée)
    text_log_fine_tuning.config(state="normal")
    text_log_fine_tuning.insert(tk.END, message + "\n")
    text_log_fine_tuning.see(tk.END) 
    text_log_fine_tuning.config(state="disabled")
    fenetre.update_idletasks()

def activer_cadre_selection_donnees(activer): # Renommée pour clarté
    etat = "normal" if activer else "disabled"
    bouton_choisir_fichier_json.config(state=etat)
    entry_chemin_donnees_json.config(state="readonly" if activer else "disabled")
    bouton_valider_fichier_donnees.config(state=etat)
    if not activer:
        var_chemin_donnees_json.set("")
        label_statut_selection_donnees.config(text="")

def activer_cadre_fine_tuning(activer): # ... (inchangée)
    etat = "normal" if activer else "disabled"
    entry_iterations.config(state=etat)
    entry_dropout.config(state=etat)
    entry_chemin_sauvegarde_modele.config(state="readonly" if activer else "disabled")
    bouton_choisir_dossier_modele.config(state=etat)
    bouton_lancer_fine_tuning.config(state=etat)
    text_log_fine_tuning.config(state="normal" if activer else "disabled")
    if not activer:
        var_iterations.set(10) 
        var_dropout.set(0.3)  
        var_chemin_sauvegarde_modele.set("")
        text_log_fine_tuning.config(state="normal")
        text_log_fine_tuning.delete(1.0, tk.END) 
        text_log_fine_tuning.config(state="disabled")

# --- Création de la fenêtre principale ---
fenetre = tk.Tk()
fenetre.title("Assistant de Fine-tuning LLM SpaCy (Multi-types)")
fenetre.geometry("700x650") # Ajustement possible de la taille

# --- Cadre 1: Choix du Modèle ---
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

# --- Cadre 2: Sélection des Données d'Entraînement Combinées ---
cadre_selection_donnees = ttk.LabelFrame(fenetre, text="2. Sélection des Données d'Entraînement (JSON)", padding=(10, 10)) # Titre modifié
cadre_selection_donnees.pack(padx=10, pady=10, fill="x")

var_chemin_donnees_json = tk.StringVar() # Nouvelle variable pour le chemin du fichier JSON

frame_json_selection = ttk.Frame(cadre_selection_donnees)
frame_json_selection.pack(fill="x", pady=2)
ttk.Label(frame_json_selection, text="Fichier Données JSON:", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_chemin_donnees_json = ttk.Entry(frame_json_selection, textvariable=var_chemin_donnees_json, state="readonly", width=40)
entry_chemin_donnees_json.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_fichier_json = ttk.Button(frame_json_selection, text="Parcourir...", command=choisir_fichier_json_donnees)
bouton_choisir_fichier_json.pack(side=tk.LEFT)

bouton_valider_fichier_donnees = ttk.Button(cadre_selection_donnees, text="Valider Fichier de Données", command=valider_fichier_donnees)
bouton_valider_fichier_donnees.pack(pady=10)
label_statut_selection_donnees = ttk.Label(cadre_selection_donnees, text="")
label_statut_selection_donnees.pack(pady=2)

# --- Cadre 3: Fine-tuning du Modèle ---
cadre_fine_tuning = ttk.LabelFrame(fenetre, text="3. Fine-tuning du Modèle", padding=(10, 10))
cadre_fine_tuning.pack(padx=10, pady=10, fill="both", expand=True)
var_iterations = tk.IntVar(value=10) 
var_dropout = tk.DoubleVar(value=0.3) 
var_chemin_sauvegarde_modele = tk.StringVar()
# ... (widgets pour iterations, dropout, sauvegarde modèle, bouton lancer, log - inchangés) ...
frame_iter = ttk.Frame(cadre_fine_tuning)
frame_iter.pack(fill="x", pady=2)
ttk.Label(frame_iter, text="Nombre d'itérations:", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_iterations = ttk.Spinbox(frame_iter, from_=1, to=1000, textvariable=var_iterations, width=10)
entry_iterations.pack(side=tk.LEFT)

frame_drop = ttk.Frame(cadre_fine_tuning)
frame_drop.pack(fill="x", pady=2)
ttk.Label(frame_drop, text="Taux de Dropout (0.0-1.0):", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_dropout = ttk.Spinbox(frame_drop, from_=0.0, to=1.0, increment=0.05, textvariable=var_dropout, width=10, format="%.2f")
entry_dropout.pack(side=tk.LEFT)

frame_sauvegarde_modele = ttk.Frame(cadre_fine_tuning)
frame_sauvegarde_modele.pack(fill="x", pady=2)
ttk.Label(frame_sauvegarde_modele, text="Dossier de sauvegarde du modèle:", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_chemin_sauvegarde_modele = ttk.Entry(frame_sauvegarde_modele, textvariable=var_chemin_sauvegarde_modele, state="readonly", width=40)
entry_chemin_sauvegarde_modele.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_dossier_modele = ttk.Button(frame_sauvegarde_modele, text="Parcourir...", command=choisir_dossier_sauvegarde_modele)
bouton_choisir_dossier_modele.pack(side=tk.LEFT)

bouton_lancer_fine_tuning = ttk.Button(cadre_fine_tuning, text="Lancer le Fine-tuning", command=lancer_fine_tuning_gui)
bouton_lancer_fine_tuning.pack(pady=10)

ttk.Label(cadre_fine_tuning, text="Log du Fine-tuning:").pack(anchor="w", pady=(5,0))
text_log_fine_tuning = scrolledtext.ScrolledText(cadre_fine_tuning, height=10, width=80, state="disabled", wrap=tk.WORD)
text_log_fine_tuning.pack(pady=5, fill="both", expand=True)

# Initialisation des états des cadres
activer_cadre_selection_donnees(False)
activer_cadre_fine_tuning(False)

fenetre.mainloop()