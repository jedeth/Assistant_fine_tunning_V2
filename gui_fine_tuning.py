import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import spacy
from spacy.training.example import Example
import os
import json
import random

# --- Logique de Pseudonymisation (adaptée de pseudonymiser_texte.py) ---
def pseudonymiser_texte_pour_gui(nlp_model, texte_original):
    """
    Pseudonymise le texte en utilisant le modèle SpaCy chargé.
    Retourne le texte pseudonymisé et la table de correspondance.
    """
    if not nlp_model:
        messagebox.showerror("Erreur Modèle", "Le modèle SpaCy n'est pas chargé pour la pseudonymisation.")
        return None, None

    doc = nlp_model(texte_original)
    
    correspondances = {}
    pseudonyme_compteur = 1
    
    entites_a_remplacer = []

    for entite in doc.ents:
        # On se concentre sur PER pour la pseudonymisation, mais le modèle peut en détecter d'autres.
        if entite.label_ == "PER": 
            nom_original = entite.text
            
            if nom_original not in correspondances:
                pseudonyme = f"[PERSONNE_{pseudonyme_compteur}]"
                correspondances[nom_original] = pseudonyme
                pseudonyme_compteur += 1
            else:
                pseudonyme = correspondances[nom_original]
            
            entites_a_remplacer.append((entite.start_char, entite.end_char, pseudonyme))

    # Trier les entités par position de début en ordre inverse pour éviter les problèmes d'indices
    entites_a_remplacer.sort(key=lambda x: x[0], reverse=True)
    
    nouveau_texte_parts = []
    dernier_index_traite = len(texte_original)

    for start_char, end_char, pseudonyme in entites_a_remplacer:
        if end_char < dernier_index_traite:
            nouveau_texte_parts.append(texte_original[end_char:dernier_index_traite])
        nouveau_texte_parts.append(pseudonyme)
        dernier_index_traite = start_char
    
    nouveau_texte_parts.append(texte_original[0:dernier_index_traite])
    texte_pseudonymise_final = "".join(reversed(nouveau_texte_parts))
            
    return texte_pseudonymise_final, correspondances

# --- Fonctions de chargement de données (pour le fine-tuning) ---
def charger_donnees_entrainement_json(chemin_fichier_json):
    # ... (Fonction inchangée de la réponse précédente)
    try:
        with open(chemin_fichier_json, 'r', encoding='utf-8') as f:
            donnees = json.load(f)
        donnees_formatees = []
        for item in donnees:
            if isinstance(item, list) and len(item) == 2: 
                 texte, annotations = item
            elif isinstance(item, tuple) and len(item) == 2: 
                 texte, annotations = item
            else:
                texte, annotations = item[0], item[1]
            entites_formatees = []
            if "entities" in annotations and isinstance(annotations["entities"], list):
                for ent_item in annotations["entities"]:
                    if isinstance(ent_item, list) and len(ent_item) == 3:
                        entites_formatees.append(tuple(ent_item))
                    elif isinstance(ent_item, tuple) and len(ent_item) == 3:
                        entites_formatees.append(ent_item)
                    else:
                        raise ValueError(f"Format d'entité incorrect: {ent_item}")
                donnees_formatees.append((texte, {"entities": entites_formatees}))
            elif "entities" not in annotations :
                donnees_formatees.append((texte, annotations))
            else: 
                donnees_formatees.append((texte, annotations))
        if not donnees_formatees:
             messagebox.showwarning("Données Vides", f"Aucune donnée valide trouvée dans '{chemin_fichier_json}'.")
             return None
        return donnees_formatees
    except FileNotFoundError:
        messagebox.showerror("Erreur Fichier JSON", f"Fichier '{chemin_fichier_json}' introuvable.")
        return None
    except Exception as e:
        messagebox.showerror("Erreur Chargement JSON", f"Erreur chargement/formatage depuis '{chemin_fichier_json}': {e}")
        return None

# Variables globales
MODELES_SPACY_FR = {"Petit (sm)": "fr_core_news_sm", "Moyen (md)": "fr_core_news_md", "Grand (lg)": "fr_core_news_lg"}
modele_spacy_selectionne = None
chemin_output_donnees_spacy = None
chemin_modele_finetune_pour_test = None # Stockera le chemin du modèle qui vient d'être fine-tuné
mapping_pseudonymes_actuel = None # Stocke le mapping pour la sauvegarde

# --- Fonctions GUI ---
def valider_choix_modele():
    # ... (Fonction inchangée)
    global modele_spacy_selectionne
    choix_utilisateur_label = choix_modele_var.get()
    modele_spacy_selectionne = MODELES_SPACY_FR.get(choix_utilisateur_label)
    if modele_spacy_selectionne:
        if verifier_et_telecharger_modele(modele_spacy_selectionne):
            messagebox.showinfo("Modèle Prêt", f"Modèle sélectionné : {modele_spacy_selectionne}\nSélectionnez vos données d'entraînement.")
            activer_cadre_selection_donnees(True) 
            bouton_valider_modele.config(state="disabled")
            activer_cadre_fine_tuning(False)
            activer_cadre_test_modele(False) # S'assurer que le test est désactivé
        else:
            activer_cadre_selection_donnees(False)
    else:
        messagebox.showwarning("Aucun Modèle", "Veuillez sélectionner un modèle.")


def verifier_et_telecharger_modele(nom_modele):
    # ... (Fonction inchangée)
    try:
        spacy.load(nom_modele)
        return True
    except OSError:
        if messagebox.askyesno("Modèle Non Trouvé", f"'{nom_modele}' non trouvé. Voulez-vous le télécharger ?"):
            status_label_dl = ttk.Label(cadre_choix_modele, text=f"Téléchargement de {nom_modele}...")
            status_label_dl.pack(pady=2)
            fenetre.update_idletasks()
            try:
                subprocess.check_call(['python', '-m', 'spacy', 'download', nom_modele], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                status_label_dl.destroy()
                messagebox.showinfo("Téléchargement Réussi", f"'{nom_modele}' téléchargé.")
                return True
            except Exception as e: # Attrape toutes les erreurs de subprocess
                status_label_dl.destroy()
                messagebox.showerror("Erreur Téléchargement", f"Impossible de télécharger '{nom_modele}'. Vérifiez la console ou essayez manuellement.")
                print(f"Erreur subprocess: {e}")
                return False
        return False

def choisir_fichier_json_donnees():
    # ... (Fonction inchangée)
    global chemin_output_donnees_spacy
    chemin_fichier = filedialog.askopenfilename(title="Sélectionner le fichier JSON de données", filetypes=(("Fichiers JSON", "*.json"), ("Tous", "*.*")))
    if chemin_fichier:
        var_chemin_donnees_json.set(chemin_fichier)
        chemin_output_donnees_spacy = chemin_fichier
        label_statut_selection_donnees.config(text=f"Fichier : {os.path.basename(chemin_fichier)}")

def valider_fichier_donnees():
    # ... (Fonction inchangée)
    if not chemin_output_donnees_spacy or not os.path.exists(chemin_output_donnees_spacy):
        messagebox.showerror("Erreur", "Sélectionnez un fichier JSON valide.")
        label_statut_selection_donnees.config(text="Aucun fichier valide.")
        return
    try:
        with open(chemin_output_donnees_spacy, 'r', encoding='utf-8') as f: json.load(f)
        messagebox.showinfo("Données Prêtes", "Fichier de données validé.\nConfigurez le fine-tuning.")
        label_statut_selection_donnees.config(text="Fichier de données prêt.")
        bouton_valider_fichier_donnees.config(state="disabled")
        activer_cadre_fine_tuning(True)
    except Exception as e: # Attrape json.JSONDecodeError et autres erreurs de lecture
        messagebox.showerror("Erreur Fichier", f"Impossible de lire ou parser le fichier JSON : {e}")
        label_statut_selection_donnees.config(text="Erreur : Fichier JSON invalide.")


def choisir_dossier_sauvegarde_modele():
    # ... (Fonction inchangée)
    chemin_dossier = filedialog.askdirectory(title="Sélectionner le dossier de sauvegarde du modèle")
    if chemin_dossier: var_chemin_sauvegarde_modele.set(chemin_dossier)

def lancer_fine_tuning_gui():
    # ... (Début de la fonction inchangé : récupération et validation des params)
    global modele_spacy_selectionne, chemin_output_donnees_spacy, chemin_modele_finetune_pour_test
    if not modele_spacy_selectionne: messagebox.showerror("Erreur", "Modèle SpaCy non sélectionné."); return
    if not chemin_output_donnees_spacy or not os.path.exists(chemin_output_donnees_spacy): messagebox.showerror("Erreur", "Fichier de données JSON introuvable."); return
    try:
        iterations = var_iterations.get(); dropout = var_dropout.get(); chemin_sauvegarde = var_chemin_sauvegarde_modele.get()
        if iterations <= 0: messagebox.showerror("Config Erreur", "Itérations > 0."); return
        if not (0.0 <= dropout <= 1.0): messagebox.showerror("Config Erreur", "Dropout entre 0.0 et 1.0."); return
        if not chemin_sauvegarde: messagebox.showerror("Config Erreur", "Spécifiez un dossier de sauvegarde."); return
    except tk.TclError: messagebox.showerror("Config Erreur", "Valeurs numériques valides pour itérations/dropout."); return
    TRAIN_DATA = charger_donnees_entrainement_json(chemin_output_donnees_spacy)
    if not TRAIN_DATA: log_fine_tuning("Échec chargement données. Vérifiez JSON."); return

    log_fine_tuning("Fine-tuning démarré...\n" + f"Modèle: {modele_spacy_selectionne}, Données: {os.path.basename(chemin_output_donnees_spacy)} ({len(TRAIN_DATA)} ex.), It: {iterations}, Drop: {dropout}, Sauvegarde: {chemin_sauvegarde}\n")
    bouton_lancer_fine_tuning.config(state="disabled"); fenetre.update_idletasks()
    try:
        nlp = spacy.load(modele_spacy_selectionne)
        log_fine_tuning(f"Modèle '{modele_spacy_selectionne}' chargé.")
        ner = nlp.get_pipe("ner") if "ner" in nlp.pipe_names else nlp.add_pipe("ner", last=True)
        labels_dans_donnees = set()
        for _, annotations in TRAIN_DATA:
            for ent in annotations.get("entities", []): labels_dans_donnees.add(ent[2])
        for label in labels_dans_donnees: ner.add_label(label)
        log_fine_tuning(f"Labels pour NER : {labels_dans_donnees}")
        with nlp.select_pipes(disable=[p for p in nlp.pipe_names if p != "ner"]):
            optimizer = nlp.begin_training()
            for i in range(iterations):
                random.shuffle(TRAIN_DATA); pertes = {}
                for texte, annots in TRAIN_DATA:
                    try:
                        doc = nlp.make_doc(texte)
                        example = Example.from_dict(doc, annots)
                        nlp.update([example], sgd=optimizer, drop=dropout, losses=pertes)
                    except Exception as e_upd: log_fine_tuning(f"Err nlp.update (it {i+1}, ex. ignoré): {e_upd}"); continue
                log_fine_tuning(f"It {i+1}/{iterations} - Perte NER: {pertes.get('ner',0.0):.4f}")
                fenetre.update_idletasks()
        if not os.path.exists(chemin_sauvegarde): os.makedirs(chemin_sauvegarde)
        nlp.to_disk(chemin_sauvegarde)
        log_fine_tuning(f"\nModèle fine-tuné sauvegardé dans : '{chemin_sauvegarde}'")
        messagebox.showinfo("Fine-tuning Terminé", f"Modèle sauvegardé dans\n{chemin_sauvegarde}")
        chemin_modele_finetune_pour_test = chemin_sauvegarde # Sauvegarder pour l'étape de test
        activer_cadre_test_modele(True) # Activer le cadre de test
    except Exception as e:
        log_fine_tuning(f"\nErreur majeure fine-tuning : {e}"); messagebox.showerror("Erreur Fine-tuning", f"Erreur : {e}")
    finally:
        bouton_lancer_fine_tuning.config(state="normal")


def log_fine_tuning(message): # ... (inchangée)
    text_log_fine_tuning.config(state="normal"); text_log_fine_tuning.insert(tk.END, message + "\n"); text_log_fine_tuning.see(tk.END); text_log_fine_tuning.config(state="disabled"); fenetre.update_idletasks()

def activer_cadre_selection_donnees(activer): # ... (inchangée)
    etat = "normal" if activer else "disabled"; bouton_choisir_fichier_json.config(state=etat); entry_chemin_donnees_json.config(state="readonly" if activer else "disabled"); bouton_valider_fichier_donnees.config(state=etat)
    if not activer: var_chemin_donnees_json.set(""); label_statut_selection_donnees.config(text="")

def activer_cadre_fine_tuning(activer): # ... (inchangée)
    etat = "normal" if activer else "disabled"; entry_iterations.config(state=etat); entry_dropout.config(state=etat); entry_chemin_sauvegarde_modele.config(state="readonly" if activer else "disabled"); bouton_choisir_dossier_modele.config(state=etat); bouton_lancer_fine_tuning.config(state=etat); text_log_fine_tuning.config(state="normal" if activer else "disabled")
    if not activer: var_iterations.set(10); var_dropout.set(0.3); var_chemin_sauvegarde_modele.set(""); text_log_fine_tuning.config(state="normal"); text_log_fine_tuning.delete(1.0, tk.END); text_log_fine_tuning.config(state="disabled")

# --- Nouvelles Fonctions pour le Cadre de Test ---
def choisir_fichier_test_txt():
    chemin_fichier = filedialog.askopenfilename(title="Sélectionner le fichier texte pour le test", filetypes=(("Fichiers Texte", "*.txt"), ("Tous les fichiers", "*.*")))
    if chemin_fichier:
        var_chemin_fichier_test.set(chemin_fichier)
        label_statut_test.config(text=f"Fichier de test : {os.path.basename(chemin_fichier)}")

def lancer_test_pseudonymisation_gui():
    global chemin_modele_finetune_pour_test, mapping_pseudonymes_actuel
    
    path_fichier_test = var_chemin_fichier_test.get()
    if not chemin_modele_finetune_pour_test:
        messagebox.showerror("Erreur", "Aucun modèle fine-tuné n'est disponible pour le test. Veuillez d'abord entraîner un modèle.")
        return
    if not path_fichier_test or not os.path.exists(path_fichier_test):
        messagebox.showerror("Erreur", "Veuillez sélectionner un fichier texte de test valide.")
        return

    label_statut_test.config(text="Chargement du modèle fine-tuné...")
    fenetre.update_idletasks()
    try:
        nlp_test = spacy.load(chemin_modele_finetune_pour_test)
        label_statut_test.config(text="Lecture du fichier de test...")
        fenetre.update_idletasks()
        with open(path_fichier_test, 'r', encoding='utf-8') as f_test:
            texte_original_test = f_test.read()
        
        label_statut_test.config(text="Pseudonymisation en cours...")
        fenetre.update_idletasks()
        texte_pseudo, mapping = pseudonymiser_texte_pour_gui(nlp_test, texte_original_test)
        
        if texte_pseudo is not None:
            text_resultat_pseudo.config(state="normal")
            text_resultat_pseudo.delete(1.0, tk.END)
            text_resultat_pseudo.insert(tk.END, texte_pseudo)
            text_resultat_pseudo.config(state="disabled")
            label_statut_test.config(text="Pseudonymisation terminée.")
            mapping_pseudonymes_actuel = mapping # Sauvegarder pour une éventuelle sauvegarde
            bouton_sauvegarder_texte_pseudo.config(state="normal")
            bouton_sauvegarder_mapping_pseudo.config(state="normal")
        else:
            label_statut_test.config(text="Erreur durant la pseudonymisation.")
            
    except Exception as e:
        messagebox.showerror("Erreur Test", f"Une erreur est survenue lors du test : {e}")
        label_statut_test.config(text=f"Erreur : {e}")

def sauvegarder_texte_resultat():
    texte_a_sauvegarder = text_resultat_pseudo.get(1.0, tk.END).strip()
    if not texte_a_sauvegarder:
        messagebox.showwarning("Rien à sauvegarder", "Aucun texte pseudonymisé à sauvegarder.")
        return

    # Obtenir le chemin du fichier de test original pour suggérer un nom
    chemin_fichier_test_original = var_chemin_fichier_test.get() # var_chemin_fichier_test est la variable Tkinter liée à l'entrée du fichier de test
    nom_initial_suggerere = "resultat_pseudonymise.txt" # Nom par défaut si l'original n'est pas disponible

    if chemin_fichier_test_original and os.path.exists(chemin_fichier_test_original): # Vérifier si le chemin est valide
        nom_base = os.path.basename(chemin_fichier_test_original)
        nom_sans_ext, extension_originale = os.path.splitext(nom_base) # extension_originale pourrait être .txt
        nom_initial_suggerere = f"{nom_sans_ext}_pseudonymise.txt" # Suggestion de nom claire avec .txt
    
    # Ouvrir la boîte de dialogue "Enregistrer sous..."
    chemin_fichier = filedialog.asksaveasfilename(
        title="Sauvegarder le texte pseudonymisé",
        initialfile=nom_initial_suggerere, # Le nom de fichier suggéré apparaîtra ici
        defaultextension=".txt",
        filetypes=(("Fichiers Texte", "*.txt"), ("Tous les fichiers", "*.*"))
    )
    
    # Si l'utilisateur a choisi un chemin et cliqué sur "Enregistrer" (chemin_fichier ne sera pas vide)
    if chemin_fichier:
        try:
            with open(chemin_fichier, 'w', encoding='utf-8') as f:
                f.write(texte_a_sauvegarder)
            messagebox.showinfo("Succès", f"Texte pseudonymisé sauvegardé dans : {chemin_fichier}")
        except Exception as e:
            messagebox.showerror("Erreur Sauvegarde", f"Impossible de sauvegarder le fichier : {e}")
    # Si l'utilisateur annule la boîte de dialogue, chemin_fichier sera vide et rien ne se passera.
def sauvegarder_mapping_resultat():
    global mapping_pseudonymes_actuel
    if not mapping_pseudonymes_actuel:
        messagebox.showwarning("Rien à sauvegarder", "Aucune table de correspondance à sauvegarder.")
        return
    chemin_fichier = filedialog.asksaveasfilename(title="Sauvegarder la table de correspondance", defaultextension=".json", filetypes=(("Fichiers JSON", "*.json"),("Tous les fichiers", "*.*")))
    if chemin_fichier:
        try:
            with open(chemin_fichier, 'w', encoding='utf-8') as f:
                json.dump(mapping_pseudonymes_actuel, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Succès", f"Table de correspondance sauvegardée dans : {chemin_fichier}")
        except Exception as e:
            messagebox.showerror("Erreur Sauvegarde", f"Impossible de sauvegarder le fichier JSON : {e}")


def activer_cadre_test_modele(activer):
    etat = "normal" if activer else "disabled"
    bouton_choisir_fichier_test.config(state=etat)
    entry_chemin_fichier_test.config(state="readonly" if activer else "disabled")
    bouton_lancer_test.config(state=etat)
    text_resultat_pseudo.config(state="disabled") # Toujours désactivé pour l'édition directe
    bouton_sauvegarder_texte_pseudo.config(state="disabled" if not activer else "normal" if text_resultat_pseudo.get(1.0, tk.END).strip() else "disabled")
    bouton_sauvegarder_mapping_pseudo.config(state="disabled" if not activer else "normal" if mapping_pseudonymes_actuel else "disabled")
    if not activer:
        var_chemin_fichier_test.set("")
        label_statut_test.config(text="")
        text_resultat_pseudo.config(state="normal")
        text_resultat_pseudo.delete(1.0, tk.END)
        text_resultat_pseudo.config(state="disabled")

# --- Création de la fenêtre principale ---
fenetre = tk.Tk()
fenetre.title("Assistant de Fine-tuning et Test LLM SpaCy")
fenetre.geometry("750x850") # Agrandir pour le nouveau cadre

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

# --- Cadre 2: Sélection des Données d'Entraînement Combinées ---
# ... (inchangé par rapport à la version précédente avec sélection de JSON)
cadre_selection_donnees = ttk.LabelFrame(fenetre, text="2. Sélection des Données d'Entraînement (JSON)", padding=(10, 10))
cadre_selection_donnees.pack(padx=10, pady=10, fill="x")
var_chemin_donnees_json = tk.StringVar()
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
# ... (inchangé)
cadre_fine_tuning = ttk.LabelFrame(fenetre, text="3. Fine-tuning du Modèle", padding=(10, 10))
cadre_fine_tuning.pack(padx=10, pady=10, fill="x") # fill x, pas both pour laisser place au cadre 4
var_iterations = tk.IntVar(value=10); var_dropout = tk.DoubleVar(value=0.3); var_chemin_sauvegarde_modele = tk.StringVar()
frame_iter = ttk.Frame(cadre_fine_tuning); frame_iter.pack(fill="x", pady=2)
ttk.Label(frame_iter, text="Nombre d'itérations:", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_iterations = ttk.Spinbox(frame_iter, from_=1, to=1000, textvariable=var_iterations, width=10); entry_iterations.pack(side=tk.LEFT)
frame_drop = ttk.Frame(cadre_fine_tuning); frame_drop.pack(fill="x", pady=2)
ttk.Label(frame_drop, text="Taux de Dropout (0.0-1.0):", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_dropout = ttk.Spinbox(frame_drop, from_=0.0, to=1.0, increment=0.05, textvariable=var_dropout, width=10, format="%.2f"); entry_dropout.pack(side=tk.LEFT)
frame_sauvegarde_modele = ttk.Frame(cadre_fine_tuning); frame_sauvegarde_modele.pack(fill="x", pady=2)
ttk.Label(frame_sauvegarde_modele, text="Dossier de sauvegarde du modèle:", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_chemin_sauvegarde_modele = ttk.Entry(frame_sauvegarde_modele, textvariable=var_chemin_sauvegarde_modele, state="readonly", width=40); entry_chemin_sauvegarde_modele.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_dossier_modele = ttk.Button(frame_sauvegarde_modele, text="Parcourir...", command=choisir_dossier_sauvegarde_modele); bouton_choisir_dossier_modele.pack(side=tk.LEFT)
bouton_lancer_fine_tuning = ttk.Button(cadre_fine_tuning, text="Lancer le Fine-tuning", command=lancer_fine_tuning_gui); bouton_lancer_fine_tuning.pack(pady=10)
ttk.Label(cadre_fine_tuning, text="Log du Fine-tuning:").pack(anchor="w", pady=(5,0))
text_log_fine_tuning = scrolledtext.ScrolledText(cadre_fine_tuning, height=8, width=80, state="disabled", wrap=tk.WORD); text_log_fine_tuning.pack(pady=5, fill="x", expand=False)


# --- Cadre 4: Tester le Modèle Fine-tuné ---
cadre_test_modele = ttk.LabelFrame(fenetre, text="4. Tester le Modèle Fine-tuné", padding=(10, 10))
cadre_test_modele.pack(padx=10, pady=10, fill="both", expand=True)

var_chemin_fichier_test = tk.StringVar()

# Sélection du fichier de test
frame_fichier_test = ttk.Frame(cadre_test_modele)
frame_fichier_test.pack(fill="x", pady=5)
ttk.Label(frame_fichier_test, text="Fichier Texte de Test (.txt):", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_chemin_fichier_test = ttk.Entry(frame_fichier_test, textvariable=var_chemin_fichier_test, state="readonly", width=40)
entry_chemin_fichier_test.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_fichier_test = ttk.Button(frame_fichier_test, text="Parcourir...", command=choisir_fichier_test_txt)
bouton_choisir_fichier_test.pack(side=tk.LEFT)

# Bouton pour lancer le test
bouton_lancer_test = ttk.Button(cadre_test_modele, text="Lancer le Test de Pseudonymisation", command=lancer_test_pseudonymisation_gui)
bouton_lancer_test.pack(pady=10)

# Label de statut pour le test
label_statut_test = ttk.Label(cadre_test_modele, text="")
label_statut_test.pack(pady=2)

# Zone de texte pour afficher le résultat pseudonymisé
ttk.Label(cadre_test_modele, text="Résultat de la Pseudonymisation:").pack(anchor="w", pady=(5,0))
text_resultat_pseudo = scrolledtext.ScrolledText(cadre_test_modele, height=10, width=80, state="disabled", wrap=tk.WORD)
text_resultat_pseudo.pack(pady=5, fill="both", expand=True)

# Boutons de sauvegarde pour le résultat du test
frame_sauvegarde_test = ttk.Frame(cadre_test_modele)
frame_sauvegarde_test.pack(pady=5)
bouton_sauvegarder_texte_pseudo = ttk.Button(frame_sauvegarde_test, text="Sauvegarder Texte Pseudonymisé", command=sauvegarder_texte_resultat, state="disabled")
bouton_sauvegarder_texte_pseudo.pack(side=tk.LEFT, padx=5)
bouton_sauvegarder_mapping_pseudo = ttk.Button(frame_sauvegarde_test, text="Sauvegarder Mapping", command=sauvegarder_mapping_resultat, state="disabled")
bouton_sauvegarder_mapping_pseudo.pack(side=tk.LEFT, padx=5)


# Initialisation des états des cadres
activer_cadre_selection_donnees(False)
activer_cadre_fine_tuning(False)
activer_cadre_test_modele(False)

fenetre.mainloop()