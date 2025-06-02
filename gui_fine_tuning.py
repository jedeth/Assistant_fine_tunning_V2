import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import spacy
from spacy.training.example import Example
import os
import json # Changé re en json car re n'était plus utilisé après suppression de generer_donnees_entrainement_interne
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
        if entite.label_ == "PER": 
            nom_original = entite.text.strip() # Ajout de strip() pour la cohérence
            
            if nom_original not in correspondances:
                pseudonyme = f"[PERSONNE_{pseudonyme_compteur}]"
                correspondances[nom_original] = pseudonyme
                pseudonyme_compteur += 1
            else:
                pseudonyme = correspondances[nom_original]
            
            entites_a_remplacer.append((entite.start_char, entite.end_char, pseudonyme))

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
    try:
        with open(chemin_fichier_json, 'r', encoding='utf-8') as f:
            donnees_brutes = json.load(f)
        
        if not isinstance(donnees_brutes, list):
            raise ValueError("Le fichier JSON doit contenir une liste d'exemples.")

        donnees_formatees = []
        for item in donnees_brutes:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(f"Chaque exemple doit être une liste ou un tuple de deux éléments: [texte, annotations_dict]. Trouvé: {item}")
            
            texte, annotations_dict = item
            
            if not isinstance(texte, str):
                raise ValueError(f"Le premier élément de l'exemple (texte) doit être une chaîne. Trouvé: {type(texte)}")
            if not isinstance(annotations_dict, dict):
                raise ValueError(f"Le deuxième élément de l'exemple (annotations) doit être un dictionnaire. Trouvé: {type(annotations_dict)}")

            entites_tuples = []
            if "entities" in annotations_dict:
                if not isinstance(annotations_dict["entities"], list):
                    raise ValueError(f"La clé 'entities' doit contenir une liste. Trouvé: {type(annotations_dict['entities'])}")
                
                for ent_item in annotations_dict["entities"]:
                    if not (isinstance(ent_item, (list, tuple)) and len(ent_item) == 3):
                        raise ValueError(f"Chaque entité doit être une liste/tuple de 3 éléments [début, fin, label]. Trouvé: {ent_item}")
                    
                    if not (isinstance(ent_item[0], int) and isinstance(ent_item[1], int)):
                         raise ValueError(f"Les indices de début/fin d'entité doivent être des entiers. Trouvé: {ent_item[:2]}")
                    if not isinstance(ent_item[2], str):
                        raise ValueError(f"Le label d'entité doit être une chaîne. Trouvé: {type(ent_item[2])}")
                        
                    entites_tuples.append(tuple(ent_item)) 
            
            donnees_formatees.append((texte, {"entities": entites_tuples}))

        if not donnees_formatees and donnees_brutes:
             messagebox.showwarning("Données Vides", f"Aucune donnée d'entraînement valide formatée depuis '{chemin_fichier_json}'. Vérifiez la structure.")
             return None
        elif not donnees_formatees: 
             messagebox.showwarning("Données Vides", f"Le fichier '{chemin_fichier_json}' est vide ou ne contient pas de données.")
             return None
        return donnees_formatees
    except FileNotFoundError:
        messagebox.showerror("Erreur Fichier JSON", f"Fichier de données d'entraînement '{chemin_fichier_json}' introuvable.")
        return None
    except json.JSONDecodeError:
        messagebox.showerror("Erreur JSON", f"Le fichier '{chemin_fichier_json}' n'est pas un JSON valide.")
        return None
    except Exception as e:
        messagebox.showerror("Erreur Chargement JSON", f"Erreur lors du chargement ou du formatage des données depuis '{chemin_fichier_json}': {e}")
        return None

# Variables globales
MODELES_SPACY_FR_BASE = { 
    "Petit (sm)": "fr_core_news_sm",
    "Moyen (md)": "fr_core_news_md",
    "Grand (lg)": "fr_core_news_lg",
}
OPTION_MODELE_EXISTANT = "Modèle fine-tuné existant..." 

modele_spacy_selectionne_pour_ft = None 
chemin_donnees_entrainement_final = None 
chemin_modele_a_tester = None 
mapping_pseudonymes_actuel = None

# --- Fonctions GUI ---
def on_model_type_changed(event=None):
    selection_label = choix_modele_var.get()
    if selection_label == OPTION_MODELE_EXISTANT:
        frame_custom_model_path.pack(fill="x", pady=(5,0), before=bouton_valider_modele_et_continuer) # Modifié pour être avant le bouton principal
        bouton_valider_modele_et_continuer.config(text="Valider Modèle Existant et Activer Test")
    else:
        frame_custom_model_path.pack_forget()
        bouton_valider_modele_et_continuer.config(text="Valider Modèle de Base et Continuer")

def valider_choix_modele():
    global modele_spacy_selectionne_pour_ft, chemin_modele_a_tester
    
    choix_utilisateur_label = choix_modele_var.get()
    valeur_modele_base = MODELES_SPACY_FR_BASE.get(choix_utilisateur_label)

    # Réinitialiser les états des cadres suivants à chaque validation de modèle
    activer_cadre_selection_donnees(False)
    activer_cadre_fine_tuning(False)
    activer_cadre_test_modele(False)

    if choix_utilisateur_label == OPTION_MODELE_EXISTANT:
        path_custom = var_custom_model_path.get()
        if not path_custom or not os.path.isdir(path_custom): 
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier de modèle fine-tuné valide.")
            return
        try:
            spacy.load(path_custom) 
            messagebox.showinfo("Modèle Valide", f"Modèle existant '{os.path.basename(path_custom)}' validé.\nVous pouvez passer directement au test.")
            chemin_modele_a_tester = path_custom
            modele_spacy_selectionne_pour_ft = None 
            activer_cadre_test_modele(True)
            bouton_valider_modele_et_continuer.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Erreur Chargement Modèle", f"Impossible de charger le modèle depuis '{path_custom}'.\nErreur: {e}")
            chemin_modele_a_tester = None
            return
    elif valeur_modele_base: 
        modele_spacy_selectionne_pour_ft = valeur_modele_base
        chemin_modele_a_tester = None 
        if verifier_et_telecharger_modele(modele_spacy_selectionne_pour_ft):
            messagebox.showinfo("Modèle Prêt", f"Modèle de base sélectionné : {modele_spacy_selectionne_pour_ft}\nPassez à la sélection des données.")
            activer_cadre_selection_donnees(True)
            bouton_valider_modele_et_continuer.config(state="disabled")
        # else: rien à faire de plus, verifier_et_telecharger_modele gère les messages
    else:
        messagebox.showwarning("Aucun Modèle", "Veuillez sélectionner une option valide dans la liste.")


def verifier_et_telecharger_modele(nom_modele):
    try:
        spacy.load(nom_modele)
        return True
    except OSError:
        if messagebox.askyesno("Modèle Non Trouvé", f"Le modèle '{nom_modele}' n'est pas trouvé. Voulez-vous le télécharger ?"):
            status_label_dl = ttk.Label(cadre_choix_modele, text=f"Téléchargement de {nom_modele}...")
            status_label_dl.pack(pady=2)
            fenetre.update_idletasks()
            try:
                subprocess.check_call(['python', '-m', 'spacy', 'download', nom_modele], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                messagebox.showinfo("Téléchargement Réussi", f"Modèle '{nom_modele}' téléchargé.")
                return True
            except Exception as e: 
                messagebox.showerror("Erreur Téléchargement", f"Impossible de télécharger '{nom_modele}'.\n{e}\nVérifiez la console ou essayez manuellement.")
                return False
            finally: # Assurer que le label de statut est enlevé
                if 'status_label_dl' in locals() and status_label_dl.winfo_exists():
                    status_label_dl.destroy()
        return False

def choisir_fichier_json_donnees():
    global chemin_donnees_entrainement_final # Modifié pour utiliser cette variable
    chemin_fichier = filedialog.askopenfilename(title="Sélectionner le fichier JSON de données", filetypes=(("Fichiers JSON", "*.json"), ("Tous", "*.*")))
    if chemin_fichier:
        var_chemin_donnees_json.set(chemin_fichier)
        chemin_donnees_entrainement_final = chemin_fichier # Mise à jour
        label_statut_selection_donnees.config(text=f"Fichier : {os.path.basename(chemin_fichier)}")

def valider_fichier_donnees():
    if not chemin_donnees_entrainement_final or not os.path.exists(chemin_donnees_entrainement_final):
        messagebox.showerror("Erreur", "Sélectionnez un fichier JSON valide.")
        label_statut_selection_donnees.config(text="Aucun fichier valide.")
        return
    try:
        # Test de chargement pour valider le format général
        test_data = charger_donnees_entrainement_json(chemin_donnees_entrainement_final)
        if test_data is None: # Si charger_donnees_entrainement_json retourne None à cause d'une erreur interne déjà affichée
            label_statut_selection_donnees.config(text="Erreur : Fichier JSON invalide ou vide.")
            return

        messagebox.showinfo("Données Prêtes", "Fichier de données validé.\nConfigurez le fine-tuning.")
        label_statut_selection_donnees.config(text="Fichier de données prêt.")
        bouton_valider_fichier_donnees.config(state="disabled")
        activer_cadre_fine_tuning(True)
    except Exception as e: 
        messagebox.showerror("Erreur Fichier", f"Impossible de lire ou parser le fichier JSON : {e}")
        label_statut_selection_donnees.config(text="Erreur : Fichier JSON invalide.")

# ***** DÉFINITION DE LA FONCTION MANQUANTE AJOUTÉE ICI *****
def choisir_dossier_sauvegarde_modele():
    chemin_dossier = filedialog.askdirectory(title="Sélectionner le dossier pour sauvegarder le modèle fine-tuné")
    if chemin_dossier:
        var_chemin_sauvegarde_modele.set(chemin_dossier)
# ***** FIN DE L'AJOUT *****

def lancer_fine_tuning_gui():
    global modele_spacy_selectionne_pour_ft, chemin_donnees_entrainement_final, chemin_modele_a_tester
    if not modele_spacy_selectionne_pour_ft: messagebox.showerror("Erreur", "Modèle SpaCy de base non sélectionné pour le fine-tuning."); return
    if not chemin_donnees_entrainement_final or not os.path.exists(chemin_donnees_entrainement_final): messagebox.showerror("Erreur", "Fichier de données JSON introuvable."); return
    try:
        iterations = var_iterations.get(); dropout = var_dropout.get(); chemin_sauvegarde = var_chemin_sauvegarde_modele.get()
        if iterations <= 0: messagebox.showerror("Config Erreur", "Itérations > 0."); return
        if not (0.0 <= dropout <= 1.0): messagebox.showerror("Config Erreur", "Dropout entre 0.0 et 1.0."); return
        if not chemin_sauvegarde: messagebox.showerror("Config Erreur", "Spécifiez un dossier de sauvegarde."); return
    except tk.TclError: messagebox.showerror("Config Erreur", "Valeurs numériques valides pour itérations/dropout."); return
    
    TRAIN_DATA = charger_donnees_entrainement_json(chemin_donnees_entrainement_final)
    if not TRAIN_DATA: log_fine_tuning("Échec chargement données. Vérifiez JSON et sa structure."); return

    log_fine_tuning("Fine-tuning démarré...\n" + f"Modèle: {modele_spacy_selectionne_pour_ft}, Données: {os.path.basename(chemin_donnees_entrainement_final)} ({len(TRAIN_DATA)} ex.), It: {iterations}, Drop: {dropout}, Sauvegarde: {chemin_sauvegarde}\n")
    bouton_lancer_fine_tuning.config(state="disabled"); fenetre.update_idletasks()
    try:
        nlp = spacy.load(modele_spacy_selectionne_pour_ft)
        log_fine_tuning(f"Modèle '{modele_spacy_selectionne_pour_ft}' chargé.")
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
        chemin_modele_a_tester = chemin_sauvegarde 
        activer_cadre_test_modele(True) 
    except Exception as e:
        log_fine_tuning(f"\nErreur majeure fine-tuning : {e}"); messagebox.showerror("Erreur Fine-tuning", f"Erreur : {e}")
    finally:
        bouton_lancer_fine_tuning.config(state="normal")

def log_fine_tuning(message): 
    text_log_fine_tuning.config(state="normal"); text_log_fine_tuning.insert(tk.END, message + "\n"); text_log_fine_tuning.see(tk.END); text_log_fine_tuning.config(state="disabled"); fenetre.update_idletasks()

def activer_cadre_selection_donnees(activer): 
    etat = "normal" if activer else "disabled"
    bouton_choisir_fichier_json.config(state=etat)
    entry_chemin_donnees_json.config(state="readonly" if activer else "disabled")
    bouton_valider_fichier_donnees.config(state=etat)
    if not activer: 
        var_chemin_donnees_json.set("")
        label_statut_selection_donnees.config(text="")
    fenetre.update_idletasks()

def activer_cadre_fine_tuning(activer): 
    etat = "normal" if activer else "disabled"
    entry_iterations.config(state=etat)
    entry_dropout.config(state=etat)
    entry_chemin_sauvegarde_modele.config(state="readonly" if activer else "disabled")
    bouton_choisir_dossier_modele.config(state=etat)
    bouton_lancer_fine_tuning.config(state=etat)
    text_log_fine_tuning.config(state="normal" if activer else "disabled") # Pour pouvoir écrire dedans
    if not activer:
        var_iterations.set(10); var_dropout.set(0.3); var_chemin_sauvegarde_modele.set("")
        text_log_fine_tuning.config(state="normal"); text_log_fine_tuning.delete(1.0, tk.END); text_log_fine_tuning.config(state="disabled")
    fenetre.update_idletasks()

def choisir_fichier_test_txt():
    chemin_fichier = filedialog.askopenfilename(title="Sélectionner le fichier texte pour le test", filetypes=(("Fichiers Texte", "*.txt"), ("Tous les fichiers", "*.*")))
    if chemin_fichier:
        var_chemin_fichier_test.set(chemin_fichier)
        label_statut_test.config(text=f"Fichier de test : {os.path.basename(chemin_fichier)}")

def lancer_test_pseudonymisation_gui():
    global chemin_modele_a_tester, mapping_pseudonymes_actuel 
    path_fichier_test = var_chemin_fichier_test.get()
    if not chemin_modele_a_tester: messagebox.showerror("Erreur", "Aucun modèle (fine-tuné ou existant) n'est prêt pour le test."); return
    if not path_fichier_test or not os.path.exists(path_fichier_test): messagebox.showerror("Erreur", "Sélectionnez un fichier texte de test valide."); return
    label_statut_test.config(text="Chargement du modèle..."); fenetre.update_idletasks()
    try:
        nlp_test = spacy.load(chemin_modele_a_tester)
        label_statut_test.config(text="Lecture fichier test..."); fenetre.update_idletasks()
        with open(path_fichier_test, 'r', encoding='utf-8') as f_test: texte_original_test = f_test.read()
        label_statut_test.config(text="Pseudonymisation..."); fenetre.update_idletasks()
        texte_pseudo, mapping = pseudonymiser_texte_pour_gui(nlp_test, texte_original_test)
        if texte_pseudo is not None:
            text_resultat_pseudo.config(state="normal"); text_resultat_pseudo.delete(1.0, tk.END); text_resultat_pseudo.insert(tk.END, texte_pseudo); text_resultat_pseudo.config(state="disabled")
            label_statut_test.config(text="Pseudonymisation terminée.")
            mapping_pseudonymes_actuel = mapping 
            bouton_sauvegarder_texte_pseudo.config(state="normal"); bouton_sauvegarder_mapping_pseudo.config(state="normal")
        else: label_statut_test.config(text="Erreur pseudonymisation.")
    except Exception as e: messagebox.showerror("Erreur Test", f"Erreur test : {e}"); label_statut_test.config(text=f"Erreur : {e}")

def sauvegarder_texte_resultat():
    texte_a_sauvegarder = text_resultat_pseudo.get(1.0, tk.END).strip()
    if not texte_a_sauvegarder: messagebox.showwarning("Rien à sauvegarder", "Aucun texte pseudonymisé."); return
    chemin_input = var_chemin_fichier_test.get()
    nom_sugg = f"{os.path.splitext(os.path.basename(chemin_input))[0]}_pseudonymise.txt" if chemin_input and os.path.exists(chemin_input) else "resultat_pseudonymise.txt"
    chemin_fichier = filedialog.asksaveasfilename(title="Sauvegarder texte pseudonymisé", initialfile=nom_sugg, defaultextension=".txt", filetypes=(("Texte", "*.txt"),("Tous", "*.*")))
    if chemin_fichier:
        try:
            with open(chemin_fichier, 'w', encoding='utf-8') as f: f.write(texte_a_sauvegarder)
            messagebox.showinfo("Succès", f"Texte sauvegardé : {chemin_fichier}")
        except Exception as e: messagebox.showerror("Erreur Sauvegarde", f"Impossible de sauvegarder : {e}")

def sauvegarder_mapping_resultat():
    global mapping_pseudonymes_actuel
    if not mapping_pseudonymes_actuel: messagebox.showwarning("Rien à sauvegarder", "Aucune table de correspondance."); return
    chemin_input = var_chemin_fichier_test.get()
    nom_sugg = f"{os.path.splitext(os.path.basename(chemin_input))[0]}_mapping.json" if chemin_input and os.path.exists(chemin_input) else "mapping_pseudonymes.json"
    chemin_fichier = filedialog.asksaveasfilename(title="Sauvegarder table de correspondance", initialfile=nom_sugg, defaultextension=".json", filetypes=(("JSON", "*.json"),("Tous", "*.*")))
    if chemin_fichier:
        try:
            with open(chemin_fichier, 'w', encoding='utf-8') as f: json.dump(mapping_pseudonymes_actuel, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Succès", f"Mapping sauvegardé : {chemin_fichier}")
        except Exception as e: messagebox.showerror("Erreur Sauvegarde", f"Impossible de sauvegarder JSON : {e}")

def activer_cadre_test_modele(activer):
    etat = "normal" if activer else "disabled"
    bouton_choisir_fichier_test.config(state=etat)
    entry_chemin_fichier_test.config(state="readonly" if activer else "disabled")
    bouton_lancer_test.config(state=etat)
    if not activer:
        var_chemin_fichier_test.set(""); label_statut_test.config(text="")
        text_resultat_pseudo.config(state="normal"); text_resultat_pseudo.delete(1.0, tk.END); text_resultat_pseudo.config(state="disabled")
        bouton_sauvegarder_texte_pseudo.config(state="disabled"); bouton_sauvegarder_mapping_pseudo.config(state="disabled")
    fenetre.update_idletasks()

# --- Création de la fenêtre principale ---
fenetre = tk.Tk()
fenetre.title("Assistant de Fine-tuning et Test LLM SpaCy")
fenetre.geometry("800x700") 

# --- Cadre 1: Choix du Modèle ---
cadre_choix_modele = ttk.LabelFrame(fenetre, text="1. Choix du Modèle", padding=(10, 10))
cadre_choix_modele.pack(padx=10, pady=10, fill="x")
ttk.Label(cadre_choix_modele, text="Modèle SpaCy français ou chemin vers modèle existant :").pack(pady=(0,5), anchor="w")
choix_modele_var = tk.StringVar(fenetre)
options_modeles_labels = list(MODELES_SPACY_FR_BASE.keys()) + [OPTION_MODELE_EXISTANT]
menu_deroulant_modeles = ttk.Combobox(cadre_choix_modele, textvariable=choix_modele_var, values=options_modeles_labels, state="readonly", width=40)
if options_modeles_labels: menu_deroulant_modeles.current(1) 
menu_deroulant_modeles.pack(pady=5, fill="x")
menu_deroulant_modeles.bind("<<ComboboxSelected>>", on_model_type_changed)

frame_custom_model_path = ttk.Frame(cadre_choix_modele) # Sera packé dynamiquement
var_custom_model_path = tk.StringVar()
ttk.Label(frame_custom_model_path, text="Chemin modèle existant:", width=22).pack(side=tk.LEFT)
entry_custom_model_path = ttk.Entry(frame_custom_model_path, textvariable=var_custom_model_path, width=30, state="readonly")
entry_custom_model_path.pack(side=tk.LEFT, expand=True, fill="x", padx=2)
bouton_browse_custom_model = ttk.Button(frame_custom_model_path, text="...", width=3, command=lambda: var_custom_model_path.set(filedialog.askdirectory(title="Sélectionner dossier du modèle fine-tuné")))
bouton_browse_custom_model.pack(side=tk.LEFT)

bouton_valider_modele_et_continuer = ttk.Button(cadre_choix_modele, text="Valider Modèle de Base et Continuer", command=valider_choix_modele) # Nom de variable du bouton modifié
bouton_valider_modele_et_continuer.pack(pady=(10,0))
on_model_type_changed() # Appel initial

# --- Cadre 2: Sélection des Données d'Entraînement Combinées ---
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
cadre_fine_tuning = ttk.LabelFrame(fenetre, text="3. Fine-tuning du Modèle", padding=(10, 10))
cadre_fine_tuning.pack(padx=10, pady=10, fill="x")
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
bouton_choisir_dossier_modele = ttk.Button(frame_sauvegarde_modele, text="Parcourir...", command=choisir_dossier_sauvegarde_modele); bouton_choisir_dossier_modele.pack(side=tk.LEFT) # Ligne 531 dans le fichier de l'utilisateur
bouton_lancer_fine_tuning = ttk.Button(cadre_fine_tuning, text="Lancer le Fine-tuning", command=lancer_fine_tuning_gui); bouton_lancer_fine_tuning.pack(pady=10)
ttk.Label(cadre_fine_tuning, text="Log du Fine-tuning:").pack(anchor="w", pady=(5,0))
text_log_fine_tuning = scrolledtext.ScrolledText(cadre_fine_tuning, height=8, width=80, state="disabled", wrap=tk.WORD); text_log_fine_tuning.pack(pady=5, fill="x", expand=False)

# --- Cadre 4: Tester le Modèle Fine-tuné ---
cadre_test_modele = ttk.LabelFrame(fenetre, text="4. Tester le Modèle Fine-tuné", padding=(10, 10))
cadre_test_modele.pack(padx=10, pady=10, fill="both", expand=True)
var_chemin_fichier_test = tk.StringVar()
frame_fichier_test = ttk.Frame(cadre_test_modele); frame_fichier_test.pack(fill="x", pady=5)
ttk.Label(frame_fichier_test, text="Fichier Texte de Test (.txt):", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_chemin_fichier_test = ttk.Entry(frame_fichier_test, textvariable=var_chemin_fichier_test, state="readonly", width=40); entry_chemin_fichier_test.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_fichier_test = ttk.Button(frame_fichier_test, text="Parcourir...", command=choisir_fichier_test_txt); bouton_choisir_fichier_test.pack(side=tk.LEFT)
bouton_lancer_test = ttk.Button(cadre_test_modele, text="Lancer le Test de Pseudonymisation", command=lancer_test_pseudonymisation_gui); bouton_lancer_test.pack(pady=10)
label_statut_test = ttk.Label(cadre_test_modele, text=""); label_statut_test.pack(pady=2)
ttk.Label(cadre_test_modele, text="Résultat de la Pseudonymisation:").pack(anchor="w", pady=(5,0))
text_resultat_pseudo = scrolledtext.ScrolledText(cadre_test_modele, height=10, width=80, state="disabled", wrap=tk.WORD); text_resultat_pseudo.pack(pady=5, fill="both", expand=True)
frame_sauvegarde_test = ttk.Frame(cadre_test_modele); frame_sauvegarde_test.pack(pady=5)
bouton_sauvegarder_texte_pseudo = ttk.Button(frame_sauvegarde_test, text="Sauvegarder Texte Pseudonymisé", command=sauvegarder_texte_resultat, state="disabled"); bouton_sauvegarder_texte_pseudo.pack(side=tk.LEFT, padx=5)
bouton_sauvegarder_mapping_pseudo = ttk.Button(frame_sauvegarde_test, text="Sauvegarder Mapping", command=sauvegarder_mapping_resultat, state="disabled"); bouton_sauvegarder_mapping_pseudo.pack(side=tk.LEFT, padx=5)

# Initialisation des états des cadres
activer_cadre_selection_donnees(False)
activer_cadre_fine_tuning(False)
activer_cadre_test_modele(False)

fenetre.mainloop()