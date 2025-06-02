import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import spacy
from spacy.training.example import Example
import os
import re 
import json
import random

# --- Logique de Pseudonymisation (de l'étape 4 - Test) ---
def pseudonymiser_texte_pour_gui(nlp_model, texte_original):
    if not nlp_model: 
        messagebox.showerror("Erreur Modèle", "Modèle SpaCy non chargé pour la pseudonymisation.")
        return None, None
    doc = nlp_model(texte_original)
    correspondances = {}
    pseudonyme_compteur = 1
    entites_a_remplacer = []
    for entite in doc.ents:
        if entite.label_ == "PER": 
            nom_o = entite.text.strip() # Normalisation
            if nom_o not in correspondances: 
                pseudonyme = f"[PERSONNE_{pseudonyme_compteur}]"
                correspondances[nom_o] = pseudonyme
                pseudonyme_compteur += 1
            else: 
                pseudonyme = correspondances[nom_o]
            entites_a_remplacer.append((entite.start_char, entite.end_char, pseudonyme))
    entites_a_remplacer.sort(key=lambda x: x[0], reverse=True)
    parts = []
    last_idx = len(texte_original)
    for start, end, pseudo in entites_a_remplacer:
        if end < last_idx: parts.append(texte_original[end:last_idx])
        parts.append(pseudo)
        last_idx = start
    parts.append(texte_original[0:last_idx])
    return "".join(reversed(parts)), correspondances

# --- Fonctions de préparation de données (issues de preparer_donnees_multi_types.py) ---
def lire_entites_depuis_fichier(chemin_fichier, log_callback=None):
    entites = []
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            for ligne in f:
                entite_texte = ligne.strip()
                if entite_texte:
                    entites.append(entite_texte)
        if not entites and log_callback: log_callback(f"Attention : Aucun contenu trouvé dans : {chemin_fichier}")
        return entites
    except FileNotFoundError:
        msg = f"ERREUR : Fichier '{chemin_fichier}' introuvable."
        if log_callback: log_callback(msg)
        else: messagebox.showerror("Erreur Fichier", msg)
        return None
    except Exception as e:
        msg = f"ERREUR lecture '{chemin_fichier}': {e}"
        if log_callback: log_callback(msg)
        else: messagebox.showerror("Erreur Lecture", msg)
        return None

def lire_phrases_modeles_specifiques(chemin_fichier_phrases, placeholder_attendu, log_callback=None):
    phrases = []
    try:
        with open(chemin_fichier_phrases, 'r', encoding='utf-8') as f:
            for ligne_num, ligne in enumerate(f, 1):
                phrase_modele = ligne.strip()
                if phrase_modele.startswith('"') and phrase_modele.endswith('",'): phrase_modele = phrase_modele[1:-2]
                elif phrase_modele.startswith("'") and phrase_modele.endswith("',"): phrase_modele = phrase_modele[1:-2]
                elif phrase_modele.startswith('"') and phrase_modele.endswith('"'): phrase_modele = phrase_modele[1:-1]
                elif phrase_modele.startswith("'") and phrase_modele.endswith("'"): phrase_modele = phrase_modele[1:-1]
                if phrase_modele and placeholder_attendu in phrase_modele: phrases.append(phrase_modele)
                elif phrase_modele and log_callback: log_callback(f"Ligne {ligne_num} ({os.path.basename(chemin_fichier_phrases)}): Placeholder '{placeholder_attendu}' manquant ou phrase vide.")
        if not phrases and log_callback: log_callback(f"Attention : Aucun modèle de phrase valide pour '{placeholder_attendu}' dans : {os.path.basename(chemin_fichier_phrases)}")
        return phrases
    except FileNotFoundError:
        msg = f"ERREUR : Fichier modèles phrases '{chemin_fichier_phrases}' introuvable."
        if log_callback: log_callback(msg)
        else: messagebox.showerror("Erreur Fichier", msg)
        return None
    except Exception as e:
        msg = f"ERREUR lecture '{chemin_fichier_phrases}': {e}"
        if log_callback: log_callback(msg)
        else: messagebox.showerror("Erreur Lecture", msg)
        return None

def generer_donnees_pour_type(liste_entites, liste_phrases_modeles, label_entite, placeholder, log_callback=None):
    donnees_entrainement = []
    if not liste_entites or not liste_phrases_modeles:
        if log_callback: log_callback(f"Données sources vides pour label '{label_entite}'.")
        return donnees_entrainement
    for entite_texte in liste_entites:
        for phrase_modele in liste_phrases_modeles:
            phrase_formatee = phrase_modele.replace(placeholder, entite_texte)
            match = re.search(re.escape(entite_texte), phrase_formatee)
            if match:
                debut, fin = match.span()
                entite_spacy = (debut, fin, label_entite)
                donnees_entrainement.append((phrase_formatee, {"entities": [entite_spacy]}))
            elif log_callback:
                log_callback(f"Entité '{entite_texte}' ({label_entite}) non trouvée dans phrase pour placeholder '{placeholder}'.")
    return donnees_entrainement

def sauvegarder_donnees_json(donnees, chemin_fichier_sortie, log_callback=None):
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
        else: messagebox.showerror("Erreur Sauvegarde JSON", f"Erreur sauvegarde JSON: {e}")
        return False

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
                    
                    # S'assurer que début et fin sont des entiers
                    if not (isinstance(ent_item[0], int) and isinstance(ent_item[1], int)):
                         raise ValueError(f"Les indices de début/fin d'entité doivent être des entiers. Trouvé: {ent_item[:2]}")
                    if not isinstance(ent_item[2], str):
                        raise ValueError(f"Le label d'entité doit être une chaîne. Trouvé: {type(ent_item[2])}")
                        
                    entites_tuples.append(tuple(ent_item)) # S'assurer que c'est un tuple
            
            donnees_formatees.append((texte, {"entities": entites_tuples}))

        if not donnees_formatees and donnees_brutes: # Si on a lu quelque chose mais rien n'a été formaté
             messagebox.showwarning("Données Vides", f"Aucune donnée d'entraînement valide formatée depuis '{chemin_fichier_json}'. Vérifiez la structure.")
             return None
        elif not donnees_formatees: # Si le fichier était initialement vide
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
MODELES_SPACY_FR = {"Petit (sm)": "fr_core_news_sm", "Moyen (md)": "fr_core_news_md", "Grand (lg)": "fr_core_news_lg"}
modele_spacy_selectionne = None
chemin_donnees_entrainement_final = None 
chemin_modele_finetune_pour_test = None
mapping_pseudonymes_actuel = None
config_widgets_preparation = [] # Doit être global pour être peuplé par creer_section et lu par activer_widgets...

# --- Fonctions GUI principales ---
def valider_choix_modele():
    global modele_spacy_selectionne
    choix_utilisateur_label = choix_modele_var.get()
    modele_spacy_selectionne = MODELES_SPACY_FR.get(choix_utilisateur_label)
    if modele_spacy_selectionne:
        if verifier_et_telecharger_modele(modele_spacy_selectionne):
            messagebox.showinfo("Modèle Prêt", f"Modèle sélectionné : {modele_spacy_selectionne}\nPassez à la préparation des données.")
            
            # Activer seulement l'onglet de préparation et ses widgets
            notebook.tab(tab_preparation_donnees, state="normal")
            notebook.tab(tab_fine_tuning, state="disabled")
            notebook.tab(tab_test_modele, state="disabled")
            notebook.select(tab_preparation_donnees)
            
            activer_widgets_onglet_preparation(True)
            activer_widgets_onglet_fine_tuning(False) 
            activer_widgets_onglet_test(False) 
            bouton_valider_modele.config(state="disabled")
        else:
            # Garder tout désactivé si le modèle n'est pas prêt
            for i in range(notebook.index("end")):
                 notebook.tab(i, state="disabled")
            activer_widgets_onglet_preparation(False)
    else:
        messagebox.showwarning("Aucun Modèle", "Veuillez sélectionner un modèle.")

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
            finally:
                status_label_dl.destroy()
        return False

def log_message_preparation(message):
    text_log_preparation.config(state="normal")
    text_log_preparation.insert(tk.END, message + "\n")
    text_log_preparation.see(tk.END)
    text_log_preparation.config(state="disabled")
    fenetre.update_idletasks()

def choisir_fichier_pour_config(type_entite_label, type_de_fichier, var_tk_path): # Ajout du label pour un titre de dialogue plus clair
    chemin = filedialog.askopenfilename(title=f"Sélectionner fichier {type_de_fichier} pour {type_entite_label}", filetypes=(("Fichiers Texte", "*.txt"), ("Tous", "*.*")))
    if chemin:
        var_tk_path.set(chemin)

def lancer_generation_donnees_gui():
    global chemin_donnees_entrainement_final
    log_message_preparation("Démarrage de la génération des données...")
    text_log_preparation.config(state="normal"); text_log_preparation.delete(1.0, tk.END); text_log_preparation.config(state="disabled") # Clear log
    
    donnees_combinees = []
    
    configs_entites_gui = [
        {"label": "PER", "placeholder_var": var_placeholder_per, "entites_var": var_entites_per, "phrases_var": var_phrases_per, "actif_var": var_actif_per},
        {"label": "LOC", "placeholder_var": var_placeholder_loc, "entites_var": var_entites_loc, "phrases_var": var_phrases_loc, "actif_var": var_actif_loc},
        {"label": "ORG", "placeholder_var": var_placeholder_org, "entites_var": var_entites_org, "phrases_var": var_phrases_org, "actif_var": var_actif_org},
    ]

    au_moins_un_type_genere = False
    for config in configs_entites_gui:
        if config["actif_var"].get():
            label = config["label"]
            placeholder = config["placeholder_var"].get()
            chemin_entites = config["entites_var"].get()
            chemin_phrases = config["phrases_var"].get()

            if not all([placeholder, chemin_entites, chemin_phrases]):
                log_message_preparation(f"INFO: Infos manquantes pour le type {label}. Il sera ignoré.")
                continue
            
            log_message_preparation(f"\n--- Traitement du type d'entité : {label} ---")
            entites = lire_entites_depuis_fichier(chemin_entites, log_message_preparation)
            if entites is None or not entites: log_message_preparation(f"Pas d'entités chargées pour {label}."); continue
            phrases_modeles = lire_phrases_modeles_specifiques(chemin_phrases, placeholder, log_message_preparation)
            if phrases_modeles is None or not phrases_modeles: log_message_preparation(f"Pas de phrases modèles chargées pour {label}."); continue
            
            log_message_preparation(f"Génération pour {label}...")
            donnees_generees = generer_donnees_pour_type(entites, phrases_modeles, label, placeholder, log_message_preparation)
            
            if donnees_generees:
                log_message_preparation(f"{len(donnees_generees)} exemples générés pour {label}.")
                donnees_combinees.extend(donnees_generees)
                au_moins_un_type_genere = True
            else:
                log_message_preparation(f"Aucune donnée générée pour {label}.")

    if not donnees_combinees:
        messagebox.showwarning("Échec", "Aucune donnée d'entraînement n'a été générée au total. Vérifiez les configurations et les fichiers.")
        log_message_preparation("Échec : Aucune donnée combinée générée.")
        return

    chemin_sauvegarde_combine = filedialog.asksaveasfilename(
        title="Sauvegarder le fichier de données combinées",
        initialfile="donnees_entrainement_combinees.json",
        defaultextension=".json",
        filetypes=(("Fichiers JSON", "*.json"), ("Tous", "*.*"))
    )

    if chemin_sauvegarde_combine:
        if sauvegarder_donnees_json(donnees_combinees, chemin_sauvegarde_combine, log_message_preparation):
            chemin_donnees_entrainement_final = chemin_sauvegarde_combine
            messagebox.showinfo("Succès", f"Données combinées sauvegardées dans :\n{chemin_donnees_entrainement_final}\nPassez à l'onglet 'Fine-tuning Modèle'.")
            log_message_preparation(f"Terminé. Fichier combiné : {chemin_donnees_entrainement_final}")
            notebook.tab(tab_fine_tuning, state="normal")
            notebook.select(tab_fine_tuning)
            activer_widgets_onglet_fine_tuning(True)
            bouton_generer_donnees_gui.config(state="disabled") 
        else:
            messagebox.showerror("Erreur", "La sauvegarde du fichier combiné a échoué.")
            log_message_preparation("Échec de la sauvegarde du fichier combiné.")
    else:
        log_message_preparation("Sauvegarde du fichier combiné annulée.")

def lancer_fine_tuning_gui():
    global modele_spacy_selectionne, chemin_donnees_entrainement_final, chemin_modele_finetune_pour_test
    if not modele_spacy_selectionne: messagebox.showerror("Erreur", "Modèle SpaCy non sélectionné."); return
    if not chemin_donnees_entrainement_final or not os.path.exists(chemin_donnees_entrainement_final): messagebox.showerror("Erreur", "Fichier de données JSON introuvable."); return
    try:
        iterations = var_iterations.get(); dropout = var_dropout.get(); chemin_sauvegarde = var_chemin_sauvegarde_modele.get()
        if iterations <= 0: messagebox.showerror("Config Erreur", "Itérations > 0."); return
        if not (0.0 <= dropout <= 1.0): messagebox.showerror("Config Erreur", "Dropout entre 0.0 et 1.0."); return
        if not chemin_sauvegarde: messagebox.showerror("Config Erreur", "Spécifiez un dossier de sauvegarde."); return
    except tk.TclError: messagebox.showerror("Config Erreur", "Valeurs numériques valides pour itérations/dropout."); return
    
    TRAIN_DATA = charger_donnees_entrainement_json(chemin_donnees_entrainement_final)
    if not TRAIN_DATA: log_message_fine_tuning("Échec chargement données. Vérifiez JSON et sa structure."); return

    log_message_fine_tuning("Fine-tuning démarré...\n" + f"Modèle: {modele_spacy_selectionne}, Données: {os.path.basename(chemin_donnees_entrainement_final)} ({len(TRAIN_DATA)} ex.), It: {iterations}, Drop: {dropout}, Sauvegarde: {chemin_sauvegarde}\n")
    bouton_lancer_fine_tuning.config(state="disabled"); fenetre.update_idletasks()
    try:
        nlp = spacy.load(modele_spacy_selectionne)
        log_message_fine_tuning(f"Modèle '{modele_spacy_selectionne}' chargé.")
        ner = nlp.get_pipe("ner") if "ner" in nlp.pipe_names else nlp.add_pipe("ner", last=True)
        labels_dans_donnees = set()
        for _, annotations in TRAIN_DATA:
            for ent in annotations.get("entities", []): labels_dans_donnees.add(ent[2])
        for label in labels_dans_donnees: ner.add_label(label)
        log_message_fine_tuning(f"Labels pour NER : {labels_dans_donnees}")
        
        with nlp.select_pipes(disable=[p for p in nlp.pipe_names if p != "ner"]):
            optimizer = nlp.begin_training()
            for i in range(iterations):
                random.shuffle(TRAIN_DATA); pertes = {}
                for texte, annots in TRAIN_DATA:
                    try:
                        doc = nlp.make_doc(texte)
                        example = Example.from_dict(doc, annots)
                        nlp.update([example], sgd=optimizer, drop=dropout, losses=pertes)
                    except Exception as e_upd: log_message_fine_tuning(f"Err nlp.update (it {i+1}, ex. ignoré): {e_upd}"); continue
                log_message_fine_tuning(f"It {i+1}/{iterations} - Perte NER: {pertes.get('ner',0.0):.4f}")
                fenetre.update_idletasks()
        
        if not os.path.exists(chemin_sauvegarde): os.makedirs(chemin_sauvegarde)
        nlp.to_disk(chemin_sauvegarde)
        log_message_fine_tuning(f"\nModèle fine-tuné sauvegardé dans : '{chemin_sauvegarde}'")
        messagebox.showinfo("Fine-tuning Terminé", f"Modèle sauvegardé dans\n{chemin_sauvegarde}")
        chemin_modele_finetune_pour_test = chemin_sauvegarde 
        notebook.tab(tab_test_modele, state="normal") 
        notebook.select(tab_test_modele) 
        activer_widgets_onglet_test(True) 
    except Exception as e:
        log_message_fine_tuning(f"\nErreur majeure fine-tuning : {e}"); messagebox.showerror("Erreur Fine-tuning", f"Erreur : {e}")
    finally:
        bouton_lancer_fine_tuning.config(state="normal")
def choisir_dossier_sauvegarde_modele():
    chemin_dossier = filedialog.askdirectory(title="Sélectionner le dossier pour sauvegarder le modèle fine-tuné")
    if chemin_dossier:
        var_chemin_sauvegarde_modele.set(chemin_dossier)

def log_message_fine_tuning(message): 
    text_log_fine_tuning.config(state="normal"); text_log_fine_tuning.insert(tk.END, message + "\n"); text_log_fine_tuning.see(tk.END); text_log_fine_tuning.config(state="disabled"); fenetre.update_idletasks()

def activer_widgets_onglet_preparation(activer):
    etat = "normal" if activer else "disabled"
    # Parcourir config_widgets_preparation qui stocke les widgets de chaque section de type d'entité
    for type_config_widgets in config_widgets_preparation:
        type_config_widgets["checkbutton"].config(state=etat)
        type_config_widgets["entites_entry"].config(state="readonly" if activer else "disabled")
        type_config_widgets["entites_btn"].config(state=etat)
        type_config_widgets["phrases_entry"].config(state="readonly" if activer else "disabled")
        type_config_widgets["phrases_btn"].config(state=etat)
        type_config_widgets["placeholder_entry"].config(state=etat) 
    
    bouton_generer_donnees_gui.config(state=etat)
    
    # Gérer l'état du widget de log de préparation
    log_state = "normal" if activer else "disabled"
    # text_log_preparation.config(state=log_state) # Peut être problématique si on veut juste écrire dedans
    if activer:
        text_log_preparation.config(state="normal")
        text_log_preparation.delete(1.0, tk.END)
        text_log_preparation.config(state="disabled")
    else: # Si on désactive, effacer les variables et le log
        var_entites_per.set(""); var_phrases_per.set("")
        var_entites_loc.set(""); var_phrases_loc.set("")
        var_entites_org.set(""); var_phrases_org.set("")
        text_log_preparation.config(state="normal")
        text_log_preparation.delete(1.0, tk.END)
        text_log_preparation.config(state="disabled")
    fenetre.update_idletasks()


def activer_widgets_onglet_fine_tuning(activer): 
    etat = "normal" if activer else "disabled"; entry_iterations.config(state=etat); entry_dropout.config(state=etat); entry_chemin_sauvegarde_modele.config(state="readonly" if activer else "disabled"); bouton_choisir_dossier_modele.config(state=etat); bouton_lancer_fine_tuning.config(state=etat); text_log_fine_tuning.config(state="normal" if activer else "disabled")
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
    global chemin_modele_finetune_pour_test, mapping_pseudonymes_actuel
    path_fichier_test = var_chemin_fichier_test.get()
    if not chemin_modele_finetune_pour_test: messagebox.showerror("Erreur", "Aucun modèle fine-tuné disponible."); return
    if not path_fichier_test or not os.path.exists(path_fichier_test): messagebox.showerror("Erreur", "Sélectionnez un fichier texte de test valide."); return
    label_statut_test.config(text="Chargement du modèle fine-tuné..."); fenetre.update_idletasks()
    try:
        nlp_test = spacy.load(chemin_modele_finetune_pour_test)
        label_statut_test.config(text="Lecture du fichier de test..."); fenetre.update_idletasks()
        with open(path_fichier_test, 'r', encoding='utf-8') as f_test: texte_original_test = f_test.read()
        label_statut_test.config(text="Pseudonymisation en cours..."); fenetre.update_idletasks()
        texte_pseudo, mapping = pseudonymiser_texte_pour_gui(nlp_test, texte_original_test)
        if texte_pseudo is not None:
            text_resultat_pseudo.config(state="normal"); text_resultat_pseudo.delete(1.0, tk.END); text_resultat_pseudo.insert(tk.END, texte_pseudo); text_resultat_pseudo.config(state="disabled")
            label_statut_test.config(text="Pseudonymisation terminée.")
            mapping_pseudonymes_actuel = mapping 
            bouton_sauvegarder_texte_pseudo.config(state="normal")
            bouton_sauvegarder_mapping_pseudo.config(state="normal")
        else:
            label_statut_test.config(text="Erreur durant la pseudonymisation.")
    except Exception as e:
        messagebox.showerror("Erreur Test", f"Erreur lors du test : {e}"); label_statut_test.config(text=f"Erreur : {e}")

def sauvegarder_texte_resultat():
    texte_a_sauvegarder = text_resultat_pseudo.get(1.0, tk.END).strip()
    if not texte_a_sauvegarder: messagebox.showwarning("Rien à sauvegarder", "Aucun texte pseudonymisé."); return
    chemin_fichier_test_original = var_chemin_fichier_test.get()
    nom_initial_suggerere = "resultat_pseudonymise.txt"
    if chemin_fichier_test_original and os.path.exists(chemin_fichier_test_original):
        nom_base = os.path.basename(chemin_fichier_test_original); nom_sans_ext, _ = os.path.splitext(nom_base)
        nom_initial_suggerere = f"{nom_sans_ext}_pseudonymise.txt"
    chemin_fichier = filedialog.asksaveasfilename(title="Sauvegarder texte pseudonymisé", initialfile=nom_initial_suggerere, defaultextension=".txt", filetypes=(("Fichiers Texte", "*.txt"),("Tous", "*.*")))
    if chemin_fichier:
        try:
            with open(chemin_fichier, 'w', encoding='utf-8') as f: f.write(texte_a_sauvegarder)
            messagebox.showinfo("Succès", f"Texte sauvegardé : {chemin_fichier}")
        except Exception as e: messagebox.showerror("Erreur Sauvegarde", f"Impossible de sauvegarder : {e}")

def sauvegarder_mapping_resultat():
    global mapping_pseudonymes_actuel
    if not mapping_pseudonymes_actuel: messagebox.showwarning("Rien à sauvegarder", "Aucune table de correspondance."); return
    chemin_fichier_test_original = var_chemin_fichier_test.get()
    nom_initial_suggerere = "mapping_pseudonymes.json"
    if chemin_fichier_test_original and os.path.exists(chemin_fichier_test_original):
        nom_base = os.path.basename(chemin_fichier_test_original); nom_sans_ext, _ = os.path.splitext(nom_base)
        nom_initial_suggerere = f"{nom_sans_ext}_mapping.json"
    chemin_fichier = filedialog.asksaveasfilename(title="Sauvegarder la table de correspondance", initialfile=nom_initial_suggerere, defaultextension=".json", filetypes=(("Fichiers JSON", "*.json"),("Tous", "*.*")))
    if chemin_fichier:
        try:
            with open(chemin_fichier, 'w', encoding='utf-8') as f: json.dump(mapping_pseudonymes_actuel, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Succès", f"Table de correspondance sauvegardée : {chemin_fichier}")
        except Exception as e: messagebox.showerror("Erreur Sauvegarde", f"Impossible de sauvegarder le JSON : {e}")

def activer_widgets_onglet_test(activer):
    etat = "normal" if activer else "disabled"
    bouton_choisir_fichier_test.config(state=etat)
    entry_chemin_fichier_test.config(state="readonly" if activer else "disabled")
    bouton_lancer_test.config(state=etat)
    #text_resultat_pseudo.config(state="disabled") # Déjà géré comme ça
    if not activer:
        var_chemin_fichier_test.set(""); label_statut_test.config(text="")
        text_resultat_pseudo.config(state="normal"); text_resultat_pseudo.delete(1.0, tk.END); text_resultat_pseudo.config(state="disabled")
        bouton_sauvegarder_texte_pseudo.config(state="disabled")
        bouton_sauvegarder_mapping_pseudo.config(state="disabled")
    fenetre.update_idletasks()


# --- Création de la fenêtre principale et du Canvas pour le défilement ---
fenetre = tk.Tk()
fenetre.title("Assistant de Fine-tuning et Test LLM SpaCy")
fenetre.geometry("800x650") 

main_canvas = tk.Canvas(fenetre)
v_scrollbar = ttk.Scrollbar(fenetre, orient="vertical", command=main_canvas.yview)
main_canvas.configure(yscrollcommand=v_scrollbar.set)
v_scrollbar.pack(side=tk.RIGHT, fill="y")
main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
content_frame = ttk.Frame(main_canvas) 
main_canvas.create_window((0, 0), window=content_frame, anchor="nw")

def on_content_frame_configure(event): main_canvas.configure(scrollregion=main_canvas.bbox("all"))
content_frame.bind("<Configure>", on_content_frame_configure)

def _on_mousewheel(event):
    scroll_speed_multiplier = 2 
    if event.num == 4: main_canvas.yview_scroll(-1 * scroll_speed_multiplier, "units")
    elif event.num == 5: main_canvas.yview_scroll(1 * scroll_speed_multiplier, "units")
    else: main_canvas.yview_scroll(int(-1*(event.delta/120) * scroll_speed_multiplier), "units")

for widget_to_bind in [main_canvas, content_frame, fenetre]:
    widget_to_bind.bind("<MouseWheel>", _on_mousewheel)
    widget_to_bind.bind("<Button-4>", _on_mousewheel)  
    widget_to_bind.bind("<Button-5>", _on_mousewheel)  

# --- Cadre 1: Choix du Modèle (Parent: content_frame) ---
cadre_choix_modele = ttk.LabelFrame(content_frame, text="1. Choix du Modèle SpaCy de Base", padding=(10, 10))
cadre_choix_modele.pack(padx=10, pady=10, fill="x", anchor="n")
label_instruction_modele = ttk.Label(cadre_choix_modele, text="Sélectionnez le modèle français à fine-tuner :")
label_instruction_modele.pack(pady=(0,10), anchor="w")
choix_modele_var = tk.StringVar(fenetre) 
liste_labels_modeles = list(MODELES_SPACY_FR.keys())
menu_deroulant_modeles = ttk.Combobox(cadre_choix_modele, textvariable=choix_modele_var, values=liste_labels_modeles, state="readonly", width=30)
if liste_labels_modeles: menu_deroulant_modeles.current(1) 
menu_deroulant_modeles.pack(pady=5)
bouton_valider_modele = ttk.Button(cadre_choix_modele, text="Valider Modèle et Continuer", command=valider_choix_modele)
bouton_valider_modele.pack(pady=10)

# --- Création du Notebook (Onglets) ---
notebook = ttk.Notebook(content_frame) 
notebook.pack(padx=10, pady=10, fill="both", expand=True)

# --- Onglet 1: Préparation des Données Multi-types ---
tab_preparation_donnees = ttk.Frame(notebook, padding=(10,10))
notebook.add(tab_preparation_donnees, text="Préparation Données") # Titre plus court

var_actif_per = tk.BooleanVar(value=True); var_entites_per = tk.StringVar(); var_phrases_per = tk.StringVar(); var_placeholder_per = tk.StringVar(value="{NOM}")
var_actif_loc = tk.BooleanVar(value=False); var_entites_loc = tk.StringVar(); var_phrases_loc = tk.StringVar(); var_placeholder_loc = tk.StringVar(value="{LOC}")
var_actif_org = tk.BooleanVar(value=False); var_entites_org = tk.StringVar(); var_phrases_org = tk.StringVar(); var_placeholder_org = tk.StringVar(value="{ORG}")

def creer_section_type_entite(parent, label_type, var_actif_chk, var_ent_path, var_phr_path, var_plh):
    frame_type = ttk.LabelFrame(parent, text=f"Configuration pour : {label_type}", padding=5)
    frame_type.pack(fill="x", expand=True, pady=3, padx=3)
    
    chk_button = ttk.Checkbutton(frame_type, text="Activer ce type", variable=var_actif_chk)
    chk_button.pack(anchor="w")
    
    f_ent = ttk.Frame(frame_type); f_ent.pack(fill="x", pady=1)
    ttk.Label(f_ent, text="Fichier Entités:", width=15).pack(side=tk.LEFT)
    entry_ent = ttk.Entry(f_ent, textvariable=var_ent_path, width=35, state="readonly")
    entry_ent.pack(side=tk.LEFT, expand=True, fill="x", padx=1)
    btn_ent = ttk.Button(f_ent, text="...", width=3, command=lambda v=var_ent_path: choisir_fichier_pour_config(label_type, "Entités", v))
    btn_ent.pack(side=tk.LEFT)
    
    f_phr = ttk.Frame(frame_type); f_phr.pack(fill="x", pady=1)
    ttk.Label(f_phr, text="Fichier Phrases:", width=15).pack(side=tk.LEFT)
    entry_phr = ttk.Entry(f_phr, textvariable=var_phr_path, width=35, state="readonly")
    entry_phr.pack(side=tk.LEFT, expand=True, fill="x", padx=1)
    btn_phr = ttk.Button(f_phr, text="...", width=3, command=lambda v=var_phr_path: choisir_fichier_pour_config(label_type, "Phrases", v))
    btn_phr.pack(side=tk.LEFT)

    f_plh = ttk.Frame(frame_type); f_plh.pack(fill="x", pady=1)
    ttk.Label(f_plh, text="Placeholder:", width=15).pack(side=tk.LEFT)
    entry_plh = ttk.Entry(f_plh, textvariable=var_plh, width=10)
    entry_plh.pack(side=tk.LEFT)

    config_widgets_preparation.append({
        "checkbutton": chk_button, 
        "entites_entry": entry_ent, "entites_btn": btn_ent,
        "phrases_entry": entry_phr, "phrases_btn": btn_phr,
        "placeholder_entry": entry_plh
    })

creer_section_type_entite(tab_preparation_donnees, "PER", var_actif_per, var_entites_per, var_phrases_per, var_placeholder_per)
creer_section_type_entite(tab_preparation_donnees, "LOC", var_actif_loc, var_entites_loc, var_phrases_loc, var_placeholder_loc)
creer_section_type_entite(tab_preparation_donnees, "ORG", var_actif_org, var_entites_org, var_phrases_org, var_placeholder_org)

bouton_generer_donnees_gui = ttk.Button(tab_preparation_donnees, text="Générer et Combiner les Données", command=lancer_generation_donnees_gui)
bouton_generer_donnees_gui.pack(pady=10)
text_log_preparation = scrolledtext.ScrolledText(tab_preparation_donnees, height=5, width=80, state="disabled", wrap=tk.WORD)
text_log_preparation.pack(pady=5, fill="x", expand=False)

# --- Onglet 2: Fine-tuning Modèle ---
tab_fine_tuning = ttk.Frame(notebook, padding=(10,10))
notebook.add(tab_fine_tuning, text="Fine-tuning Modèle", state="disabled") 
var_iterations = tk.IntVar(value=10); var_dropout = tk.DoubleVar(value=0.3); var_chemin_sauvegarde_modele = tk.StringVar()
frame_iter = ttk.Frame(tab_fine_tuning); frame_iter.pack(fill="x", pady=2)
ttk.Label(frame_iter, text="Nombre d'itérations:", width=25).pack(side=tk.LEFT, padx=(0,5))
entry_iterations = ttk.Spinbox(frame_iter, from_=1, to=1000, textvariable=var_iterations, width=10); entry_iterations.pack(side=tk.LEFT)
frame_drop = ttk.Frame(tab_fine_tuning); frame_drop.pack(fill="x", pady=2)
ttk.Label(frame_drop, text="Taux de Dropout:", width=25).pack(side=tk.LEFT, padx=(0,5)) # Texte simplifié
entry_dropout = ttk.Spinbox(frame_drop, from_=0.0, to=1.0, increment=0.05, textvariable=var_dropout, width=10, format="%.2f"); entry_dropout.pack(side=tk.LEFT)
frame_sauvegarde_modele = ttk.Frame(tab_fine_tuning); frame_sauvegarde_modele.pack(fill="x", pady=2)
ttk.Label(frame_sauvegarde_modele, text="Dossier sauvegarde modèle:", width=25).pack(side=tk.LEFT, padx=(0,5)) # Texte simplifié
entry_chemin_sauvegarde_modele = ttk.Entry(frame_sauvegarde_modele, textvariable=var_chemin_sauvegarde_modele, state="readonly", width=40); entry_chemin_sauvegarde_modele.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_dossier_modele = ttk.Button(frame_sauvegarde_modele, text="...", width=3, command=choisir_dossier_sauvegarde_modele); bouton_choisir_dossier_modele.pack(side=tk.LEFT)
bouton_lancer_fine_tuning = ttk.Button(tab_fine_tuning, text="Lancer le Fine-tuning", command=lancer_fine_tuning_gui); bouton_lancer_fine_tuning.pack(pady=10)
ttk.Label(tab_fine_tuning, text="Log du Fine-tuning:").pack(anchor="w", pady=(5,0))
text_log_fine_tuning = scrolledtext.ScrolledText(tab_fine_tuning, height=8, width=80, state="disabled", wrap=tk.WORD); text_log_fine_tuning.pack(pady=5, fill="x", expand=False)

# --- Onglet 3: Test Modèle ---
tab_test_modele = ttk.Frame(notebook, padding=(10,10))
notebook.add(tab_test_modele, text="Test Modèle", state="disabled") 
var_chemin_fichier_test = tk.StringVar()
frame_fichier_test = ttk.Frame(tab_test_modele); frame_fichier_test.pack(fill="x", pady=5)
ttk.Label(frame_fichier_test, text="Fichier Texte de Test:", width=25).pack(side=tk.LEFT, padx=(0,5)) # Texte simplifié
entry_chemin_fichier_test = ttk.Entry(frame_fichier_test, textvariable=var_chemin_fichier_test, state="readonly", width=40); entry_chemin_fichier_test.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_fichier_test = ttk.Button(frame_fichier_test, text="...", width=3, command=choisir_fichier_test_txt); bouton_choisir_fichier_test.pack(side=tk.LEFT)
bouton_lancer_test = ttk.Button(tab_test_modele, text="Lancer Test Pseudonymisation", command=lancer_test_pseudonymisation_gui); bouton_lancer_test.pack(pady=10)
label_statut_test = ttk.Label(tab_test_modele, text=""); label_statut_test.pack(pady=2)
ttk.Label(tab_test_modele, text="Résultat Pseudonymisation:").pack(anchor="w", pady=(5,0))
text_resultat_pseudo = scrolledtext.ScrolledText(tab_test_modele, height=10, width=80, state="disabled", wrap=tk.WORD); text_resultat_pseudo.pack(pady=5, fill="both", expand=True)
frame_sauvegarde_test = ttk.Frame(tab_test_modele); frame_sauvegarde_test.pack(pady=5)
bouton_sauvegarder_texte_pseudo = ttk.Button(frame_sauvegarde_test, text="Sauvegarder Texte", command=sauvegarder_texte_resultat, state="disabled"); bouton_sauvegarder_texte_pseudo.pack(side=tk.LEFT, padx=5) # Texte simplifié
bouton_sauvegarder_mapping_pseudo = ttk.Button(frame_sauvegarde_test, text="Sauvegarder Mapping", command=sauvegarder_mapping_resultat, state="disabled"); bouton_sauvegarder_mapping_pseudo.pack(side=tk.LEFT, padx=5)

# Initialisation des états
activer_widgets_onglet_preparation(False) 
activer_widgets_onglet_fine_tuning(False)
activer_widgets_onglet_test(False)
for i in range(notebook.index("end")): notebook.tab(i, state="disabled")

fenetre.mainloop()