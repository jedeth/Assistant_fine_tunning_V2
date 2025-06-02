import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import spacy
from spacy.training.example import Example
import os
import re 
import json
import random
import logique_preparation_donnees as lpd
import logique_fine_tuning as lft 
import logique_pseudonymisation as lp
import utils # Si utils.py


# --- Logique de Pseudonymisation ---
def pseudonymiser_texte_pour_gui(nlp_model, texte_original):
    if not nlp_model: messagebox.showerror("Erreur Modèle", "Modèle SpaCy non chargé."); return None, None
    doc = nlp_model(texte_original); correspondances = {}; compteur = 1; ent_remplacer = []
    for entite in doc.ents:
        if entite.label_ == "PER": 
            nom_o = entite.text.strip() 
            if nom_o not in correspondances: pseudo = f"[PERSONNE_{compteur}]"; correspondances[nom_o] = pseudo; compteur += 1
            else: pseudo = correspondances[nom_o]
            ent_remplacer.append((entite.start_char, entite.end_char, pseudo))
    ent_remplacer.sort(key=lambda x: x[0], reverse=True)
    parts = []; last_idx = len(texte_original)
    for start, end, pseudo in ent_remplacer:
        if end < last_idx: parts.append(texte_original[end:last_idx])
        parts.append(pseudo); last_idx = start
    parts.append(texte_original[0:last_idx])
    return "".join(reversed(parts)), correspondances

def lancer_fine_tuning_gui():
    global modele_spacy_selectionne_pour_ft, chemin_donnees_entrainement_final, chemin_modele_a_tester
    
    if not modele_spacy_selectionne_pour_ft:
        messagebox.showerror("Erreur", "Aucun modèle SpaCy de base n'a été sélectionné pour le fine-tuning.")
        return
    if not chemin_donnees_entrainement_final or not os.path.exists(chemin_donnees_entrainement_final):
        messagebox.showerror("Erreur", "Le fichier de données d'entraînement JSON n'a pas été généré ou est introuvable.")
        return

    try:
        iterations = var_iterations.get()
        dropout = var_dropout.get()
        chemin_sauvegarde = var_chemin_sauvegarde_modele.get()

        if iterations <= 0:
            messagebox.showerror("Erreur de Configuration", "Le nombre d'itérations doit être supérieur à 0.")
            return
        if not (0.0 <= dropout <= 1.0): # Dropout est un float
            messagebox.showerror("Erreur de Configuration", "Le taux de dropout doit être entre 0.0 et 1.0.")
            return
        if not chemin_sauvegarde:
            messagebox.showerror("Erreur de Configuration", "Veuillez spécifier un dossier pour sauvegarder le modèle fine-tuné.")
            return
    except tk.TclError:
        messagebox.showerror("Erreur de Configuration", "Veuillez entrer des valeurs numériques valides pour les itérations et le dropout.")
        return
    
    log_message_fine_tuning(f"Préparation du fine-tuning...\nModèle: {modele_spacy_selectionne_pour_ft}, Données: {os.path.basename(chemin_donnees_entrainement_final)}, It: {iterations}, Drop: {dropout}, Sauvegarde: {chemin_sauvegarde}\n")
    bouton_lancer_fine_tuning.config(state="disabled")
    fenetre.update_idletasks() # Mettre à jour l'interface avant la tâche potentiellement longue

    # Appel de la fonction de logique externe  v
    # Le callback log_message_fine_tuning mettra à jour text_log_fine_tuning dans le GUI
    succes, chemin_modele_sauvegarde = lft.executer_fine_tuning(
        nom_modele_base=modele_spacy_selectionne_pour_ft,
        chemin_donnees_entrainement=chemin_donnees_entrainement_final,
        chemin_sauvegarde_modele=chemin_sauvegarde,
        iterations=iterations,
        dropout_rate=dropout,
        log_callback=log_message_fine_tuning # Passer la fonction de log de l'interface
    )

    bouton_lancer_fine_tuning.config(state="normal") # Réactiver le bouton après l'exécution

    if succes and chemin_modele_sauvegarde:
        messagebox.showinfo("Fine-tuning Terminé", f"Le modèle a été fine-tuné et sauvegardé dans\n{chemin_modele_sauvegarde}")
        chemin_modele_a_tester = chemin_modele_sauvegarde # Mettre à jour pour l'étape de test
        
        # Activer et sélectionner l'onglet Test
        try:
            # S'assurer que l'onglet test existe avant de le configurer (au cas où il aurait été 'forgotten')
            notebook.index(tab_test_modele) 
        except tk.TclError: # Si l'onglet n'est pas connu (a été "forgotten" dans un autre scénario)
            notebook.add(tab_test_modele, text="Étape 4: Test Modèle Fine-tuné") 
        
        notebook.tab(tab_test_modele, state="normal") 
        notebook.select(tab_test_modele) 
        activer_widgets_onglet_test(True) 
    else:
        messagebox.showerror("Erreur Fine-tuning", "Le fine-tuning a échoué. Consultez les logs pour plus de détails.")

# Variables globales
MODELES_SPACY_FR_BASE = {"Petit (sm)": "fr_core_news_sm", "Moyen (md)": "fr_core_news_md", "Grand (lg)": "fr_core_news_lg"}
OPTION_MODELE_EXISTANT = "Modèle fine-tuné existant..." 
modele_spacy_selectionne_pour_ft = None 
chemin_donnees_entrainement_final = None 
chemin_modele_a_tester = None 
mapping_pseudonymes_actuel = None
config_widgets_preparation = [] 

# --- Fonctions GUI ---
def on_model_type_changed(event=None):
    selection_label = choix_modele_var.get()
    if selection_label == OPTION_MODELE_EXISTANT:
        if not frame_custom_model_path.winfo_ismapped(): 
            frame_custom_model_path.pack(fill="x", pady=(5,0), before=bouton_valider_modele) # Nom du bouton corrigé
        bouton_valider_modele.config(text="Valider Modèle Existant et Tester")
    else:
        frame_custom_model_path.pack_forget()
        bouton_valider_modele.config(text="Valider Modèle de Base et Continuer")

def valider_choix_modele():
    global modele_spacy_selectionne_pour_ft, chemin_modele_a_tester
    
    choix_utilisateur_label = choix_modele_var.get()
    valeur_modele_base = MODELES_SPACY_FR_BASE.get(choix_utilisateur_label)

    # Cacher et nettoyer le notebook avant de reconfigurer
    if notebook.winfo_ismapped():
        notebook.pack_forget()
    for tab_id in reversed(notebook.tabs()): 
        notebook.forget(tab_id)
    
    activer_widgets_onglet_preparation(False)
    activer_widgets_onglet_fine_tuning(False)
    activer_widgets_onglet_test(False)

    if choix_utilisateur_label == OPTION_MODELE_EXISTANT:
        path_custom = var_custom_model_path.get()
        if not path_custom or not os.path.isdir(path_custom): 
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier de modèle fine-tuné valide.")
            return
        try:
            # Essayer de charger pour valider (ne pas garder en mémoire)
            spacy.load(path_custom) 
            messagebox.showinfo("Modèle Valide", f"Modèle existant '{os.path.basename(path_custom)}' validé.\nL'onglet de Test est activé.")
            chemin_modele_a_tester = path_custom
            modele_spacy_selectionne_pour_ft = None 
            
            notebook.add(tab_test_modele, text="Étape 2: Tester Modèle") 
            if not notebook.winfo_ismapped():
                 notebook.pack(padx=10, pady=10, fill="both", expand=True, anchor="n")
            
            notebook.select(tab_test_modele)
            activer_widgets_onglet_test(True)
            bouton_valider_modele.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Erreur Chargement Modèle", f"Impossible de charger le modèle depuis '{path_custom}'.\nErreur: {e}")
            chemin_modele_a_tester = None
            return
            
    elif valeur_modele_base: 
        modele_spacy_selectionne_pour_ft = valeur_modele_base
        chemin_modele_a_tester = None
        
        # Utilisation de la fonction utilitaire de utils.py
        resultat_verification = utils.verifier_existence_modele_spacy(modele_spacy_selectionne_pour_ft)
        
        modele_pret = False
        if resultat_verification["status"] == "existe":
            print(f"Le modèle '{modele_spacy_selectionne_pour_ft}' est déjà disponible.")
            modele_pret = True
        elif resultat_verification["status"] == "non_trouve":
            reponse_dl = messagebox.askyesno("Modèle Non Trouvé", 
                                          f"Le modèle '{modele_spacy_selectionne_pour_ft}' n'est pas trouvé. Voulez-vous le télécharger ?")
            if reponse_dl:
                status_label_dl = ttk.Label(cadre_choix_modele, text=f"Téléchargement de {modele_spacy_selectionne_pour_ft}...")
                status_label_dl.pack(pady=2)
                fenetre.update_idletasks()
                
                # Utilisation de la fonction utilitaire de utils.py pour le téléchargement
                resultat_telechargement = utils.telecharger_modele_spacy(modele_spacy_selectionne_pour_ft, print) # print comme log_callback simple
                
                if status_label_dl.winfo_exists(): status_label_dl.destroy()

                if resultat_telechargement["status"] == "telechargement_reussi":
                    messagebox.showinfo("Téléchargement Réussi", f"Modèle '{modele_spacy_selectionne_pour_ft}' téléchargé.")
                    modele_pret = True
                else: # echec ou python_non_trouve
                    messagebox.showerror("Erreur Téléchargement", f"Impossible de télécharger '{modele_spacy_selectionne_pour_ft}'.\nErreur: {resultat_telechargement.get('error', 'Inconnue')}")
            else: # L'utilisateur a refusé le téléchargement
                 messagebox.showinfo("Téléchargement Annulé", "Le téléchargement a été annulé.")
        else: # Erreur de vérification
            messagebox.showerror("Erreur Vérification Modèle", f"Erreur lors de la vérification du modèle: {resultat_verification.get('error', 'Inconnue')}")

        if modele_pret:
            messagebox.showinfo("Modèle Prêt", f"Modèle de base : {modele_spacy_selectionne_pour_ft}\nPassez à la préparation des données.")
            notebook.add(tab_preparation_donnees, text="Étape 2: Préparation Données")
            notebook.add(tab_fine_tuning, text="Étape 3: Fine-tuning Modèle")
            notebook.add(tab_test_modele, text="Étape 4: Test Modèle Fine-tuné")
            if not notebook.winfo_ismapped():
                 notebook.pack(padx=10, pady=10, fill="both", expand=True, anchor="n")
            
            notebook.tab(tab_preparation_donnees, state="normal")
            notebook.tab(tab_fine_tuning, state="disabled")
            notebook.tab(tab_test_modele, state="disabled")
            
            notebook.select(tab_preparation_donnees)
            activer_widgets_onglet_preparation(True)
            bouton_valider_modele.config(state="disabled")
        else:
            if notebook.winfo_ismapped(): notebook.pack_forget()
            
    else: 
        messagebox.showwarning("Sélection Invalide", "Veuillez sélectionner une option de modèle valide.")
        if notebook.winfo_ismapped(): notebook.pack_forget()

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
                messagebox.showinfo("Téléchargement Réussi", f"'{nom_modele}' téléchargé.")
                return True
            except Exception as e: 
                messagebox.showerror("Erreur Téléchargement", f"Impossible de télécharger '{nom_modele}'.\n{e}\nVérifiez la console ou essayez manuellement.")
                return False
            finally:
                if 'status_label_dl' in locals() and status_label_dl.winfo_exists(): status_label_dl.destroy()
        return False

def log_message_preparation(message):
    # ... (Fonction inchangée)
    text_log_preparation.config(state="normal"); text_log_preparation.insert(tk.END, message + "\n"); text_log_preparation.see(tk.END); text_log_preparation.config(state="disabled"); fenetre.update_idletasks()

def choisir_fichier_pour_config(type_entite_label, type_de_fichier, var_tk_path):
    # ... (Fonction inchangée)
    chemin = filedialog.askopenfilename(title=f"Sélectionner fichier {type_de_fichier} pour {type_entite_label}", filetypes=(("Fichiers Texte", "*.txt"), ("Tous", "*.*")))
    if chemin: var_tk_path.set(chemin)

def lancer_generation_donnees_gui():
    global chemin_donnees_entrainement_final
    log_message_preparation("Démarrage de la génération des données...")
    text_log_preparation.config(state="normal"); text_log_preparation.delete(1.0, tk.END); text_log_preparation.config(state="disabled") 
    
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
            # Utilisation des fonctions importées avec l'alias lpd
            entites = lpd.lire_entites_depuis_fichier(chemin_entites, log_message_preparation)
            if entites is None or not entites: 
                log_message_preparation(f"Pas d'entités chargées pour {label} depuis '{chemin_entites}'.")
                continue
            
            phrases_modeles = lpd.lire_phrases_modeles_specifiques(chemin_phrases, placeholder, log_message_preparation)
            if phrases_modeles is None or not phrases_modeles: 
                log_message_preparation(f"Pas de phrases modèles chargées pour {label} depuis '{chemin_phrases}'.")
                continue
            
            log_message_preparation(f"Génération pour {label}...")
            donnees_generees = lpd.generer_donnees_pour_type(entites, phrases_modeles, label, placeholder, log_message_preparation)
            
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
        # Utilisation de la fonction importée avec l'alias lpd
        if lpd.sauvegarder_donnees_json(donnees_combinees, chemin_sauvegarde_combine, log_message_preparation):
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
    # ... (Fonction inchangée)
    global modele_spacy_selectionne_pour_ft, chemin_donnees_entrainement_final, chemin_modele_a_tester 
    if not modele_spacy_selectionne_pour_ft: messagebox.showerror("Erreur", "Modèle SpaCy de base non sélectionné pour le fine-tuning."); return
    if not chemin_donnees_entrainement_final or not os.path.exists(chemin_donnees_entrainement_final): messagebox.showerror("Erreur", "Fichier de données JSON introuvable."); return
    try:
        iterations = var_iterations.get(); dropout = var_dropout.get(); chemin_sauvegarde = var_chemin_sauvegarde_modele.get()
        if iterations <= 0: messagebox.showerror("Config Erreur", "Itérations > 0."); return
        if not (0.0 <= dropout <= 1.0): messagebox.showerror("Config Erreur", "Dropout entre 0.0 et 1.0."); return
        if not chemin_sauvegarde: messagebox.showerror("Config Erreur", "Spécifiez un dossier de sauvegarde."); return
    except tk.TclError: messagebox.showerror("Config Erreur", "Valeurs numériques valides pour itérations/dropout."); return
    TRAIN_DATA = lft.charger_donnees_entrainement_json(chemin_donnees_entrainement_final, log_message_fine_tuning)
    if not TRAIN_DATA: log_message_fine_tuning("Échec chargement données. Vérifiez JSON et sa structure."); return
    log_message_fine_tuning("Fine-tuning démarré...\n" + f"Modèle: {modele_spacy_selectionne_pour_ft}, Données: {os.path.basename(chemin_donnees_entrainement_final)} ({len(TRAIN_DATA)} ex.), It: {iterations}, Drop: {dropout}, Sauvegarde: {chemin_sauvegarde}\n")
    bouton_lancer_fine_tuning.config(state="disabled"); fenetre.update_idletasks()
    try:
        nlp = spacy.load(modele_spacy_selectionne_pour_ft) 
        log_message_fine_tuning(f"Modèle '{modele_spacy_selectionne_pour_ft}' chargé.")
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
        chemin_modele_a_tester = chemin_sauvegarde 
        
        # S'assurer que l'onglet de test est présent avant de le configurer
        tab_test_present = False
        for i in range(len(notebook.tabs())):
            if notebook.nametowidget(notebook.tabs()[i]) == tab_test_modele:
                tab_test_present = True
                break
        if not tab_test_present: # S'il avait été 'forgotten' et non ré-ajouté
            notebook.add(tab_test_modele, text="Étape 4: Test Modèle Fine-tuné") # ou le nom approprié

        notebook.tab(tab_test_modele, state="normal") 
        notebook.select(tab_test_modele) 
        activer_widgets_onglet_test(True) 
    except Exception as e:
        log_message_fine_tuning(f"\nErreur majeure fine-tuning : {e}"); messagebox.showerror("Erreur Fine-tuning", f"Erreur : {e}")
    finally:
        bouton_lancer_fine_tuning.config(state="normal")

def log_message_fine_tuning(message): 
    # ... (Fonction inchangée)
    text_log_fine_tuning.config(state="normal"); text_log_fine_tuning.insert(tk.END, message + "\n"); text_log_fine_tuning.see(tk.END); text_log_fine_tuning.config(state="disabled"); fenetre.update_idletasks()

def activer_widgets_onglet_preparation(activer):
    # ... (Fonction inchangée)
    etat = "normal" if activer else "disabled"
    for type_config_widgets in config_widgets_preparation:
        type_config_widgets["checkbutton"].config(state=etat)
        type_config_widgets["entites_entry"].config(state="readonly" if activer else "disabled")
        type_config_widgets["entites_btn"].config(state=etat)
        type_config_widgets["phrases_entry"].config(state="readonly" if activer else "disabled")
        type_config_widgets["phrases_btn"].config(state=etat)
        type_config_widgets["placeholder_entry"].config(state=etat) 
    bouton_generer_donnees_gui.config(state=etat)
    text_log_preparation.config(state="normal")
    if activer: text_log_preparation.delete(1.0, tk.END)
    text_log_preparation.config(state="disabled") 
    if not activer:
        var_entites_per.set(""); var_phrases_per.set(""); var_placeholder_per.set("{NOM}")
        var_entites_loc.set(""); var_phrases_loc.set(""); var_placeholder_loc.set("{LOC}")
        var_entites_org.set(""); var_phrases_org.set(""); var_placeholder_org.set("{ORG}")
    fenetre.update_idletasks()

def activer_widgets_onglet_fine_tuning(activer): 
    # ... (Fonction inchangée)
    etat = "normal" if activer else "disabled"; entry_iterations.config(state=etat); entry_dropout.config(state=etat); entry_chemin_sauvegarde_modele.config(state="readonly" if activer else "disabled"); bouton_choisir_dossier_modele.config(state=etat); bouton_lancer_fine_tuning.config(state=etat)
    text_log_fine_tuning.config(state="normal")
    if not activer:
        var_iterations.set(10); var_dropout.set(0.3); var_chemin_sauvegarde_modele.set("")
        text_log_fine_tuning.delete(1.0, tk.END)
    text_log_fine_tuning.config(state="disabled")
    fenetre.update_idletasks()

def choisir_dossier_sauvegarde_modele(): # Définition de la fonction
    chemin_dossier = filedialog.askdirectory(title="Sélectionner le dossier pour sauvegarder le modèle fine-tuné")
    if chemin_dossier:
        var_chemin_sauvegarde_modele.set(chemin_dossier)

def choisir_fichier_test_txt():
    # ... (Fonction inchangée)
    chemin_fichier = filedialog.askopenfilename(title="Sélectionner fichier texte pour test", filetypes=(("Fichiers Texte", "*.txt"), ("Tous", "*.*")))
    if chemin_fichier: var_chemin_fichier_test.set(chemin_fichier); label_statut_test.config(text=f"Fichier: {os.path.basename(chemin_fichier)}")

def lancer_test_pseudonymisation_gui():
    global chemin_modele_a_tester, mapping_pseudonymes_actuel 
    
    path_fichier_test = var_chemin_fichier_test.get()

    if not chemin_modele_a_tester: # Le chemin est défini soit après fine-tuning, soit par sélection d'un modèle existant
        messagebox.showerror("Erreur", "Aucun modèle n'est prêt pour le test. Entraînez ou sélectionnez un modèle existant.")
        return
    if not path_fichier_test or not os.path.exists(path_fichier_test): 
        messagebox.showerror("Erreur", "Veuillez sélectionner un fichier texte de test valide.")
        return

    label_statut_test.config(text="Chargement du modèle...")
    fenetre.update_idletasks()
    
    try:
        nlp_test = spacy.load(chemin_modele_a_tester) # Le chemin_modele_a_tester est déjà validé
        label_statut_test.config(text="Lecture du fichier de test...")
        fenetre.update_idletasks()
        
        with open(path_fichier_test, 'r', encoding='utf-8') as f_test:
            texte_original_test = f_test.read()
        
        label_statut_test.config(text="Pseudonymisation en cours...")
        fenetre.update_idletasks()
        
        # Appel corrigé ici :
        texte_pseudo, mapping = lp.pseudonymiser_texte(nlp_test, texte_original_test)
        
        if texte_pseudo is not None: # La fonction logique retourne (None, None) en cas d'erreur interne
            text_resultat_pseudo.config(state="normal")
            text_resultat_pseudo.delete(1.0, tk.END)
            text_resultat_pseudo.insert(tk.END, texte_pseudo)
            text_resultat_pseudo.config(state="disabled")
            label_statut_test.config(text="Pseudonymisation terminée.")
            mapping_pseudonymes_actuel = mapping 
            bouton_sauvegarder_texte_pseudo.config(state="normal")
            bouton_sauvegarder_mapping_pseudo.config(state="normal")
        else:
            # Une erreur s'est produite dans la fonction logique (par exemple, modèle non fourni, bien que vérifié avant)
            # Ou la fonction a retourné None explicitement pour une autre raison
            label_statut_test.config(text="Erreur durant la pseudonymisation. Vérifiez la console.")
            # Le message d'erreur spécifique aurait été printé par la fonction logique si nlp_model était None
            
    except Exception as e:
        messagebox.showerror("Erreur Test", f"Une erreur est survenue lors du test : {e}")
        label_statut_test.config(text=f"Erreur : {e}")
        # Logguer aussi la trace complète dans la console pour débogage
        import traceback
        print("Traceback de l'erreur de test GUI:")
        print(traceback.format_exc())
def sauvegarder_texte_resultat():
    # ... (Fonction inchangée)
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
    # ... (Fonction inchangée)
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

def activer_widgets_onglet_test(activer):
    # ... (Fonction inchangée)
    etat = "normal" if activer else "disabled"
    bouton_choisir_fichier_test.config(state=etat); entry_chemin_fichier_test.config(state="readonly" if activer else "disabled"); bouton_lancer_test.config(state=etat)
    if not activer:
        var_chemin_fichier_test.set(""); label_statut_test.config(text="")
        text_resultat_pseudo.config(state="normal"); text_resultat_pseudo.delete(1.0, tk.END); text_resultat_pseudo.config(state="disabled")
        bouton_sauvegarder_texte_pseudo.config(state="disabled"); bouton_sauvegarder_mapping_pseudo.config(state="disabled")
    fenetre.update_idletasks()

# --- Interface Graphique ---
# ... (Définition de fenetre, main_canvas, v_scrollbar, content_frame, on_content_frame_configure, _on_mousewheel)
fenetre = tk.Tk()
fenetre.title("Assistant Fine-tuning & Test SpaCy")
fenetre.geometry("800x700") 

main_canvas = tk.Canvas(fenetre)
v_scrollbar = ttk.Scrollbar(fenetre, orient="vertical", command=main_canvas.yview)
main_canvas.configure(yscrollcommand=v_scrollbar.set)
v_scrollbar.pack(side=tk.RIGHT, fill="y")
main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
content_frame = ttk.Frame(main_canvas) 
main_canvas.create_window((0, 0), window=content_frame, anchor="nw", tags="content_frame")
def on_content_frame_configure(event): main_canvas.configure(scrollregion=main_canvas.bbox("all"))
content_frame.bind("<Configure>", on_content_frame_configure)
def _on_mousewheel(event):
    scroll_speed_multiplier = 2 
    if event.num == 4: main_canvas.yview_scroll(-1 * scroll_speed_multiplier, "units")
    elif event.num == 5: main_canvas.yview_scroll(1 * scroll_speed_multiplier, "units")
    else: main_canvas.yview_scroll(int(-1*(event.delta/120) * scroll_speed_multiplier), "units")
for widget_to_bind in [main_canvas, content_frame, fenetre]:
    widget_to_bind.bind("<MouseWheel>", _on_mousewheel); widget_to_bind.bind("<Button-4>", _on_mousewheel); widget_to_bind.bind("<Button-5>", _on_mousewheel)  

# --- Cadre 1: Choix du Modèle (Parent: content_frame) ---
cadre_choix_modele = ttk.LabelFrame(content_frame, text="1. Choix du Modèle", padding=(10, 10))
cadre_choix_modele.pack(padx=10, pady=10, fill="x", anchor="n")
ttk.Label(cadre_choix_modele, text="Modèle SpaCy français ou chemin vers modèle existant :").pack(pady=(0,5), anchor="w") 
choix_modele_var = tk.StringVar(fenetre)
options_modeles_labels = list(MODELES_SPACY_FR_BASE.keys()) + [OPTION_MODELE_EXISTANT]
menu_deroulant_modeles = ttk.Combobox(cadre_choix_modele, textvariable=choix_modele_var, values=options_modeles_labels, state="readonly", width=40)
if options_modeles_labels: menu_deroulant_modeles.current(1) 
menu_deroulant_modeles.pack(pady=5, fill="x")
menu_deroulant_modeles.bind("<<ComboboxSelected>>", on_model_type_changed)

frame_custom_model_path = ttk.Frame(cadre_choix_modele) 
var_custom_model_path = tk.StringVar()
ttk.Label(frame_custom_model_path, text="Chemin modèle existant:", width=22).pack(side=tk.LEFT)
entry_custom_model_path = ttk.Entry(frame_custom_model_path, textvariable=var_custom_model_path, width=30, state="readonly")
entry_custom_model_path.pack(side=tk.LEFT, expand=True, fill="x", padx=2)
bouton_browse_custom_model = ttk.Button(frame_custom_model_path, text="...", width=3, command=lambda: var_custom_model_path.set(filedialog.askdirectory(title="Sélectionner dossier du modèle fine-tuné")))
bouton_browse_custom_model.pack(side=tk.LEFT)

bouton_valider_modele = ttk.Button(cadre_choix_modele, text="Valider Modèle de Base et Continuer", command=valider_choix_modele) 
bouton_valider_modele.pack(pady=(10,0)) 
on_model_type_changed() 

# --- Notebook pour les étapes suivantes (initialement non packé) ---
notebook = ttk.Notebook(content_frame) 
# Le notebook est packé conditionnellement dans valider_choix_modele

# Définir les frames des onglets une seule fois globalement
tab_preparation_donnees = ttk.Frame(notebook, padding=(10,10))
tab_fine_tuning = ttk.Frame(notebook, padding=(10,10))
tab_test_modele = ttk.Frame(notebook, padding=(10,10))

# --- Contenu de l'Onglet 1: Préparation Données ---
var_actif_per = tk.BooleanVar(value=True); var_entites_per = tk.StringVar(); var_phrases_per = tk.StringVar(); var_placeholder_per = tk.StringVar(value="{NOM}")
var_actif_loc = tk.BooleanVar(value=True); var_entites_loc = tk.StringVar(); var_phrases_loc = tk.StringVar(); var_placeholder_loc = tk.StringVar(value="{LOC}")
var_actif_org = tk.BooleanVar(value=True); var_entites_org = tk.StringVar(); var_phrases_org = tk.StringVar(); var_placeholder_org = tk.StringVar(value="{ORG}")
def creer_section_type_entite(parent, label_type, var_actif_chk, var_ent_path, var_phr_path, var_plh):
    frame_type = ttk.LabelFrame(parent, text=f" {label_type} ", padding=5); frame_type.pack(fill="x", expand=True, pady=3, padx=3)
    chk_button = ttk.Checkbutton(frame_type, text="Activer ce type", variable=var_actif_chk); chk_button.pack(anchor="w")
    f_ent = ttk.Frame(frame_type); f_ent.pack(fill="x", pady=1)
    ttk.Label(f_ent, text="Fichier Entités:", width=15).pack(side=tk.LEFT)
    entry_ent = ttk.Entry(f_ent, textvariable=var_ent_path, width=35, state="readonly"); entry_ent.pack(side=tk.LEFT, expand=True, fill="x", padx=1)
    btn_ent = ttk.Button(f_ent, text="...", width=3, command=lambda v=var_ent_path, lt=label_type: choisir_fichier_pour_config(lt, "Entités", v)); btn_ent.pack(side=tk.LEFT)
    f_phr = ttk.Frame(frame_type); f_phr.pack(fill="x", pady=1)
    ttk.Label(f_phr, text="Fichier Phrases:", width=15).pack(side=tk.LEFT)
    entry_phr = ttk.Entry(f_phr, textvariable=var_phr_path, width=35, state="readonly"); entry_phr.pack(side=tk.LEFT, expand=True, fill="x", padx=1)
    btn_phr = ttk.Button(f_phr, text="...", width=3, command=lambda v=var_phr_path, lt=label_type: choisir_fichier_pour_config(lt, "Phrases", v)); btn_phr.pack(side=tk.LEFT)
    f_plh = ttk.Frame(frame_type); f_plh.pack(fill="x", pady=1)
    ttk.Label(f_plh, text="Placeholder:", width=15).pack(side=tk.LEFT)
    entry_plh = ttk.Entry(f_plh, textvariable=var_plh, width=10); entry_plh.pack(side=tk.LEFT)
    config_widgets_preparation.append({ # Stocker les références pour activer/désactiver
        "checkbutton": chk_button, "entites_entry": entry_ent, "entites_btn": btn_ent,
        "phrases_entry": entry_phr, "phrases_btn": btn_phr, "placeholder_entry": entry_plh
    })
creer_section_type_entite(tab_preparation_donnees, "PER", var_actif_per, var_entites_per, var_phrases_per, var_placeholder_per)
creer_section_type_entite(tab_preparation_donnees, "LOC", var_actif_loc, var_entites_loc, var_phrases_loc, var_placeholder_loc)
creer_section_type_entite(tab_preparation_donnees, "ORG", var_actif_org, var_entites_org, var_phrases_org, var_placeholder_org)
bouton_generer_donnees_gui = ttk.Button(tab_preparation_donnees, text="Générer et Combiner les Données", command=lancer_generation_donnees_gui); bouton_generer_donnees_gui.pack(pady=10)
text_log_preparation = scrolledtext.ScrolledText(tab_preparation_donnees, height=5, width=80, state="disabled", wrap=tk.WORD); text_log_preparation.pack(pady=5, fill="x", expand=False)

# --- Contenu de l'Onglet 2: Fine-tuning Modèle ---
var_iterations = tk.IntVar(value=10); var_dropout = tk.DoubleVar(value=0.3); var_chemin_sauvegarde_modele = tk.StringVar()
frame_iter = ttk.Frame(tab_fine_tuning); frame_iter.pack(fill="x", pady=2)
ttk.Label(frame_iter, text="Nombre d'itérations:", width=22).pack(side=tk.LEFT, padx=(0,5))
entry_iterations = ttk.Spinbox(frame_iter, from_=1, to=1000, textvariable=var_iterations, width=10); entry_iterations.pack(side=tk.LEFT)
frame_drop = ttk.Frame(tab_fine_tuning); frame_drop.pack(fill="x", pady=2)
ttk.Label(frame_drop, text="Taux de Dropout:", width=22).pack(side=tk.LEFT, padx=(0,5)) 
entry_dropout = ttk.Spinbox(frame_drop, from_=0.0, to=1.0, increment=0.05, textvariable=var_dropout, width=10, format="%.2f"); entry_dropout.pack(side=tk.LEFT)
frame_sauvegarde_modele = ttk.Frame(tab_fine_tuning); frame_sauvegarde_modele.pack(fill="x", pady=2)
ttk.Label(frame_sauvegarde_modele, text="Dossier sauvegarde modèle:", width=22).pack(side=tk.LEFT, padx=(0,5)) 
entry_chemin_sauvegarde_modele = ttk.Entry(frame_sauvegarde_modele, textvariable=var_chemin_sauvegarde_modele, state="readonly", width=40); entry_chemin_sauvegarde_modele.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_dossier_modele = ttk.Button(frame_sauvegarde_modele, text="...", width=3, command=choisir_dossier_sauvegarde_modele); bouton_choisir_dossier_modele.pack(side=tk.LEFT)
bouton_lancer_fine_tuning = ttk.Button(tab_fine_tuning, text="Lancer le Fine-tuning", command=lancer_fine_tuning_gui); bouton_lancer_fine_tuning.pack(pady=10)
ttk.Label(tab_fine_tuning, text="Log du Fine-tuning:").pack(anchor="w", pady=(5,0))
text_log_fine_tuning = scrolledtext.ScrolledText(tab_fine_tuning, height=7, width=80, state="disabled", wrap=tk.WORD); text_log_fine_tuning.pack(pady=5, fill="x", expand=False)

# --- Contenu de l'Onglet 3: Test Modèle ---
var_chemin_fichier_test = tk.StringVar()
frame_fichier_test = ttk.Frame(tab_test_modele); frame_fichier_test.pack(fill="x", pady=5)
ttk.Label(frame_fichier_test, text="Fichier Texte de Test:", width=22).pack(side=tk.LEFT, padx=(0,5)) 
entry_chemin_fichier_test = ttk.Entry(frame_fichier_test, textvariable=var_chemin_fichier_test, state="readonly", width=40); entry_chemin_fichier_test.pack(side=tk.LEFT, expand=True, fill="x", padx=(0,5))
bouton_choisir_fichier_test = ttk.Button(frame_fichier_test, text="...", width=3, command=choisir_fichier_test_txt); bouton_choisir_fichier_test.pack(side=tk.LEFT)
bouton_lancer_test = ttk.Button(tab_test_modele, text="Lancer Test Pseudonymisation", command=lancer_test_pseudonymisation_gui); bouton_lancer_test.pack(pady=10)
label_statut_test = ttk.Label(tab_test_modele, text=""); label_statut_test.pack(pady=2)
ttk.Label(tab_test_modele, text="Résultat Pseudonymisation:").pack(anchor="w", pady=(5,0))
text_resultat_pseudo = scrolledtext.ScrolledText(tab_test_modele, height=8, width=80, state="disabled", wrap=tk.WORD); text_resultat_pseudo.pack(pady=5, fill="both", expand=True)
frame_sauvegarde_test = ttk.Frame(tab_test_modele); frame_sauvegarde_test.pack(pady=5)
bouton_sauvegarder_texte_pseudo = ttk.Button(frame_sauvegarde_test, text="Sauvegarder Texte", command=sauvegarder_texte_resultat, state="disabled"); bouton_sauvegarder_texte_pseudo.pack(side=tk.LEFT, padx=5) 
bouton_sauvegarder_mapping_pseudo = ttk.Button(frame_sauvegarde_test, text="Sauvegarder Mapping", command=sauvegarder_mapping_resultat, state="disabled"); bouton_sauvegarder_mapping_pseudo.pack(side=tk.LEFT, padx=5)

# Initialisation (le notebook n'est pas packé, les widgets internes des onglets sont désactivés par défaut par leurs fonctions d'activation)
activer_widgets_onglet_preparation(False) 
activer_widgets_onglet_fine_tuning(False)
activer_widgets_onglet_test(False)
# Les onglets ne sont pas ajoutés au notebook ici, mais dans valider_choix_modele

fenetre.mainloop()