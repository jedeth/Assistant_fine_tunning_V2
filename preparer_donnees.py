import re # Utilisé pour trouver la position exacte du nom

def lire_noms(chemin_fichier_noms):
    """
    Lit les noms à partir d'un fichier texte.
    Chaque nom est attendu sur une nouvelle ligne.
    """
    noms = []
    with open(chemin_fichier_noms, 'r', encoding='utf-8') as f:
        for ligne in f:
            nom_propre = ligne.strip() # Enlève les espaces au début/fin
            if nom_propre: # S'assure que la ligne n'est pas vide
                noms.append(nom_propre)
    return noms

def lire_phrases_modeles(chemin_fichier_phrases):
    """
    Lit les phrases modèles à partir d'un fichier texte.
    Chaque phrase modèle est attendue sur une nouvelle ligne.
    Le placeholder {NOM} doit être présent.
    """
    phrases = []
    # Le fichier fourni semble contenir des chaînes Python avec des virgules.
    # Cette fonction est simplifiée et s'attend à une phrase modèle par ligne,
    # sans les guillemets de début/fin de chaîne ni les virgules de fin de ligne.
    # Exemple de ligne attendue dans le fichier :
    # Après une analyse approfondie du dossier soumis par {NOM}, le comité a décidé.
    with open(chemin_fichier_phrases, 'r', encoding='utf-8') as f:
        for ligne in f:
            phrase_modele = ligne.strip()
            # Nettoyage simple pour enlever les guillemets et la virgule de fin si présents
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
    return phrases

def generer_donnees_entrainement(noms, phrases_modeles):
    """
    Génère les données d'entraînement au format SpaCy.
    """
    donnees_entrainement = []
    label_entite = "PER" # Étiquette pour les personnes

    for nom in noms:
        for phrase_modele in phrases_modeles:
            # Remplace le placeholder {NOM} par le nom actuel
            phrase_formatee = phrase_modele.replace("{NOM}", nom)
            
            # Trouve les indices de début et de fin du nom dans la phrase formatée
            # Utiliser re.escape pour gérer les caractères spéciaux dans les noms (s'il y en avait)
            match = re.search(re.escape(nom), phrase_formatee)
            if match:
                debut, fin = match.span()
                entite = (debut, fin, label_entite)
                donnees_entrainement.append((phrase_formatee, {"entities": [entite]}))
            else:
                # Cela peut arriver si le nom contient des caractères que .replace modifie
                # ou si le nom est un sous-ensemble d'un mot plus grand après remplacement.
                # Pour des noms simples, cela ne devrait pas poser problème.
                print(f"Attention : Impossible de trouver le nom '{nom}' dans la phrase générée '{phrase_formatee}'")

    return donnees_entrainement

# --- Programme Principal ---
if __name__ == "__main__":
    chemin_annuaire = "annuaire_simplifié.txt"
    chemin_phrases = "phrases_modeles.txt"

    print(f"Lecture des noms depuis : {chemin_annuaire}")
    noms_propres = lire_noms(chemin_annuaire) # [cite: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Nombre de noms lus : {len(noms_propres)}")
    if noms_propres:
        print(f"Exemple de nom lu : '{noms_propres[0]}'")

    print(f"\nLecture des phrases modèles depuis : {chemin_phrases}")
    modeles_phrases = lire_phrases_modeles(chemin_phrases) # [cite: 11, 12, 13, 14, 15]
    print(f"Nombre de phrases modèles lues : {len(modeles_phrases)}")
    if modeles_phrases:
        print(f"Exemple de phrase modèle lue : '{modeles_phrases[0]}'")

    if noms_propres and modeles_phrases:
        print("\nGénération des données d'entraînement...")
        donnees_pour_spacy = generer_donnees_entrainement(noms_propres, modeles_phrases)
        print(f"Nombre d'exemples d'entraînement générés : {len(donnees_pour_spacy)}")

        if donnees_pour_spacy:
            print("\nExemple de données d'entraînement générées :")
            for i in range(min(3, len(donnees_pour_spacy))): # Affiche les 3 premiers exemples
                print(donnees_pour_spacy[i])
        
        # Optionnel: Sauvegarder les données générées dans un fichier (par exemple, format JSON)
        import json
        with open("donnees_entrainement_spacy.json", "w", encoding="utf-8") as outfile:
             json.dump(donnees_pour_spacy, outfile, ensure_ascii=False, indent=4)
        print("\nDonnées d'entraînement sauvegardées dans 'donnees_entrainement_spacy.json'")
    else:
        print("\nImpossible de générer les données d'entraînement car la liste de noms ou de phrases modèles est vide.")