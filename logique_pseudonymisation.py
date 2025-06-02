# logique_pseudonymisation.py

# Pas besoin d'importer spacy ici si nlp_model est toujours un objet SpaCy chargé.
# tkinter.messagebox ne doit pas être utilisé ici.

def pseudonymiser_texte(nlp_model, texte_original):
    """
    Pseudonymise le texte en utilisant le modèle SpaCy chargé.
    Retourne le texte pseudonymisé et la table de correspondance.
    En cas d'erreur (ex: modèle non fourni), retourne (None, None).
    """
    if not nlp_model:
        print("ERREUR (logique_pseudonymisation): Aucun modèle SpaCy n'a été fourni pour la pseudonymisation.")
        return None, None # Indique une erreur à l'appelant

    doc = nlp_model(texte_original)
    
    correspondances = {}
    pseudonyme_compteur = 1
    
    entites_a_remplacer = []

    for entite in doc.ents:
        # On se concentre sur PER pour la pseudonymisation par défaut, 
        # mais le modèle peut en détecter d'autres.
        if entite.label_ == "PER": 
            nom_original = entite.text.strip() 
            
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