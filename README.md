# Projet d'Assistant de Fine-tuning et Pseudonymisation avec SpaCy

## Description
Ce projet vise à fournir des outils pour fine-tuner des modèles de langage SpaCy (spécifiquement les modèles français `fr_core_news_sm/md/lg`) pour la reconnaissance d'entités nommées (NER) et à utiliser ces modèles fine-tunés pour la pseudonymisation de textes. Il comprend un script en ligne de commande pour la préparation avancée des données d'entraînement et une application avec interface graphique (GUI) pour orchestrer l'ensemble du processus, du choix du modèle au test final.

L'objectif principal est d'améliorer la capacité des modèles SpaCy à identifier des types d'entités spécifiques (Personnes, Lieux, Organisations, etc.) afin d'effectuer une pseudonymisation plus précise et de réduire les erreurs, comme la confusion de dates avec des noms de personnes.

## Fonctionnalités

### 1. Préparation des Données d'Entraînement (`preparer_donnees_multi_types.py`)
Cet outil en ligne de commande est crucial pour générer des données d'entraînement de qualité au format SpaCy. Il est conçu pour gérer plusieurs types d'entités (par exemple, PER, LOC, ORG).

* **Entrées :**
    * Des fichiers texte (`.txt`) contenant des listes d'entités, une par ligne (ex: `annuaire_noms.txt`, `lieux.txt`, `organisations.txt`).
    * Des fichiers texte (`.txt`) contenant des phrases modèles avec des placeholders correspondants (ex: `{NOM}` pour les personnes, `{LOC}` pour les lieux, `{ORG}` pour les organisations).
* **Sorties :**
    * Des fichiers JSON distincts contenant les données d'entraînement pour chaque type d'entité (ex: `donnees_entrainement_loc.json`).
    * Un fichier JSON principal (`donnees_entrainement_combinees.json`) qui regroupe toutes les données générées. Ce fichier est ensuite utilisé par l'application GUI pour le fine-tuning.

### 2. Application GUI Complète (`gui_fine_tuning.py`)
Une application de bureau développée avec Tkinter qui guide l'utilisateur à travers un pipeline complet :

* **Étape 1: Choix du Modèle SpaCy de Base**
    * Permet de sélectionner un modèle SpaCy français pré-entraîné (`fr_core_news_sm`, `fr_core_news_md`, ou `fr_core_news_lg`).
    * Gère le téléchargement automatique du modèle sélectionné via `python -m spacy download nom_du_modele` si celui-ci n'est pas déjà installé sur le système.
* **Étape 2: Sélection des Données d'Entraînement**
    * L'utilisateur sélectionne le fichier JSON de données d'entraînement combinées (préalablement généré par `preparer_donnees_multi_types.py`).
* **Étape 3: Fine-tuning du Modèle**
    * Permet de configurer les paramètres essentiels du fine-tuning : nombre d'itérations et taux de dropout.
    * L'utilisateur choisit un dossier de destination pour sauvegarder le modèle SpaCy une fois fine-tuné.
    * Le processus de fine-tuning est lancé, et les informations de progression (comme la perte ou "loss" à chaque itération) sont affichées dans une zone de log.
* **Étape 4: Tester le Modèle Fine-tuné**
    * Après un fine-tuning réussi, cette section s'active.
    * L'utilisateur sélectionne un fichier texte (`.txt`) à utiliser pour le test.
    * Le modèle qui vient d'être fine-tuné lors de la session est utilisé pour pseudonymiser le contenu du fichier de test.
    * Le texte pseudonymisé est affiché dans l'interface.
    * Des options sont disponibles pour sauvegarder le texte pseudonymisé dans un nouveau fichier (un nom de fichier de sortie, par exemple `original_pseudonymise.txt`, est automatiquement suggéré) et pour sauvegarder la table de correspondance (mapping) entre les noms originaux et leurs pseudonymes dans un fichier JSON.

## Prérequis et Installation
1.  **Python** : Version 3.7 ou ultérieure recommandée.
2.  **SpaCy et Modèles Français** :
    Installez SpaCy :
    ```bash
    pip install spacy
    ```
    Les modèles SpaCy français de base (`fr_core_news_sm`, `fr_core_news_md`, `fr_core_news_lg`) peuvent être téléchargés via l'interface graphique si absents, ou manuellement :
    ```bash
    python -m spacy download fr_core_news_md 
    ```
3.  **Bibliothèques Python** :
    Le projet utilise principalement des bibliothèques standard de Python : `tkinter` (pour l'interface graphique, généralement inclus avec Python), `json`, `os`, `re`, `random`, `subprocess`. Aucune installation de bibliothèque externe majeure n'est requise au-delà de SpaCy.

## Guide d'Utilisation

### A. Préparation des Données d'Entraînement (Fortement Recommandé)
Avant d'utiliser l'interface graphique pour le fine-tuning, préparez des données d'entraînement de haute qualité avec le script `preparer_donnees_multi_types.py`.
1.  **Créez vos fichiers d'entités** :
    * Un fichier `.txt` par type d'entité (ex: `mon_annuaire.txt` pour les personnes, `mes_lieux.txt`, `mes_organisations.txt`). Listez chaque entité sur une nouvelle ligne.
2.  **Créez vos fichiers de phrases modèles** :
    * Un fichier `.txt` par type d'entité, contenant des phrases avec un placeholder unique pour ce type (ex: `phrases_personnes.txt` avec `{NOM}`, `phrases_lieux.txt` avec `{LOC}`).
3.  **Configurez le script** : Ouvrez `preparer_donnees_multi_types.py` et modifiez la liste `types_a_generer` pour qu'elle pointe vers vos fichiers et utilise les bons labels et placeholders.
4.  **Exécutez le script** :
    ```bash
    python preparer_donnees_multi_types.py
    ```
    Cela générera (entre autres) un fichier `donnees_entrainement_combinees.json`.

### B. Utilisation de l'Application GUI (`gui_fine_tuning.py`)
1.  **Lancez l'application** :
    ```bash
    python gui_fine_tuning.py
    ```
2.  **Suivez les étapes dans l'interface** :
    * **Étape 1**: Choisissez un modèle SpaCy de base (sm, md, ou lg) et cliquez sur "Valider Modèle et Continuer". Le modèle sera téléchargé si nécessaire.
    * **Étape 2**: Cliquez sur "Parcourir..." pour sélectionner votre fichier `donnees_entrainement_combinees.json` (ou le nom que vous lui avez donné). Cliquez ensuite sur "Valider Fichier de Données".
    * **Étape 3**: Entrez le nombre d'itérations souhaité, le taux de dropout, et choisissez un dossier où votre modèle fine-tuné sera sauvegardé. Cliquez sur "Lancer le Fine-tuning". Suivez la progression dans la zone de log.
    * **Étape 4**: Une fois le fine-tuning terminé, ce cadre s'activera. Choisissez un fichier `.txt` à tester. Cliquez sur "Lancer le Test de Pseudonymisation". Visualisez le résultat et utilisez les boutons pour sauvegarder le texte pseudonymisé et/ou la table de correspondance.

## Amélioration Continue du Modèle
La performance de votre modèle de pseudonymisation dépendra grandement de la qualité, de la quantité et de la diversité de vos données d'entraînement.
* Pour corriger des erreurs spécifiques (par exemple, si le modèle confond des dates avec des personnes), il est crucial d'ajouter des exemples d'entraînement pour les types d'entités que vous souhaitez que le modèle reconnaisse correctement (PER, LOC, ORG, et potentiellement DATE si vous voulez qu'il les identifie spécifiquement).
* Utilisez `preparer_donnees_multi_types.py` pour continuellement enrichir et affiner votre jeu de données.

## Considérations sur le Matériel
* Par défaut, le fine-tuning des modèles SpaCy avec ce script s'effectue sur le **CPU**.
* Pour une accélération significative, l'utilisation d'un **GPU** (NVIDIA avec CUDA ou AMD avec ROCm) est recommandée. Cela nécessite une installation de SpaCy avec le support GPU (ex: `pip install spacy[cuda]`) et une configuration correcte des pilotes et bibliothèques associées.
* L'utilisation de **NPU** (comme Intel AI Boost) pour le *fine-tuning* n'est pas directement prise en charge par ces scripts de manière standard. Les NPU sont plus souvent ciblés pour l'accélération de l'*inférence* de modèles, souvent après une conversion du modèle (par exemple au format ONNX) et en utilisant des outils spécifiques comme OpenVINO. Le fine-tuning sur NPU est un domaine plus avancé et moins directement accessible avec les bibliothèques standards actuelles pour ce type de modèle.

---
N'hésitez pas à adapter ce README avec plus de détails sur votre projet, des exemples spécifiques, ou des instructions de contribution si vous le souhaitez !