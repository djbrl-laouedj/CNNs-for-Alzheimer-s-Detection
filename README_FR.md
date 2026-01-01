⚠️ En raison des limitations de taille de fichiers imposées par GitHub, le modèle d’auto-encodeur entraîné (best_auto-encodeur.keras) n’est pas inclus dans ce dépôt.

Pour le générer, ouvrez et exécutez le notebook suivant :
```
AutoEncoder.ipynb
```

---

## Environnement d’exécution

Toutes les expériences ont été réalisées sur **Google Colab avec GPU**.

---

## Inspiration

La maladie d’Alzheimer est une pathologie neurologique progressive qui impacte profondément les patients, leurs familles et les systèmes de santé. L’un des principaux défis réside dans **la détection précoce et l’évaluation fiable de la sévérité de la maladie**, d’autant plus que les différences visuelles entre les IRM peuvent être subtiles et difficiles à interpréter, même pour des cliniciens expérimentés.

Nous sommes partis d’une question simple mais centrale :

**Les modèles de deep learning peuvent-ils non seulement classifier les stades de la maladie d’Alzheimer, mais aussi aider à comprendre comment les structures cérébrales s’éloignent progressivement d’un cerveau sain ?**

Plutôt que de nous limiter à une seule approche, nous avons choisi d’explorer **plusieurs perspectives complémentaires** :

*L’apprentissage supervisé classique, le transfer learning avec des architectures de pointe, et enfin une approche auto-supervisée de détection d’anomalies ne nécessitant aucun label.*

---

## Ce que fait le projet

Notre projet analyse des IRM cérébrales afin d’étudier la maladie d’Alzheimer à travers **trois approches complémentaires** :

**1. Modèles CNN supervisés**

Des réseaux de neurones convolutifs entraînés from scratch pour classifier les IRM en quatre stades :

- Non-Demented

- Very Mild Dementia

- Mild Dementia

- Moderate Dementia

**2. Modèles en Transfer Learning**

Des modèles EfficientNet et ResNet, pré-entraînés sur ImageNet, puis fine-tunés pour l’imagerie médicale.

**3. Détection d’anomalies auto-supervisée**

**Un auto-encodeur entraîné uniquement sur des cerveaux sains** (Non-Demented) permettant :

- La détection de déviations structurelles

- La visualisation de cartes d’anomalies mettant en évidence les régions qui divergent du patron sain appris

Ces approches combinées offrent à la fois une **évaluation quantitative** (performances de classification) et une **interprétation qualitative** (visualisation des anomalies) de la progression de la maladie.

---

## Comment nous l’avons construit

**1. CNN supervisés – modèles de base**

Nous avons commencé par un CNN simple entraîné sur des IRM en niveaux de gris (128×128).

À partir de ce modèle de base, nous avons progressivement amélioré les performances via :

- L’ajout de data augmentation

- La gestion du déséquilibre des classes avec des class weights

- L’oversampling des classes minoritaires

- Le tuning des hyperparamètres

Chaque variante a été évaluée indépendamment afin de mesurer précisément l’impact de chaque amélioration.

**2. Transfer Learning (EfficientNet & ResNet)**

Pour aller plus loin, nous avons implémenté du **transfer learning** avec :

- EfficientNetB0

- ResNet50

Choix clés :

- Conversion des IRM en niveaux de gris vers RGB

- Gel initial des couches pré-entraînées

- **Fine-tuning** progressif des couches profondes

- Utilisation d’un **learning rate adaptatif** (ReduceLROnPlateau) et de **l’early stopping**

Cette stratégie nous a permis de tirer parti de représentations riches tout en les adaptant au domaine médical.

**3. Évaluation “production-like” multi-datasets**

Afin de simuler un scénario réaliste de déploiement, nous avons testé tous les modèles sur des **datasets IRM totalement inédits / inconnus**.

Cette étape a mis en évidence un point clé :

*Des modèles affichant une excellente accuracy en validation peuvent s’effondrer face à un changement de distribution des données.*

Cela nous a poussés à explorer une approche radicalement différente.

**4. Détection d’anomalies auto-supervisée par auto-encodeur**

Plutôt que de prédire des labels, nous avons entraîné un **auto-encodeur convolutionnel uniquement sur des IRM Non-Demented**.

Principe :

- Apprendre une **représentation compacte du cerveau sain**

- Reconstruire fidèlement les images saines

- Observer **les erreurs de reconstruction** lorsque la structure cérébrale diverge de la normalité

Nous avons utilisé :

- **Keras Tuner** pour rechercher automatiquement l’architecture encodeur–décodeur optimale

- La MSE comme fonction de perte

- Un seuil d’anomalie **pixel-wise calibré statistiquement** à partir d’images saines

Résultats :

- Cartes d’anomalies pixel par pixel

- Indicateurs quantitatifs de sévérité (erreur moyenne, ratio d’anomalies)

- Visualisation claire de la progression structurelle entre les stades de la maladie

---

## Structure du dépôt

```
CNNs-for-Alzheimer-s-Detection/

├── AutoEncoder.ipynb
│   # Auto-encodeur auto-supervisé pour la détection d’anomalies (entraîné sur Non-Demented)

├── Kaggle_MRI_Alzheimers_Djebril_Redha_vf.ipynb
│   # Notebook principal : CNN supervisés, transfer learning (EfficientNet, ResNet),
│   # tests en conditions proches de la production et évaluation cross-datasets

├── OAS1_0003_MR1_mpr-3_105.jpg
├── OAS1_0004_MR1_mpr-2_116.jpg
├── OAS1_0028_MR1_mpr-2_105.jpg
├── OAS1_0308_MR1_mpr-3_123.jpg
│   # Images IRM utilisées pour les tests qualitatifs et visualisations

├── README.md
│   # Documentation Anglaise du projet (motivation, méthodes, résultats, limites)

├── README_FR.md
│   # Documentation Francaise du projet (motivation, méthodes, résultats, limites)

├── best_model.keras
│   # Meilleur CNN supervisé (baseline / optimisé)

├── best_model_v2.keras
│   # CNN fine-tuné avec stratégie de learning rate adaptatif

├── train.parquet
│   # Métadonnées et labels du jeu d’entraînement prétraité

├── test.parquet
│   # Métadonnées et labels du jeu de test prétraité

└── .gitignore
    # Fichiers exclus du versionnement
```

---

## Difficultés rencontrées

- **Hétérogénéité des datasets** : différences de contraste, résolution et protocoles d’acquisition

- **Déséquilibre des classes** : sous-représentation des stades avancés

- **Généralisation** : une bonne accuracy ne garantit pas la robustesse

- **Interprétabilité** : la classification seule n’explique pas les prédictions

- **Contraintes matérielles** : gestion fine de la mémoire pour éviter les OOM GPU

---

## Ce dont nous sommes fiers

- Avoir construit **une pipeline complète**, du CNN de base au self-supervised learning

- Avoir mis en évidence les **limites du supervisé pur** face au dataset shift

- Obtenir une **séparation visuelle claire** des stades via les cartes d’anomalies

- Proposer une approche économe en labels, **interprétable** et **intuitive**

- Combiner métriques **quantitatives** et explications **visuelles** dans un même projet

---

## Ce que nous avons appris

- Une accuracy élevée ne garantit ni robustesse ni utilité clinique

- Le transfer learning améliore fortement les performances mais reste sensible au domaine

- L’auto-supervisé est une alternative puissante quand les labels sont rares ou biaisés

- Les auto-encodeurs révèlent **des changements structurels progressifs** sans supervision

- La visualisation est essentielle pour instaurer la confiance en IA médicale

---

## Perspectives

Avec plus de temps, nous souhaiterions explorer :

- Les **Masked Autoencoders (MAE)**

- Des approches hybrides combinant **scores d’anomalie** + **classification supervisée**

- Une analyse régionale de la sévérité via des masques anatomiques cérébraux

---

## Avertissement

Ce projet a été réalisé dans le cadre du **Hackathon AI 4 Alzheimer’s** et est destiné exclusivement à des fins de **recherche, d’enseignement et d’exploration.**

Les modèles et visualisations présentés **ne sont pas des dispositifs médicaux** et **ne doivent pas être utilisés pour le diagnostic**, **le traitement ou la prise de décision clinique.**
Les zones mises en évidence **ne correspondent pas à des lésions médicales exactes**, mais à des régions où la structure cérébrale diverge du modèle appris à partir des données.

Les résultats doivent être interprétés avec prudence et **ne remplacent en aucun cas l’avis d’un professionnel de santé qualifié**.

---

## 👤 Auteurs

Ce projet a été développé par **Djebril Laouedj** et **Redha Ibbou** [@KYX6](https://github.com/KYX6), 
étudiants en dernière année en **Big Data & Intelligence Artificielle** à **l'ECE Paris**.
