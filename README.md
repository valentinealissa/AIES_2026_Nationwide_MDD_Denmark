<h1 align="center">Nationwide AI Models for Depression Prediction: Regional Generalizability and Culturally Sensitive Variable Selection</h1>
Authors: Alissa Valentine, Helen Coupland, Eike Petersen, Emily Beaman, Merete Osler, Aasa Feragen, Melanie Ganz

# Contents
This repo contains supplementary materials, sample code, and a mapping file used in the paper "Nationwide AI Models for Depression Prediction: Regional Generalizability and Culturally Sensitive Variable Selection", accepted to AIES 2026. 

# Supplementary Materials
Please view the `Supplementary_materials.pdf` file to see information on:
1. Data pre-processing for MDD prediction task
2. Tabular transformer model and training for MDD prediction task
3. Model performance with multiple seeds
4. Feature importance for MDD prediction
5. Table 4: Performance by model and seed for the two MDD prediction models fine-tuned on data from all five administrative regions.
6. Table 5: Sociodemographic characteristics of the matched cohort dataset used in our analyses.
7. Table 6: Socioeconomic characteristics of the matched cohort dataset used in our analyses.
8. Figure 2: Fairness evaluation of the Heritage-Informed Model trained on data from the 5 administrative regions of Denmark, produced by the meval package (Sutariya and Petersen 2025).
9. Figure 3: Global Feature Importance for the Baseline and the Heritage-Informed Models.

# Sample Code
Please view the `run_transformer.py` file for example code on how to train the TF-Transformer model for the binary classification of MDD diagnosis.

# Mapping File
Please view the `WHO_UN_country_map251211.csv` file for a mapping file that includes country names, UN regions, and other classifications that can be used to reproduce our work.
