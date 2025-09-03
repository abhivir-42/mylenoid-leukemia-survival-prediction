# Overall Survival Prediction for Patients Diagnosed with Myeloid Leukemia

## Challenge Context

In recent years, healthcare has increasingly adopted data-driven approaches, particularly in the prognosis and treatment of complex diseases like cancer. Predictive models have revolutionized patient care by enabling personalized and effective treatment strategies. In oncology, accurate prognostic models significantly influence the quality and timing of treatment decisions, making them critical for improving patient outcomes.

## Challenge Goals

In collaboration with **Institut Gustave Roussy**, the QRT Data Challenge focuses on predicting disease risk for patients diagnosed with **adult myeloid leukemias**, a subtype of blood cancer. The primary measure of risk assessment is **overall survival (OS)**, defined as the period from initial diagnosis to either the patient’s death or the last recorded follow-up.

### Why This Matters
Accurate prognosis estimation is crucial for tailoring treatment plans:
- **Low-risk patients** may receive supportive therapies to improve blood counts and quality of life.
- **High-risk patients** may be prioritized for intensive treatments, such as **hematopoietic stem cell transplantation**.

Precise risk predictions enhance clinical decision-making, improve patient outcomes, and optimize resource allocation in healthcare facilities. This challenge provides participants with an opportunity to work with **real data from 24 clinical centers** and contribute to impactful applications of data science in medicine.

## Data Description

The dataset is divided into three components:
- **X_train.zip**: Training set with data on **3,323 patients**.
- **X_test.zip**: Test set with data on **1,193 patients**.
- **Y_train.csv**: Contains outcome data for the training set.

The input data is organized into two categories:
- **Clinical Data**: One row per patient, containing clinical metrics.
- **Molecular Data**: One row per somatic mutation per patient, detailing genetic mutations.

The **ID** column serves as the **unique patient identifier**, linking Clinical Data, Molecular Data, and Y_train.

### Clinical Data (One Row per Patient)
Each patient is associated with detailed clinical information:
- **ID**: Unique patient identifier.
- **CENTER**: Clinical center where the patient is treated.
- **BM_BLAST**: Bone marrow blasts percentage, indicating the proportion of abnormal blood cells in the bone marrow.
- **WBC**: White Blood Cell count (Giga/L).
- **ANC**: Absolute Neutrophil Count (Giga/L).
- **MONOCYTES**: Monocyte count (Giga/L).
- **HB**: Hemoglobin level (g/dL).
- **PLT**: Platelet count (Giga/L).
- **CYTOGENETICS**: Description of chromosomal abnormalities in blood cancer cells, following the **ISCN standard** (e.g., “46,XX” for normal female karyotype, “46,XY” for normal male karyotype). Abnormalities like **monosomy 7** indicate higher-risk disease.

### Gene Molecular Data (One Row per Somatic Mutation per Patient)
Somatic mutations are specific to cancerous cells and not present in normal cells. The data includes:
- **ID**: Unique patient identifier.
- **CHR, START, END**: Chromosomal position of the mutation.
- **REF, ALT**: Reference and alternate (mutant) nucleotides.
- **GENE**: Affected gene.
- **PROTEIN_CHANGE**: Impact of the mutation on the protein.
- **EFFECT**: Classification of the mutation’s impact on gene function.
- **VAF**: Variant Allele Fraction, indicating the proportion of cells with the mutation.

### Outcome
The goal is to predict **overall survival (OS)** for patients diagnosed with blood cancer. The **Y_train.csv** file provides two key outcomes:
- **OS_YEARS**: Survival time in years from diagnosis.
- **OS_STATUS**: Survival status indicator (1 = death, 0 = alive at last follow-up).

The expected output is a CSV file with:
- **ID**: Patient identifier (as index).
- **risk_score**: Predicted risk of death, where a lower score for patient *i* compared to patient *j* indicates that *i* is predicted to survive longer than *j*.

The scale of the predictions is irrelevant; only the order matters. A sample submission file with random predictions is provided in the Files section.

## Loss Metric: Concordance Index for Right-Censored Data with IPCW

The evaluation metric is the **Concordance Index for Right-Censored Data with IPCW (IPCW-C-index)**.

### Concordance Index (C-index)
The **C-index** measures the model’s ability to correctly rank survival times. It calculates the proportion of comparable patient pairs where the predicted disease risk aligns with actual survival times.

For a pair of patients *i* and *j* with survival times *T_i* and *T_j*:
- A pair is **comparable** if *T_i < T_j* and both survival times are known (i.e., not censored before *T_j*).
- A pair is **concordant** if the predicted risk for *i* is higher than for *j* (*R_i > R_j*) when *T_i < T_j*.

The C-index is computed as: C = (Number of Concordant Pairs) / (Total Number of Comparable Pairs)

- **C = 1**: Perfect ranking.
- **C = 0.5**: Random predictions.
- **C = 0**: Completely incorrect ranking.

### IPCW-C-index
The **IPCW-C-index** extends the C-index to handle **right-censored data** by applying **inverse probability of censoring weights (IPCW)**. Right-censoring occurs when a patient’s survival time is only partially known (e.g., the patient is alive at the last follow-up). IPCW assigns weights to pairs based on the likelihood of observing the survival time, improving the metric’s robustness.

The metric is implemented using the **scikit-survival** library (see [scikit-survival documentation](https://scikit-survival.readthedocs.io/)) and is clipped to a maximum horizon of **7 years**.

## Benchmark Description

Two benchmark models are provided:
1. **LightGBM Model**: A simple model using only clinical data, ignoring censoring.
2. **Cox Proportional Hazards Model**: Incorporates clinical data and limited gene mutation data, accounting for censoring.

The **LightGBM model** serves as an example, while the **Cox model** score determines the challenge ranking. A benchmark notebook with the code for these models is available in the additional files section.

### About Cox Proportional Hazards Model
The **Cox Proportional Hazards Model** is a statistical technique for survival analysis, modeling the hazard function (instantaneous risk of an event) as a function of multiple variables. It assumes that the effect of predictors is multiplicative and constant over time. The model is semi-parametric, not requiring a specific baseline hazard function, making it suitable for censored data (see [Wikipedia](https://en.wikipedia.org/wiki/Cox_proportional_hazards_model)).

Participants are encouraged to:
- Build custom features.
- Draw inspiration from the provided benchmarks.
- Develop their own models to improve prediction accuracy.
