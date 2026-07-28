<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=26&duration=3000&pause=1000&color=8E24AA&center=true&vCenter=true&multiline=true&repeat=true&width=850&height=90&lines=TD+Wealth+%7C+AI-Driven+Risk+Tolerance+Classifier;Turning+Advisor+Notes+into+Compliance+Signals;NLP+%2B+TF-IDF+%2B+Logistic+Regression" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLP-NLTK-2E7D32?style=for-the-badge)
![TFIDF](https://img.shields.io/badge/Vectorization-TF--IDF-F9A825?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Tableau](https://img.shields.io/badge/Downstream-Tableau-E97627?style=for-the-badge&logo=tableau&logoColor=white)

![Accuracy](https://img.shields.io/badge/Accuracy-~95%25-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

<img src="https://user-images.githubusercontent.com/74038190/213910845-af37a709-8995-40d6-be59-724526e3c3d7.gif" width="450">

</div>

---

## 🧭 Table of Contents

- [Overview](#-overview)
- [Business Problem](#-business-problem)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack--methodology)
- [NLP Pipeline](#-nlp-pipeline)
- [Modeling Approach](#-modeling-approach)
- [Compliance Mismatch Engine](#-compliance-mismatch-engine)
- [Key Outcomes](#-key-outcomes--business-value)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Sample Outputs](#-sample-outputs)
- [Author](#-author)

---

## 📖 Overview

> This project simulates an **NLP + AI compliance solution** for the TD Wealth Analytics, Insights & Artificial Intelligence (AI2) team.

It reads **unstructured advisor meeting notes**, automatically classifies each client's *stated* Risk Tolerance (Conservative / Moderate / Aggressive), and then cross-checks that against what the client's **portfolio actually holds** — surfacing compliance risks and rebalancing opportunities that would otherwise take a human reviewer hours to find.

<div align="center">

```mermaid
flowchart LR
    A[📝 Advisor Meeting Notes] --> B[🧹 NLP Preprocessing]
    B --> C[🔢 TF-IDF Vectorization]
    C --> D[🤖 Multi-Class Classifier]
    D --> E{⚖️ Stated Risk vs Portfolio Risk}
    E -->|Match| F[✅ No Action]
    E -->|Mismatch| G[🚨 Compliance Action Plan]

    style A fill:#8E24AA,color:#fff
    style B fill:#2E7D32,color:#fff
    style C fill:#F9A825,color:#000
    style D fill:#3776AB,color:#fff
    style E fill:#455A64,color:#fff
    style F fill:#43A047,color:#fff
    style G fill:#D32F2F,color:#fff
```

</div>

---

## 🎯 Business Problem

Wealth Management compliance teams spend **thousands of manual hours** reading advisor notes to verify that a client's portfolio actually matches their stated risk appetite. A misaligned portfolio isn't just an inconvenience — it's a **regulatory and financial exposure** if markets turn and a client's holdings don't reflect what they agreed to.

| Manual Compliance Review ❌ | NLP-Automated Audit ✅ |
|---|---|
| Analysts read notes one client at a time | Model scores every client's notes in seconds |
| Risk mismatches found reactively, after complaints | Mismatches surfaced proactively, before they escalate |
| No standardized "stated risk" extraction | Consistent, model-driven risk label for every client |
| Output buried in case files | Clean CSV ready for Tableau dashboards |

---

## 🏗 Architecture

```mermaid
graph TD
    subgraph Synthetic Data
        D1[client_profiles - 5,000 clients]
        D2[actual_portfolio_risk]
        D3[advisor_notes - free text]
    end

    subgraph NLP Layer
        N1[Lowercase + Regex Cleaning]
        N2[Tokenization]
        N3[NLTK Stop-word Removal]
        N4[TF-IDF Vectorizer - top 100 terms]
    end

    subgraph Modeling Layer
        M1[Logistic Regression]
        M2[Random Forest]
        M3[Best Model Selection]
    end

    subgraph Compliance Engine
        E1[Map Risk to Rank: Conservative=1, Moderate=2, Aggressive=3]
        E2[Compare NLP Risk Rank vs Portfolio Risk Rank]
        E3[Flag: Over-invested / Under-invested / Matched]
        E4[Generate Advisor Action]
    end

    D3 --> N1 --> N2 --> N3 --> N4
    N4 --> M1
    N4 --> M2
    M1 --> M3
    M2 --> M3
    M3 --> E1
    D2 --> E1
    E1 --> E2 --> E3 --> E4
    E4 --> OUT[nlp_risk_mismatch_action_plan.csv]
```

---

## 🛠 Tech Stack & Methodology

<div align="center">

| Layer | Tools |
|:---|:---|
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) Pandas, NumPy |
| **NLP Preprocessing** | Regex cleaning, tokenization, ![NLTK](https://img.shields.io/badge/-NLTK-2E7D32?style=flat-square) stop-word removal |
| **Feature Extraction** | ![TF-IDF](https://img.shields.io/badge/-TF--IDF-F9A825?style=flat-square) Term Frequency–Inverse Document Frequency |
| **Machine Learning** | ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) Multi-class Logistic Regression & Random Forest |
| **Business Logic** | Custom rule-based engine: NLP Risk vs Portfolio Risk mismatch detection |
| **Downstream** | ![Tableau](https://img.shields.io/badge/-Tableau-E97627?style=flat-square&logo=tableau&logoColor=white) CSV export for BI dashboarding |

</div>

---

## 🔤 NLP Pipeline

The heart of this project is turning **messy, free-text advisor notes** into a clean, model-ready signal.

**Preprocessing steps applied to every note:**

1. 🔡 Lowercase all text
2. ✂️ Strip punctuation and special characters via regex (`[^a-z\s]`)
3. 🪙 Tokenize into individual words
4. 🚫 Remove English stop-words using **NLTK**
5. 🔗 Rejoin cleaned tokens into a normalized string

```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)
```

A **word cloud** of "Aggressive" risk notes is generated during EDA to sanity-check that the vocabulary genuinely separates by risk class (growth, equities, upside-type language vs. capital-preservation language).

**Vectorization:** cleaned notes are converted into numeric features with `TfidfVectorizer(max_features=100)`, capping the vocabulary to the 100 most informative terms so the model focuses on signal, not noise.

---

## 🤖 Modeling Approach

<div align="center">

| Model | Role |
|:---|:---|
| **Logistic Regression** 🏆 | Final model — best fit for sparse, high-dimensional TF-IDF text data |
| Random Forest | Benchmark comparison, non-linear alternative |

</div>

- Target: 3-class `stated_risk_target` (Conservative / Moderate / Aggressive)
- Stratified 80/20 train/test split to preserve class balance
- Evaluated with `classification_report` (precision, recall, F1 per class) and a **confusion matrix heatmap**

```mermaid
graph LR
    A[TF-IDF Features] --> B[Logistic Regression]
    A --> C[Random Forest]
    B --> D{Compare Accuracy}
    C --> D
    D --> E[Logistic Regression Selected: ~95% Accuracy]
```

---

## ⚖️ Compliance Mismatch Engine

Once every client has an NLP-predicted risk label, it's ranked against their **actual portfolio risk** to detect misalignment:

| Risk Level | Rank |
|:---|:---:|
| Conservative | 1 |
| Moderate | 2 |
| Aggressive | 3 |

```mermaid
flowchart TD
    A[NLP Risk Rank vs Portfolio Risk Rank] --> B{Compare}
    B -->|NLP Rank > Portfolio Rank| C["🟢 Under-invested\n(Client wants MORE risk than portfolio has)"]
    B -->|NLP Rank < Portfolio Rank| D["🔴 Over-invested\n(Client wants LESS risk than portfolio has)"]
    B -->|Equal| E["✅ Matched — No Action"]
    C --> F[OPPORTUNITY: Discuss adding growth equities]
    D --> G[URGENT: Rebalance — shift equity to fixed income]
```

This is the piece that turns a classification model into a **compliance tool**: it doesn't just say "this client is Aggressive," it says *"this client is Aggressive but their portfolio is Conservative — here's what to do about it."*

---

## 📈 Key Outcomes & Business Value

<div align="center">

| Metric | Result |
|:---|:---:|
| 🎯 Model Accuracy | **~95%** (Logistic Regression, multi-class) |
| 🚨 Compliance Automation | Both **Over-invested** and **Under-invested** clients identified from synthetic data |
| 📋 Actionable Output | Prioritized compliance action plan, exported for Tableau dashboarding |

</div>

**Sample recommendation logic:**

| Compliance Status | Advisor Action |
|:---|:---|
| Over-invested (wants LESS risk than held) | **URGENT** — Rebalance portfolio, move equity to fixed income |
| Under-invested (wants MORE risk than held) | **OPPORTUNITY** — Discuss adding growth equities |
| Matched | No action needed |

---

## 📁 Repository Structure

```
📦 td-wealth-nlp-risk-classifier
 ┣ 📓 AI_Driven_Client_Risk_Tolerance_Classifier_using_NLP.ipynb
 ┣ 📄 nlp_risk_mismatch_action_plan.csv     # Final output for Tableau
 ┣ 📁 data/
 ┃ ┗ synthetic_client_notes.csv             # 5,000 clients + advisor notes
 ┗ 📜 README.md
```

---

## 🚀 How to Run

```bash
# 1️⃣ Install dependencies
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud

# 2️⃣ Download NLTK stopwords (handled automatically in-notebook)
python -c "import nltk; nltk.download('stopwords')"

# 3️⃣ Run the notebook end-to-end
jupyter notebook AI_Driven_Client_Risk_Tolerance_Classifier_using_NLP.ipynb
```

<details>
<summary>⚙️ <b>What the notebook produces (click to expand)</b></summary>

- Word clouds per risk category
- TF-IDF feature matrix preview
- Classification reports for Logistic Regression & Random Forest
- Confusion matrix heatmap
- Mismatch distribution charts (count + equity % boxplot)
- `nlp_risk_mismatch_action_plan.csv` — final compliance output

</details>

---

## 🖼 Sample Outputs

<div align="center">

| Word Cloud (Aggressive Notes) | Confusion Matrix | Mismatch Distribution |
|:---:|:---:|:---:|
| ☁️ top terms by risk class | 🔵 3x3 heatmap | 📊 Over vs Under-invested counts |

*(Generated live inside the notebook — see `AI_Driven_Client_Risk_Tolerance_Classifier_using_NLP.ipynb`)*

</div>

---

## 👤 Author

<div align="center">

Built as an end-to-end simulation of an **NLP-driven compliance & risk analytics** use case — from raw advisor notes to a Tableau-ready action plan.

![Profile Views](https://komarev.com/ghpvc/?username=td-wealth-nlp-risk-classifier&color=8E24AA&style=flat-square)

<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="100%">

⭐ **If this project was useful, consider starring the repo!** ⭐

</div>
