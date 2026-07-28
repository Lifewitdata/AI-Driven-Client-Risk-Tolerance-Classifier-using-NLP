# AI-Driven Client Risk Tolerance Classifier using NLP

## 📖 Overview
This project simulates an NLP and AI solution for the TD Wealth AI2 team. It analyzes unstructured advisor meeting notes to automatically classify a client's stated Risk Tolerance. It then compares this NLP-extracted risk against the client's actual portfolio composition to identify compliance risks and rebalancing opportunities.

## 🎯 Business Problem
Wealth Management compliance teams spend thousands of manual hours reviewing advisor notes to ensure portfolios match client risk profiles. Misaligned portfolios pose massive regulatory and financial risks if markets turn. This project automates that audit process using NLP.

## 🛠 Tech Stack & Methodology
* **Language:** Python (Pandas, NLTK, Scikit-learn)
* **NLP Preprocessing:** Regex cleaning, tokenization, NLTK stop-word removal
* **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization
* **Machine Learning:** Multi-Class Classification (Logistic Regression, Random Forest)
* **Business Logic:** Custom rule-based engine to compare NLP risk vs. Portfolio Risk and flag mismatches.

## 📈 Key Outcomes & Business Value
* **Model Performance:** Logistic Regression achieved ~95% accuracy on multi-class risk classification from text data.
* **Compliance Automation:** Successfully identified portfolio risk mismatches (both "Over-invested" and "Under-invested" clients) from synthetic data.
* **Actionable Output:** Generated a prioritized compliance action plan flagging specific clients for urgent rebalancing or growth opportunities, output as a CSV for Tableau dashboarding.
