# Project: Titanic Survival Prediction

This project predicts Titanic passenger survival using `scikit-learn`. It showcases a complete data science workflow, building robust `Pipelines` for preprocessing. It compares two tuned models, `RandomForestClassifier` and `LogisticRegression`, both achieving over 80% accuracy.

---

## 🎯 Project Goals

* Build a `scikit-learn` model to solve a classification problem.
* Implement a `Pipeline` to combine complex preprocessing steps and a model.
* Use `GridSearchCV` for hyperparameter tuning.
* Train, evaluate, and compare two different classifiers (Random Forest and Logistic Regression).
* Interpret model results by visualizing feature importances and coefficients.

---

## 🛠️ Workflow & Pipeline

The core of this project is a `scikit-learn` `Pipeline` that automates the entire preprocessing and modeling workflow.

1.  **Data Cleaning:** The dataset is loaded from `seaborn`, and irrelevant or sparse columns (`deck`, `embark_town`, `alive`, `embarked`) are dropped.
2.  **Preprocessing:** A `ColumnTransformer` is used to apply different transformations to different feature types:
    * **Numerical Features** (`age`, `fare`, `pclass`, `sibsp`, `parch`): Missing values are imputed using the `median`, and all features are scaled using `StandardScaler`.
    * **Categorical Features** (`sex`, `class`, `who`): Missing values are imputed using the `most_frequent` value, and features are then one-hot encoded.
3.  **Modeling:** The preprocessed data is fed directly into a classifier.
4.  **Tuning:** The entire `Pipeline` (preprocessing + model) is wrapped in a `GridSearchCV` to find the best hyperparameters for the classifier.

---

## 📊 Results

Both models were trained and tuned using 5-fold stratified cross-validation. The Logistic Regression model achieved a slightly higher accuracy on the held-out test set.

| Model | Test Set Accuracy |
| :--- | :--- |
| **Random Forest** | 80.22% |
| **Logistic Regression**| 82.09% |

Feature importance plots and confusion matrices for both models are available in the notebook.

---

## 🚀 How to Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/your-username/titanic-survival-prediction.git](https://github.com/your-username/titanic-survival-prediction.git)
    cd titanic-survival-prediction
    ```
2.  It is recommended to create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required libraries:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn jupyter
    ```
4.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
5.  Open `Titanic_Survival_Prediction.ipynb` and run the cells sequentially.

---

## 📚 Libraries Used

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
