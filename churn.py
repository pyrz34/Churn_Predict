# =====================================================
# Telco Churn - Uçtan Uca Modelleme (Hatasız, Tek Blok)
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, VotingClassifier)

# XGBoost / LightGBM varsa kullan, yoksa otomatik atla
HAS_XGB, HAS_LGBM = True, True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LGBM = False

RANDOM_STATE = 42

# -----------------------------
# 1) Veri Yükleme ve Temizleme
# -----------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")

# ID kolonu model için gereksiz
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# TotalCharges string -> numeric, çevrilemeyenleri NaN yap ve median ile doldur (inplace kullanmadan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# --------------------------------
# 2) Feature Engineering & Encoding
# --------------------------------
def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Yeni değişkenler
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-12", "12-24", "24-48", "48-60", "60-72"],
        include_lowest=True
    ).astype(str)
    df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Hedefi 0/1'e çevir
    if df["Churn"].dtype == "O":
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Diğer binary object kolonları label encode (Churn hariç)
    binary_obj_cols = [c for c in df.columns
                       if df[c].dtype == "O" and df[c].nunique() == 2 and c != "Churn"]
    le = LabelEncoder()
    for c in binary_obj_cols:
        df[c] = le.fit_transform(df[c])

    # Geri kalan object kolonları one-hot (tenure_group dahil)
    ohe_cols = [c for c in df.columns if df[c].dtype == "O" and c != "Churn"]
    if len(ohe_cols) > 0:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    return df

df = preprocess_for_model(df)

# -----------------------------
# 3) X, y ve Train/Test Split
# -----------------------------
y = df["Churn"].astype(int)
X = df.drop(columns=["Churn"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# -----------------------------
# 4) Standardization (sadece X)
# -----------------------------
def scale_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()

    num_cols = X_train_s.select_dtypes(include=["int64", "float64"]).columns
    # (Tüm numerikleri ölçekliyoruz; ağaç modellerine zarar vermez, KNN/SVM/LogReg için faydalıdır)
    X_train_s[num_cols] = scaler.fit_transform(X_train_s[num_cols])
    X_test_s[num_cols] = scaler.transform(X_test_s[num_cols])
    return X_train_s, X_test_s

X_train_s, X_test_s = scale_numeric(X_train, X_test)

# -----------------------------
# 5) Base Modeller (CV ile skor)
# -----------------------------
def base_models(X, y, scoring="roc_auc", cv=3):
    print("Base Models (CV)".center(40, "-"))

    models = [
        ("LR",  LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("KNN", KNeighborsClassifier()),
        ("SVC", SVC(probability=True, random_state=RANDOM_STATE)),
        ("CART", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ("RF",  RandomForestClassifier(random_state=RANDOM_STATE)),
        ("ADA", AdaBoostClassifier(random_state=RANDOM_STATE)),
        ("GBM", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]
    if HAS_XGB:
        models.append(("XGB", XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, use_label_encoder=False)))
    if HAS_LGBM:
        models.append(("LGBM", LGBMClassifier(random_state=RANDOM_STATE)))

    for name, clf in models:
        cv_res = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"{name}: {scoring} = {cv_res['test_score'].mean():.4f}")

base_models(X_train_s, y_train, scoring="roc_auc", cv=3)

# ---------------------------------------
# 6) Hiperparametre Optimizasyonu (Grid)
# ---------------------------------------
def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("\nHyperparameter Optimization".center(50, "-"))
    best_models = {}

    grids = [
        ("KNN", KNeighborsClassifier(),
         {"n_neighbors": list(range(2, 51)),
          "weights": ["uniform", "distance"],
          "p": [1, 2]}),

        ("CART", DecisionTreeClassifier(random_state=RANDOM_STATE),
         {"max_depth": list(range(2, 21)),
          "min_samples_split": [2, 5, 10, 20],
          "min_samples_leaf": [1, 2, 4, 8]}),

        ("RF", RandomForestClassifier(random_state=RANDOM_STATE),
         {"n_estimators": [200, 300],
          "max_depth": [None, 10, 20],
          "min_samples_split": [2, 5, 10],
          "max_features": ["sqrt", "log2", None]}),

        ("GBM", GradientBoostingClassifier(random_state=RANDOM_STATE),
         {"n_estimators": [200, 300],
          "learning_rate": [0.1, 0.05],
          "max_depth": [2, 3, 4]})
    ]

    if HAS_XGB:
        grids.append(
            ("XGB", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False),
             {"n_estimators": [200, 300],
              "learning_rate": [0.1, 0.05],
              "max_depth": [3, 5, 7],
              "subsample": [0.8, 1.0],
              "colsample_bytree": [0.8, 1.0]})
        )

    if HAS_LGBM:
        grids.append(
            ("LGBM", LGBMClassifier(random_state=RANDOM_STATE),
             {"n_estimators": [300, 500],
              "learning_rate": [0.1, 0.05],
              "num_leaves": [31, 63],
              "colsample_bytree": [0.8, 1.0]})
        )

    for name, clf, param_grid in grids:
        # Önce mevcut modelle bir CV
        before = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"{name} {scoring} (before): {before['test_score'].mean():.4f}")

        # GridSearch
        gs = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring, n_jobs=-1).fit(X, y)
        best_clf = gs.best_estimator_

        after = cross_validate(best_clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"{name} {scoring} (after) : {after['test_score'].mean():.4f}")
        print(f"{name} best params      : {gs.best_params_}\n")

        # En iyi parametrelerle eğitilmiş modeli sakla
        best_models[name] = best_clf

    return best_models

best_models = hyperparameter_optimization(X_train_s, y_train, cv=3, scoring="roc_auc")

# ---------------------------------------------------
# 7) Voting Classifier (Soft) + Test Üzerinde Skorlar
# ---------------------------------------------------
# Elimizdeki en iyi modellerden 2-3 tanesini seçelim (mevcut olanlara göre)
voting_estimators = []
for key in ["KNN", "RF", "LGBM", "GBM", "XGB"]:
    if key in best_models:
        voting_estimators.append((key, best_models[key]))

# En az 2 model olsun
if len(voting_estimators) < 2:
    # Yedek plan: RF ve LR (LR'ı yeniden fit edip ekleyelim)
    voting_estimators = [("RF", best_models["RF"])]
    lr_backup = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(X_train_s, y_train)
    voting_estimators.append(("LR", lr_backup))

voting_clf = VotingClassifier(estimators=voting_estimators, voting="soft", n_jobs=-1)
voting_clf.fit(X_train_s, y_train)

# Test tahminleri
y_pred = voting_clf.predict(X_test_s)

# Olasılık bazlı metrik (ROC-AUC)
if hasattr(voting_clf, "predict_proba"):
    y_proba = voting_clf.predict_proba(X_test_s)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_proba)
else:
    # SVC gibi decision_function varsa kullanılır; ama biz probability=True kullandık
    y_proba = None
    test_roc_auc = None

# -----------------------------
# 8) Raporlama
# -----------------------------
print("\n" + "Test Sonuçları".center(40, "-"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
if test_roc_auc is not None:
    print(f"ROC-AUC  : {test_roc_auc:.4f}")
