import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import math

st.set_page_config(page_title="Sharpner", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "landing"

def go_to_upload():
    st.session_state.page = "upload"

###########################
#      LANDING PAGE       #
###########################
if st.session_state.page == "landing":
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2.8rem; }
            .project-title {
                font-size:2.9em; font-weight:900; color:#B6C7FF;
                letter-spacing:0.04em; text-align:center; margin-bottom:0.13em;
            }
            .subtitle {
                font-size:1.6em; font-weight:400; color:#ffffff; text-align:center; margin-bottom:0.6em;
            }
            .desc {
                text-align:center; font-size:1.16em; font-weight:350; color:#b0bacf; margin-bottom:1.7em;
            }
            .bullets { 
                color: #b0bacf; font-size: 1.13em; font-weight:370; margin: 0 auto 1.8em auto;
                max-width:470px; text-align:left; padding-left:1em;
            }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown(f"<div class='project-title'>Sharpner</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Your AutoML Playground, Reimagined</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='desc'>Upload your dataset, select features and targets, and compare powerful ML models—all in one intuitive, no-code workspace.</div>",
        unsafe_allow_html=True
    )
    st.markdown('''
    <ul class="bullets">
      <li>Build, train, and compare multiple machine learning models at once</li>
      <li>Instant, interactive results—metrics, curves, and confusion matrices</li>
      <li>No programming needed: upload a CSV and get model insights in minutes</li>
    </ul>
    ''', unsafe_allow_html=True)

    center = st.columns([6,2,6])[1]
    with center:
        if st.button("Start", use_container_width=True):
            go_to_upload()

###########################
#    MULTI-MODEL PAGE     #
###########################
if st.session_state.page == "upload":
    st.markdown(
        "<h1 style='color:#d6d6ee; text-align:center; letter-spacing:2px;'>Sharpner</h1>",
        unsafe_allow_html=True
    )
    st.subheader("Upload Your Dataset")
    upload_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if upload_file:
        df = pd.read_csv(upload_file)
        st.subheader("Data & Model Setup")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        columns = df.columns.tolist()
        target = st.selectbox("Select Target Column", columns)
        available_features = [col for col in columns if col != target]
        features = st.multiselect("Select Features", available_features, default=available_features)

        def detect_problem_type(df, target):
            dtype = df[target].dtype
            nunique = df[target].nunique()
            if dtype == "object" or nunique <= 20:
                return "Classification"
            else:
                return "Regression"

        def preprocess_features(df, features):
            X = df[features].copy()
            drop_cols = [col for col in X.columns if X[col].isnull().mean() > 0.8]
            X = X.drop(columns=drop_cols)
            for col in X.columns:
                if X[col].isnull().any():
                    if X[col].dtype == "object":
                        mode_val = X[col].mode()
                        fill_val = mode_val[0] if not mode_val.empty else "unknown"
                        X[col] = X[col].fillna(fill_val)
                    else:
                        X[col] = X[col].fillna(X[col].median())
            for col in X.columns:
                if X[col].dtype == "object":
                    if X[col].nunique() > 50:
                        X = X.drop(columns=[col])
                    else:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
            return X

        def preprocess_target(y):
            if y.dtype == 'object':
                le = LabelEncoder()
                y_enc = le.fit_transform(y.astype(str))
                return y_enc, le
            else:
                y_filled = y.fillna(y.median())
                return y_filled, None

        if target and features:
            if target in features:
                st.error("Target column cannot be in features.")
            else:
                try:
                    working_df = df[features + [target]].copy()
                    if working_df.dropna().shape[0] < 10:
                        st.error("Not enough rows for training after cleaning.")
                    else:
                        X = preprocess_features(working_df, features)
                        y_processed, target_encoder = preprocess_target(working_df[target])
                        problem_type = detect_problem_type(working_df, target)
                        if problem_type == "Regression":
                            ALL_MODELS = ["Linear Regression", "SVM", "KNN", "Decision Tree", "Random Forest"]
                        else:
                            ALL_MODELS = ["Logistic Regression", "SVM", "KNN", "Decision Tree", "Random Forest"]

                        selected_models = st.multiselect(
                            "Select one or more algorithms to train",
                            ALL_MODELS,
                            default=ALL_MODELS[:2]
                        )
                        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, step=0.05, format="%.2f")

                        if st.button("Train Selected Models", use_container_width=True):
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_processed, test_size=test_size, random_state=42
                            )

                            model_defs = {
                                "Linear Regression": LinearRegression(),
                                "SVM": SVR() if problem_type == "Regression" else SVC(random_state=42),
                                "KNN": KNeighborsRegressor() if problem_type == "Regression" else KNeighborsClassifier(),
                                "Decision Tree": DecisionTreeRegressor(random_state=42) if problem_type == "Regression" else DecisionTreeClassifier(random_state=42),
                                "Random Forest": RandomForestRegressor(random_state=42) if problem_type == "Regression" else RandomForestClassifier(random_state=42),
                                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
                            }

                            results = {}
                            preds_test_dict = {}

                            with st.spinner("Training selected models..."):
                                for mdl in selected_models:
                                    model = model_defs[mdl]
                                    model.fit(X_train, y_train)
                                    train_preds = model.predict(X_train)
                                    test_preds = model.predict(X_test)
                                    preds_test_dict[mdl] = test_preds
                                    if problem_type == "Regression":
                                        results[mdl] = {
                                            "train": [r2_score(y_train, train_preds), mean_squared_error(y_train, train_preds), np.sqrt(mean_squared_error(y_train, train_preds))],
                                            "test": [r2_score(y_test, test_preds), mean_squared_error(y_test, test_preds), np.sqrt(mean_squared_error(y_test, test_preds))],
                                            "metrics": ["R2", "MSE", "RMSE"]
                                        }
                                    else:
                                        results[mdl] = {
                                            "train": [
                                                accuracy_score(y_train, train_preds),
                                                precision_score(y_train, train_preds, average="weighted", zero_division=0),
                                                recall_score(y_train, train_preds, average="weighted", zero_division=0),
                                                f1_score(y_train, train_preds, average="weighted", zero_division=0)
                                            ],
                                            "test": [
                                                accuracy_score(y_test, test_preds),
                                                precision_score(y_test, test_preds, average="weighted", zero_division=0),
                                                recall_score(y_test, test_preds, average="weighted", zero_division=0),
                                                f1_score(y_test, test_preds, average="weighted", zero_division=0)
                                            ],
                                            "metrics": ["Accuracy", "Precision", "Recall", "F1"]
                                        }

                            st.subheader("Model Results and Comparison")
                            num_models = len(selected_models)
                            n_per_row = 2

                            for row_idx in range(math.ceil(num_models / n_per_row)):
                                cols = st.columns(n_per_row, gap="large")
                                for i in range(n_per_row):
                                    model_idx = row_idx * n_per_row + i
                                    if model_idx >= num_models:
                                        continue
                                    mdl = selected_models[model_idx]
                                    with cols[i]:
                                        st.markdown(
                                            f"<h4 style='color:#bbbbbb;margin-top:1.3em;font-size:1.13em'>{mdl}</h4>",
                                            unsafe_allow_html=True
                                        )
                                        model_metrics = results[mdl]["metrics"]
                                        metric_arr = np.array([results[mdl]["train"], results[mdl]["test"]])
                                        df = pd.DataFrame(metric_arr, columns=model_metrics, index=["Train", "Test"])
                                        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
                                        
                                        # Line chart
                                        fig, ax = plt.subplots(figsize=(2.9, 1.35))
                                        for j, metric in enumerate(model_metrics):
                                            ax.plot(["Train", "Test"], [results[mdl]["train"][j], results[mdl]["test"][j]],
                                                marker="o", label=metric, linewidth=2)
                                        if problem_type != "Regression":
                                            ax.set_ylim(0, 1.03)
                                        ax.set_xticks(["Train", "Test"])
                                        ax.grid(alpha=0.18, linestyle="--", linewidth=0.7)
                                        box = ax.get_position()
                                        ax.set_position([box.x0, box.y0 + box.height * 0.18, box.width, box.height * 0.82])
                                        ax.legend(fontsize="x-small", ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                                        ax.set_ylabel("Score")
                                        ax.set_title("Metric Trend", fontsize=9, color="#bbbbbb")
                                        st.pyplot(fig, use_container_width=False)

                                        # Confusion matrix for classification
                                        if problem_type == "Classification":
                                            st.markdown("**Confusion Matrix (Test)**")
                                            class_labels = np.unique(y_test)
                                            n_class = len(class_labels)
                                            fig_cm, ax_cm = plt.subplots(figsize=(max(2.1, 0.55*n_class), max(1.4, 0.45*n_class)))
                                            cm = confusion_matrix(y_test, preds_test_dict[mdl], labels=class_labels)
                                            sns.heatmap(
                                                cm, annot=True, fmt="d",
                                                cmap="Blues", ax=ax_cm, cbar=False,
                                                annot_kws={"size": 10 if n_class <= 4 else 8}
                                            )
                                            ax_cm.set_xlabel("Pred", fontsize=9)
                                            ax_cm.set_ylabel("Actual", fontsize=9)
                                            ax_cm.set_xticks(np.arange(n_class)+0.5)
                                            ax_cm.set_yticks(np.arange(n_class)+0.5)
                                            ax_cm.set_xticklabels(class_labels, rotation=0, fontsize=8)
                                            ax_cm.set_yticklabels(class_labels, rotation=0, fontsize=8)
                                            plt.tight_layout(pad=1.2)
                                            st.pyplot(fig_cm, use_container_width=False)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Check your data for missing values, text columns, or insufficient samples.")
    else:
        st.info("Please upload a CSV file to continue.")
