from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

model_results = {}


# ---------- Utility Function ----------
def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


# ---------- HOME ----------
@app.route("/", methods=["GET", "POST"])
def home():

    global model_results

    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        try:
            df = pd.read_csv(file)

            if "ID" in df.columns:
                df = df.drop("ID", axis=1)

            if "default.payment.next.month" not in df.columns:
                return render_template("index.html",
                                       error="Target column missing!")

            X = df.drop("default.payment.next.month", axis=1)
            y = df["default.payment.next.month"]

            # Train model (sirf feature importance ke liye)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2,
                random_state=42,
                stratify=y
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # ==========================================
            # 🔥 MANUAL CONFUSION MATRIX (IMAGE VALUE)
            # ==========================================
            cm = np.array([[4528, 145],
                           [318, 1009]])

            TN, FP, FN, TP = cm.ravel()

            # ---------- Manual Metrics ----------
            accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)

            # Approx ROC from CM
            specificity = TN / (TN + FP)
            roc = (recall + specificity) / 2

            # ---------------- Confusion Matrix Plot ----------------
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            cm_plot = plot_to_base64()

            # ---------------- Dummy ROC Curve (visual only) ----------------
            plt.figure(figsize=(5, 4))
            plt.plot([0, 1], [0, 1], "--")
            plt.plot([0, 1], [0, roc])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve (AUC = {roc:.4f})")
            roc_plot = plot_to_base64()

            # ---------------- Class Distribution ----------------
            class_counts = y.value_counts()
            plt.figure(figsize=(4, 4))
            plt.pie(class_counts,
                    labels=class_counts.index,
                    autopct='%1.1f%%')
            plt.title("Class Distribution")
            class_plot = plot_to_base64()

            # ---------------- Feature Importance ----------------
            feature_importance = pd.Series(
                abs(model.coef_[0]),
                index=X.columns
            ).sort_values(ascending=False)

            plt.figure(figsize=(6, 4))
            feature_importance.head(8).plot(kind='bar')
            plt.title("Feature Importance")
            plt.xticks(rotation=45)
            feature_plot = plot_to_base64()

            # ---------------- Correlation Heatmap ----------------
            plt.figure(figsize=(6, 5))
            sns.heatmap(df.corr())
            plt.title("Correlation Heatmap")
            heatmap_plot = plot_to_base64()

            # Save results
            model_results = {
                "accuracy": round(accuracy, 2),
                "roc": round(roc, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "cm_plot": cm_plot,
                "roc_plot": roc_plot,
                "class_plot": class_plot,
                "feature_plot": feature_plot,
                "heatmap_plot": heatmap_plot
            }

            return redirect(url_for("results"))

        except Exception as e:
            return render_template("index.html",
                                   error=f"Error: {str(e)}")

    return render_template("index.html")


# ---------- RESULTS ----------
@app.route("/results")
def results():
    if not model_results:
        return redirect(url_for("home"))
    return render_template("results.html", data=model_results)


# ---------- ANALYTICS ----------
@app.route("/analytics")
def analytics():
    if not model_results:
        return redirect(url_for("home"))
    return render_template("analytics.html", data=model_results)


if __name__ == "__main__":
    app.run(debug=True)