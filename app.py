from flask import Flask, render_template, request, redirect, session, url_for
import sqlite3
import joblib
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
app.secret_key = "heart_disease_secret_key"

# ---------------- LOAD MODELS ----------------
binary_model = joblib.load("data/model_binary.pkl")
severity_model = joblib.load("data/model_severity.pkl")
scaler = joblib.load("data/scaler.pkl")

# ---------------- DATABASE ----------------


def get_db():
    return sqlite3.connect("database.db")


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        age INTEGER,
        binary_result TEXT,
        severity_result TEXT,
        no_disease_prob REAL,
        disease_prob REAL,
        severity_mild REAL,
        severity_moderate REAL,
        severity_high REAL,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()


init_db()

# ---------------- AUTH ----------------


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users VALUES (NULL,?,?,?)",
                (name, email, password)
            )
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email already exists")

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- DASHBOARD ----------------


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", name=session["user_name"])

# ---------------- PREDICTION ----------------


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        feature_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        # ✅ SAFE INPUT ORDER
        values = [float(request.form[f]) for f in feature_names]

        input_df = pd.DataFrame([values], columns=feature_names)
        scaled = scaler.transform(input_df)

        # ---------- BINARY ----------
        binary_pred = binary_model.predict(scaled)[0]
        binary_proba = binary_model.predict_proba(scaled)[0]

        no_disease_prob = round(binary_proba[0] * 100, 2)
        disease_prob = round(binary_proba[1] * 100, 2)

        binary_result = "Disease likely" if binary_pred == 1 else "No disease"

        severity_result = None
        severity_probs = None
        mild = moderate = high = None

        # ---------- SEVERITY (ONLY IF DISEASE) ----------
        if binary_pred == 1:
            severity_proba = severity_model.predict_proba(scaled)[0]

            class_labels = severity_model.classes_
            prob_map = dict(zip(class_labels, severity_proba))

            mild = round(prob_map.get(1, 0) * 100, 2)
            moderate = round(prob_map.get(2, 0) * 100, 2)
            high = round((prob_map.get(3, 0) + prob_map.get(4, 0)) * 100, 2)

            severity_probs = {
                "mild": mild,
                "moderate": moderate,
                "high": high
            }

            # ✅ FINAL SEVERITY LABEL = HIGHEST PROBABILITY
            severity_result = max(
                {"Mild": mild, "Moderate": moderate, "High Risk": high},
                key=lambda k: {"Mild": mild, "Moderate": moderate, "High Risk": high}[k]
            )

        # ---------- SAVE HISTORY ----------
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO history (
            user_id, age, binary_result, severity_result,
            no_disease_prob, disease_prob,
            severity_mild, severity_moderate, severity_high,
            date
        )
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            session["user_id"],
            int(values[0]),
            binary_result,
            severity_result if severity_result else "N/A",
            no_disease_prob,
            disease_prob,
            mild,
            moderate,
            high,
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            binary_result=binary_result,
            no_disease_prob=no_disease_prob,
            disease_prob=disease_prob,
            severity_result=severity_result,
            severity_probs=severity_probs
        )

    return render_template("predict.html")

# ---------------- HISTORY ----------------


@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    SELECT id, age, binary_result, severity_result, date
    FROM history
    WHERE user_id=?
    ORDER BY id DESC
    """, (session["user_id"],))
    rows = cur.fetchall()
    conn.close()

    return render_template("history.html", rows=rows)
# ---------------- VIEW PREDICTION DETAILS ----------------


@app.route("/view_prediction/<int:pid>")
def view_prediction(pid):
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    SELECT binary_result, severity_result,
           no_disease_prob, disease_prob,
           severity_mild, severity_moderate, severity_high
    FROM history
    WHERE id=? AND user_id=?
    """, (pid, session["user_id"]))

    row = cur.fetchone()
    conn.close()

    if not row:
        return redirect(url_for("history"))

    binary_result, severity_result, no_p, dis_p, mild, mod, high = row

    severity_probs = None
    if binary_result == "Disease likely":
        severity_probs = {
            "mild": mild,
            "moderate": mod,
            "high": high
        }

    return render_template(
        "result.html",
        binary_result=binary_result,
        no_disease_prob=no_p,
        disease_prob=dis_p,
        severity_result=None if severity_result == "N/A" else severity_result,
        severity_probs=severity_probs
    )


# ---------------- RUN ----------------


if __name__ == "__main__":
    app.run(debug=True)
