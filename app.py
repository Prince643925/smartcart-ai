from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import pandas as pd
import numpy as np
import os
import uuid

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

SESSION_STORE = {}

# ================= GPT ================= #
def get_gpt_insights(summary):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Explain this segmentation:\n{summary}"}]
        )
        return response.choices[0].message.content
    except:
        return "AI insights unavailable"


# ================= HOME ================= #
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<html>
<head>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Segoe UI', sans-serif;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.container-box {
    text-align: center;
    max-width: 700px;
    width: 100%;
}

.title {
    font-size: 60px;
    font-weight: bold;
}

.subtitle {
    color: #b0bec5;
    margin-bottom: 30px;
}

.upload-box {
    background: #1c1c1c;
    padding: 20px;
    border-radius: 15px;
}

input[type="file"] {
    background: #2a2a2a;
    color: white;
    border: none;
}

.btn-analyze {
    background: #00c6ff;
    border: none;
    padding: 12px 30px;
    font-size: 18px;
    border-radius: 10px;
}

.btn-analyze:hover {
    background: #00a6d6;
}

/* 🔄 FLOAT ANIMATION */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

/* ✨ GLOW EFFECT */
.logo {
    animation: float 3s ease-in-out infinite;
    filter: drop-shadow(0 0 10px rgba(0,198,255,0.6));
    transition: 0.3s;
}

/* 🔥 HOVER EFFECT */
.logo:hover {
    filter: drop-shadow(0 0 20px rgba(0,198,255,1));
    transform: scale(1.1);
}

</style>
</head>

<body>

<div class="container-box">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" width="90" class="logo" style="margin-bottom:20px;">

    <div class="title">🛒 SmartCart AI</div>
    <div class="subtitle">Customer Intelligence Platform</div>

    <div class="upload-box">

        <form action="/upload" method="post" enctype="multipart/form-data">
            
            <input type="file" name="file" required class="form-control mb-3">
            
            <button class="btn btn-analyze">🚀 Analyze Customers</button>

        </form>

    </div>

    <p class="mt-4 text-secondary">
        AI-powered decision system for modern businesses
    </p>

    <p class="mt-1" style="font-size:14px; color:#90caf9;">
        Developed by Prince Kumar(23ERACS211) & Prince Kumar(22ERACS028) 🚀

</div>

</body>
</html>
""")


# ================= UPLOAD ================= #
@app.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(None),
    columns: list[str] = Form(None)
):
    try:
        session_id = request.cookies.get("session_id")

        if file:
            df = pd.read_csv(file.file)
            session_id = str(uuid.uuid4())
            SESSION_STORE[session_id] = df
        elif session_id in SESSION_STORE:
            df = SESSION_STORE[session_id]
        else:
            return RedirectResponse(url="/", status_code=303)

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "")
            .str.replace("_", "")
        )

        column_map = {
            "annualincome": ["annualincome", "income", "salary", "earnings"],
            "spendingscore": ["spendingscore", "spending", "score"],
            "frequency": ["frequency", "freq", "transactions", "purchases"],
            "date": ["date", "transactiondate", "orderdate"]
        }

        # Apply mapping
        for standard, variations in column_map.items():
            for col in df.columns:
                if col in variations:
                    df.rename(columns={col: standard}, inplace=True)

        # ================= SAFE COLUMN GUARANTEE ================= #
        def safe_column(col, low, high):
            if col not in df.columns:
                df[col] = np.random.randint(low, high, size=len(df))

        safe_column("annualincome", 20, 100)
        safe_column("spendingscore", 1, 100)
        safe_column("frequency", 1, 10)

        # Convert all numeric safely
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
            
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.shape[1] == 0:
            return HTMLResponse("<h2 style='color:white'>❌ No numeric data found in CSV</h2>")

        all_cols = list(numeric_df.columns)

        selected_cols = columns if columns else all_cols

        if len(selected_cols) < 2:
            return HTMLResponse("<h2 style='color:white'>⚠️ Select at least 2 columns</h2>")

        numeric_df = numeric_df[selected_cols]
        numeric_df = numeric_df.fillna(0)


        # ================= FEATURE ENGINEERING (ADD HERE) ================= #
        from datetime import datetime

        today = datetime.now()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["recency"] = (today - df["date"]).dt.days.fillna(0)
        else:
             df["recency"] = np.random.randint(1, 100, size=len(df))
        
        df["total_spent"] = df["annualincome"] * np.random.uniform(0.1, 0.5, size=len(df))

        if "frequency" in df.columns:
            df["avg_purchase"] = df["total_spent"] / (df["frequency"] + 1)
        else:
            df["avg_purchase"] = df["total_spent"]

        # Also add to numeric_df (IMPORTANT)
        numeric_df["recency"] = df["recency"]
        numeric_df["total_spent"] = df["total_spent"]
        numeric_df["avg_purchase"] = df["avg_purchase"]

        # ================= CUSTOMER VALUE SCORE ================= #
        if "frequency" in numeric_df.columns and "spendingscore" in numeric_df.columns:
            numeric_df["engagement_score"] = numeric_df["frequency"] * numeric_df["spendingscore"]
        elif "spendingscore" in numeric_df.columns:
            numeric_df["engagement_score"] = numeric_df["spendingscore"]
        else:
            numeric_df["engagement_score"] = np.ones(len(numeric_df))
        

        eng_max = numeric_df["engagement_score"].max() or 1
        engagement_norm = numeric_df["engagement_score"] / eng_max

        spending_max = numeric_df["spendingscore"].max() or 1
        income_max = numeric_df["annualincome"].max() or 1

        spending = numeric_df["spendingscore"] / spending_max
        income = numeric_df["annualincome"] / income_max

        
        if "frequency" in numeric_df.columns:
            frequency = numeric_df["frequency"] / (numeric_df["frequency"].max() or 1)
        else:
            frequency = 0.5

        recency_max = numeric_df["recency"].max() or 1
        recency_norm = 1 - (numeric_df["recency"] / recency_max)


        numeric_df["value_score"] = (
            0.25 * spending +
            0.2 * income +
            0.25 * frequency +
            0.15 * recency_norm +
            0.15 * engagement_norm
        )


        scaler = StandardScaler()
        X = scaler.fit_transform(numeric_df)

        best_score = -1
        best_labels = None
        best_k = 2

        for k in range(2, min(6, max(3, len(X)//2))):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X)

            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

        if best_labels is None:
            best_labels = np.zeros(len(X), dtype=int)

        df["cluster"] = best_labels
        df["value_score"] = numeric_df["value_score"]
        summary_df = df.groupby("cluster").mean(numeric_only=True)

        

        gpt_text = get_gpt_insights(summary_df.to_string())

        # PCA SAFE
        if X.shape[1] >= 2:
            X_vis = PCA(n_components=2).fit_transform(X)
        else:
            X_vis = np.column_stack((X[:, 0], np.zeros(len(X))))

        # ================= SEGMENTS ================= #
        segment_cards = ""

        # ✅ choose smart column
        target_col = "value_score"

        values = summary_df[target_col]

        # percentile-based segmentation (always balanced)
        low_th = values.quantile(0.33)
        high_th = values.quantile(0.66)


        segments = {
            "💎 VIP Customers": [],
            "🛍️ Regular Customers": [],
            "⚠️ At Risk Customers": []
        }

        for c in summary_df.index:
            spend = summary_df.loc[c][target_col]

            if spend >= high_th:
                segments["💎 VIP Customers"].append(c)

            elif spend <= low_th:
                segments["⚠️ At Risk Customers"].append(c)

            else:
                segments["🛍️ Regular Customers"].append(c)
                

        segment_cards = ""

        for tag, cluster_list in segments.items():

            if not cluster_list:
                continue

            if "VIP" in tag:
                color = "#00c6ff"
                action = "Focus on retention, loyalty programs & premium experience"

            elif "Risk" in tag:
                color = "#ff4d4d"
                action = "Run discounts, re-engagement campaigns"

            else:
                color = "#00c851"
                action = "Upsell & cross-sell opportunities"

        

            segment_cards += f"""
            <div class="col-md-4">
                <div class="card p-3">
                    <h4>{tag}</h4>
                    <p><b>Clusters:</b> {cluster_list}</p>
                    <p><b>Action:</b> {action}</p>
                </div>
            </div>
            """


        preview = df.head().to_html(classes="table table-dark")
        score = round(best_score, 2)

        # ================= CHART DATA ================= #
        # Bar chart (avg value_score per cluster)
        bar_x = summary_df.index.tolist()
        bar_y = summary_df[target_col].tolist()

        # Pie chart (cluster distribution)
        cluster_counts = df["cluster"].value_counts().sort_index()
        pie_labels = cluster_counts.index.tolist()
        pie_values = cluster_counts.values.tolist()


        # Trend (if time column exists)
        trend_x = []
        trend_y = []

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            trend_df = df.groupby("date")[target_col].mean()
            trend_x = trend_df.index.astype(str).tolist()
            trend_y = trend_df.values.tolist()

    except Exception as e:
        print("🔥 ERROR:", e)
        raise e


    response = HTMLResponse(f"""
<html>
<head>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<style>
body {{
    background:#0f2027;
    color:#ffffff;
    font-family: 'Segoe UI', sans-serif;
}}

.card {{
    background:#1c1c1c;
    margin-top:20px;
    padding:20px;
    border-radius:15px;
    color:#ffffff;
}}

h1, h2, h3, h4, h5 {{
    color:#ffffff;
}}

.table {{
    color:#ffffff !important;
}}

.table th {{
    color:#00c6ff !important;
    background:#111 !important;
}}

.table td {{
    color:#e6e6e6 !important;
}}

.table-dark {{
    background:#1c1c1c !important;
}}

select {{
    background:#1c1c1c !important;
    color:white !important;
    border:1px solid #444 !important;
}}

option {{
    background:#1c1c1c;
    color:white;
}}

.btn-info {{
    background:#00c6ff;
    border:none;
}}

.btn-light {{
    background:#e6e6e6;
    color:black;
}}

#chart {{
    background:#1c1c1c;
}}
</style>

</head>

<body>

<div class="container mt-5">

<h1>📊 SmartCart Business Dashboard</h1>

<div class="row">
    <div class="col-md-4"><div class="card">Customers<br><h2>{len(df)}</h2></div></div>
    <div class="col-md-4"><div class="card">Segments<br><h2>{best_k}</h2></div></div>
    <div class="col-md-4"><div class="card">Model Quality<br><h2>{score}</h2></div></div>
</div>

<div class="card">
<h3>🧠 Executive Summary</h3>
<p>Your customers are segmented into {best_k} groups.</p>
</div>

<div class="row">

    <div class="col-md-6">
        <div id="scatter" class="card"></div>
    </div>

    <div class="col-md-6">
        <div id="bar" class="card"></div>
    </div>

</div>

<div class="row mt-3">

    <div class="col-md-6">
        <div id="pie" class="card"></div>
    </div>

    <div class="col-md-6">
        <div id="trend" class="card"></div>
    </div>

</div>

<script>

// SCATTER
Plotly.newPlot('scatter', [{{
    x: {X_vis[:,0].tolist()},
    y: {X_vis[:,1].tolist()},
    mode: 'markers',
    marker: {{
        color: {best_labels.tolist()},
        colorscale: 'Viridis',
        size:10
    }}
}}], {{
    title: "Customer Clusters",
    paper_bgcolor: '#1c1c1c',
    plot_bgcolor: '#1c1c1c',
    font: {{ color: '#ffffff' }}
}});

// BAR
Plotly.newPlot('bar', [{{
    x: {bar_x},
    y: {bar_y},
    type: 'bar'
}}], {{
    title: "Avg Value Score",
    paper_bgcolor: '#1c1c1c',
    plot_bgcolor: '#1c1c1c',
    font: {{ color: '#ffffff' }}
}});

// PIE
Plotly.newPlot('pie', [{{
    labels: {pie_labels},
    values: {pie_values},
    type: 'pie'
}}], {{
    title: "Customer Distribution",
    paper_bgcolor: '#1c1c1c',
    font: {{ color: '#ffffff' }}
}});

// TREND
Plotly.newPlot('trend', [{{
    x: {trend_x},
    y: {trend_y},
    mode: 'lines+markers'
}}], {{
    title: "Customer Trend",
    paper_bgcolor: '#1c1c1c',
    plot_bgcolor: '#1c1c1c',
    font: {{ color: '#ffffff' }}
}});

</script>

<div class="card">
<h3>📄 Data Preview</h3>
{preview}
</div>

<div class="card">
<h3>🎯 Customer Segments</h3>
<div class="row">
{segment_cards}
</div>
</div>

<div class="card">
<h3>📊 Detailed Analysis</h3>
{summary_df.to_html(classes="table table-dark")}
</div>

<div class="card">
<form method="post" action="/upload">
<select name="columns" multiple class="form-control">
{"".join([f'<option value="{c}" selected>{c}</option>' for c in all_cols])}
</select>
<button class="btn btn-info mt-2">Apply Filter</button>
</form>
</div>

<a href="/" class="btn btn-light mt-3">Reset</a>

<p class="text-center mt-4" style="font-size:13px; color:#90caf9;">
    Developed by Prince Kumar(23ERACS211) & Prince Kumar(22ERACS028) 🚀

</div>
</body>
</html>
""")

    response.set_cookie("session_id", session_id)
    return response