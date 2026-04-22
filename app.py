from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from preprocess import preprocess_data
from model import run_clustering

app = FastAPI()


# ✅ HOME PAGE
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
    <html>
    <head>
        <title>SmartCart AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>

    <body style="background:#0f2027;color:white;text-align:center;padding-top:120px;">

        <h1 style="font-size:60px;">🛒 SmartCart AI</h1>
        <p style="font-size:22px;">Customer Intelligence Platform</p>

        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required class="form-control w-50 mx-auto">
            <br>
            <button class="btn btn-success btn-lg">Analyze Customers</button>
        </form>

        <p style="margin-top:40px;opacity:0.6;">AI-powered decision system for modern businesses</p>

    </body>
    </html>
    """)


# ✅ RESULT PAGE
@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        if df.empty:
            return HTMLResponse("<h2>❌ Empty file uploaded</h2>")

        X_pca, df_processed = preprocess_data(df)
        labels, best_k, score = run_clustering(X_pca)

        df_processed["Cluster"] = labels

        summary_df = df_processed.groupby("Cluster").mean()

        summary = summary_df.to_html(classes="table table-dark table-hover")
        preview = df_processed.head().to_html(classes="table table-dark")

        # 📈 Plot
        plt.figure()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()

        # ================= BUSINESS LOGIC ================= #

        segment_cards = ""

        for c in summary_df.index:
            spend = summary_df.loc[c].get("SpendingScore", 0)

            if spend > 70:
                tag = "💎 High Value"
                action = "Retain with premium experience & loyalty rewards"
                color = "#00c6ff"
            elif spend < 30:
                tag = "⚠️ Low Value"
                action = "Re-engage using discounts & campaigns"
                color = "#ff4d4d"
            else:
                tag = "🛍️ Medium Value"
                action = "Upsell and cross-sell products"
                color = "#00c851"

            segment_cards += f"""
            <div class="col-md-4">
                <div class="segment-card">
                    <h4>Cluster {c}</h4>
                    <h5 style="color:{color};">{tag}</h5>
                    <p>Spending Score: {round(spend,1)}</p>
                    <p><b>Action:</b> {action}</p>
                </div>
            </div>
            """

        # AI SUMMARY
        ai_summary = f"""
        <div class="card-box">
        <h3>🧠 Executive Summary</h3>
        <p>
        Your customers are segmented into <b>{best_k}</b> groups.
        High-value users drive revenue, while low-value users require activation strategies.
        </p>
        </div>
        """

        # RISK
        if score < 0.3:
            risk = "<span style='color:red;'>Weak segmentation</span>"
        elif score < 0.5:
            risk = "<span style='color:orange;'>Moderate segmentation</span>"
        else:
            risk = "<span style='color:lightgreen;'>Strong segmentation</span>"

    except:
        return HTMLResponse("""
        <body style="background:#0f2027;color:white;text-align:center;padding-top:100px;">
        <h2>Invalid Dataset</h2>
        <a href="/">Go Back</a>
        </body>
        """)

    return HTMLResponse(f"""
    <html>
    <head>
        <title>Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

        <style>
        body {{
            background:#0f2027;
            color:white;
            font-family:sans-serif;
        }}

        .main {{
            padding:40px;
        }}

        .kpi {{
            background:#1c1c1c;
            padding:25px;
            border-radius:15px;
            text-align:center;
            transition:0.3s;
        }}

        .kpi:hover {{
            transform:scale(1.05);
        }}

        .card-box {{
            background:#1c1c1c;
            padding:25px;
            border-radius:15px;
            margin-top:20px;
        }}

        .segment-card {{
            background:#1c1c1c;
            padding:20px;
            border-radius:15px;
            margin-top:15px;
            text-align:center;
            transition:0.3s;
        }}

        .segment-card:hover {{
            transform:translateY(-5px);
        }}

        h1 {{
            font-weight:600;
        }}
        </style>
    </head>

    <body>

    <div class="main">

        <h1>📊 SmartCart Business Dashboard</h1>
        <p>AI-powered insights for business growth</p>

        <!-- KPI -->
        <div class="row mt-4">

            <div class="col-md-4">
                <div class="kpi">
                    <h5>Customers</h5>
                    <h2>{len(df_processed)}</h2>
                </div>
            </div>

            <div class="col-md-4">
                <div class="kpi">
                    <h5>Segments</h5>
                    <h2>{best_k}</h2>
                </div>
            </div>

            <div class="col-md-4">
                <div class="kpi">
                    <h5>Model Quality</h5>
                    <h2>{round(score,2)}</h2>
                    <p>{risk}</p>
                </div>
            </div>

        </div>

        {ai_summary}

        <!-- GRAPH -->
        <div class="card-box">
            <h3>📈 Customer Segmentation</h3>
            <img src="data:image/png;base64,{image_base64}" width="500">
        </div>

        <!-- DATA -->
        <div class="card-box">
            <h3>📄 Data Preview</h3>
            {preview}
        </div>

        <!-- SEGMENTS -->
        <div class="card-box">
            <h3>🎯 Customer Segments</h3>
            <div class="row">
                {segment_cards}
            </div>
        </div>

        <!-- TABLE -->
        <div class="card-box">
            <h3>📊 Detailed Analysis</h3>
            {summary}
        </div>

        <a href="/" class="btn btn-light mt-3">⬅ Back</a>

    </div>

    </body>
    </html>
    """)