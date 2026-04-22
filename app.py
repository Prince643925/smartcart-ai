from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from preprocess import preprocess_data
from model import run_clustering

app = FastAPI()


# ✅ HOME ROUTE (FIXES 404)
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartCart AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>

    <body style="background: linear-gradient(to right, #1f4037, #99f2c8); color:white; text-align:center; padding-top:100px;">

        <h1 style="font-size:50px;">🛒 SmartCart AI</h1>
        <p style="font-size:20px;">AI-based Customer Segmentation System</p>

        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required class="form-control w-50 mx-auto">
            <br>
            <button class="btn btn-primary btn-lg">Analyze Customers</button>
        </form>

        <br><br>
        <p style="opacity:0.7;">Developed by Prince Singh</p>

    </body>
    </html>
    """)


# ✅ ONLY ONE UPLOAD ROUTE (VERY IMPORTANT)
@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    # Preprocess
    X_pca, df_processed = preprocess_data(df)

    # ✅ FIXED: unpack 3 values correctly
    labels, best_k, score = run_clustering(X_pca)

    # ✅ DEBUG (check lengths)
    print("Rows:", len(df_processed))
    print("Labels:", len(labels))

    # ✅ IMPORTANT FIX (prevents mismatch error)
    if len(labels) != len(df_processed):
        return HTMLResponse("<h2>Error: Label size mismatch. Check preprocessing.</h2>")

    df_processed["Cluster"] = labels

    # 📊 Table
    summary = df_processed.groupby("Cluster").mean().to_html(
        classes="table table-dark table-hover table-bordered"
    )

    # 📈 Graph
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("Customer Segmentation")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()

    # 🎯 Insights
    insights = df_processed.groupby("Cluster").mean()

    high_spender = insights["SpendingScore"].idxmax() if "SpendingScore" in insights.columns else "N/A"
    low_spender = insights["SpendingScore"].idxmin() if "SpendingScore" in insights.columns else "N/A"

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartCart Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

        <style>
            body {{
                margin: 0;
                font-family: 'Segoe UI', sans-serif;
                background: #0f2027;
                color: white;
            }}

            .sidebar {{
                position: fixed;
                height: 100%;
                width: 220px;
                background: #111;
                padding-top: 30px;
            }}

            .sidebar h2 {{
                text-align: center;
                margin-bottom: 30px;
            }}

            .sidebar a {{
                display: block;
                padding: 15px;
                color: white;
                text-decoration: none;
            }}

            .sidebar a:hover {{
                background: #00c6ff;
            }}

            .main {{
                margin-left: 220px;
                padding: 30px;
            }}

            .card-box {{
                background: #1c1c1c;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
            }}

            .metric {{
                text-align: center;
                padding: 20px;
                border-radius: 10px;
                background: linear-gradient(135deg, #00c6ff, #0072ff);
            }}
        </style>
    </head>

    <body>

    <div class="sidebar">
        <h2>🛒 SmartCart</h2>
        <a href="/">🏠 Home</a>
        <a href="#">📊 Dashboard</a>
    </div>

    <div class="main">

        <h1>📊 Analytics Dashboard</h1>

        <div class="row mt-4">

            <div class="col-md-4">
                <div class="metric">
                    <p>Clusters</p>
                    <h2>{best_k}</h2>
                </div>
            </div>

            <div class="col-md-4">
                <div class="metric">
                    <p>Silhouette</p>
                    <h2>{round(score,3)}</h2>
                </div>
            </div>

            <div class="col-md-4">
                <div class="metric">
                    <p>Customers</p>
                    <h2>{len(df_processed)}</h2>
                </div>
            </div>

        </div>

        <div class="card-box mt-4">
            <h3>📈 Segmentation Plot</h3>
            <img src="data:image/png;base64,{image_base64}" class="img-fluid">
        </div>

        <div class="card-box">
            <h3>📊 Cluster Analysis</h3>
            {summary}
        </div>

        <div class="card-box">
            <h3>🎯 Insights</h3>
            <p>💰 Cluster <b>{high_spender}</b> → High Value</p>
            <p>📉 Cluster <b>{low_spender}</b> → Low Value</p>
        </div>

        <a href="/" class="btn btn-light">⬅ Back</a>

        <br><br>
        <p style="opacity:0.6;">Developed by Prince Singh</p>

    </div>

    </body>
    </html>
    """)