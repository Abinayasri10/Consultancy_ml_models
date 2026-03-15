import pandas as pd
from pymongo import MongoClient
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import json

app = Flask(__name__)
CORS(app)

VERSION = "v2.0.1"

@app.route('/api/ml/sales-analysis', methods=['GET'])
def analyze_sales():
    print(f"\n📥 [{VERSION}] Processing Advanced Sales Analysis...")
    try:
        # 1. MongoDB Connection
        try:
            MONGO_URI = os.environ.get(
                "MONGODB_URI",
                "mongodb+srv://abinayasrij23cse_db_user:CmweYIkDsfUXnM2C@cluster0.ixomlpk.mongodb.net/pesticide_db?appName=Cluster0"
            )
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client["pesticide_db"]
            sales_cursor = db.sales.find()
            sales_list = list(sales_cursor)
            print(f"✅ Found {len(sales_list)} sales records.")
        except Exception as mongo_err:
            print(f"❌ MongoDB Error: {mongo_err}")
            return jsonify({"message": "Database unreachable"}), 200

        if not sales_list:
            return jsonify({
                "message": "Empty data",
                "seasonal": [],
                "topProducts": [],
                "customerPurchaseData": [],
                "insights": ["No data available."]
            }), 200

        df = pd.DataFrame(sales_list)
        
        # Ensure total is numeric and saleDate is datetime
        df['total'] = pd.to_numeric(df['total'], errors='coerce').fillna(0)
        df['saleDate'] = pd.to_datetime(df['saleDate'], errors='coerce')
        
        # 2. CUSTOMER RANKING
        # Use 'customer' field for grouping. Ensure it's treated as string.
        df['customer'] = df['customer'].astype(str)
        customer_analysis = df.groupby('customer').agg({
            'total': 'sum',
            '_id': 'count'
        }).rename(columns={'total': 'totalSpent', '_id': 'purchaseCount'}).reset_index()
        
        customer_analysis = customer_analysis.sort_values(by='totalSpent', ascending=False)
        customer_data_list = customer_analysis.to_dict(orient='records')

        # 3. SEASONAL ANALYSIS
        df['month'] = df['saleDate'].dt.strftime('%b')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal = df.groupby('month')['total'].sum().reindex(month_order).fillna(0).reset_index()

        # 4. TOP PRODUCTS
        items_list = []
        for _, row in df.iterrows():
            if 'items' in row and isinstance(row['items'], list):
                for item in row['items']:
                    items_list.append({
                        "name": str(item.get('productName', 'Unknown')), 
                        "qty": pd.to_numeric(item.get('qty', 0), errors='coerce'),
                        "price": pd.to_numeric(item.get('price', 0), errors='coerce')
                    })
        
        top_products = []
        avg_prices = {}
        if items_list:
            items_df = pd.DataFrame(items_list)
            top_products = items_df.groupby('name')['qty'].sum().sort_values(ascending=False).head(5).reset_index().to_dict(orient='records')
            avg_prices = items_df.groupby('name')['price'].mean().to_dict()

        # 5. AI INSIGHTS & SUGGESTIONS
        insights = []
        if not customer_analysis.empty:
            top_customer = customer_analysis.iloc[0]
            insights.append({
                "type": "VIP",
                "title": "Top Customer Identified",
                "content": f"Customer '{top_customer['customer']}' has significantly higher purchasing power, contributing ₹{top_customer['totalSpent']:,.0f}."
            })
            
            if len(customer_analysis) > 1:
                loyalty_suggestion = f"Consider offering a loyalty discount to '{top_customer['customer']}' to maintain their high volume."
                insights.append({
                    "type": "STOCK",
                    "title": "Loyalty Program",
                    "content": loyalty_suggestion
                })

        if top_products:
            main_product = top_products[0]['name']
            insights.append({
                "type": "PROD",
                "title": "Most Moving Stock",
                "content": f"'{main_product}' is currently your top-selling product. Ensure stock levels are maintained to avoid stockouts."
            })
            
            # Find high-price items
            sorted_by_price = sorted(avg_prices.items(), key=lambda x: x[1], reverse=True)
            if sorted_by_price:
                prime_p, prime_v = sorted_by_price[0]
                insights.append({
                    "type": "PRICE",
                    "title": "Premium Inventory",
                    "content": f"{prime_p} is your most expensive item (avg ₹{prime_v:,.0f}). Focus on targeted sales for high-margin groups."
                })

        print("✅ Advanced Analysis Complete.")
        return jsonify({
            "seasonal": seasonal.to_dict(orient='records'),
            "topProducts": top_products,
            "customerPurchaseData": customer_data_list,
            "insights": insights
        })

    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/agri-research', methods=['GET'])
def get_agri_research():
    print(f"\n📥 [{VERSION}] Fetching Agricultural Research Data...")
    try:
        agri_json_path = os.path.join(os.path.dirname(__file__), "agri_insights.json")
        if os.path.exists(agri_json_path):
            with open(agri_json_path, 'r') as f:
                data = json.load(f)
                return jsonify(data)
        else:
            return jsonify({"error": "Agri insights not found. Please run training script."}), 404
    except Exception as e:
        print(f"🔥 Agri Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5005, host='0.0.0.0', debug=False)
