from flask import Flask, request, render_template
import pandas as pd
import plotly.express as px

from src.pipeline.prediction import CustomData, PredictionPipeline

app = Flask(__name__)


# ----------------------------
# HOME
# ----------------------------
@app.route('/')
def home():
    return render_template(
        'index.html',
        form_data={},          # prevent Jinja error
        theme="light"          # default theme
    )


# ----------------------------
# PREDICT
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        # ----------------------------
        # INPUT DATA (SAFE)
        # ----------------------------
        data = CustomData(
            city=form_data.get('city'),
            furnishing_status=form_data.get('furnishing_status'),
            tenant_preferred=form_data.get('tenant_preferred'),
            area_type=form_data.get('area_type'),
            bhk=int(form_data.get('bhk') or 0),
            size=int(form_data.get('size') or 0),
            bathrooms=int(form_data.get('bathrooms') or 0),
            floor_level=int(form_data.get('floor_level') or 0)
        )

        pred_df = data.get_data_as_data_frame()

        pipeline = PredictionPipeline()
        result = pipeline.predict(pred_df)[0]

        # ----------------------------
        # LOAD DATASET
        # ----------------------------
        df = pd.read_csv("House_Rent_Dataset.csv")

        # ----------------------------
        # GRAPH 1 → CITY
        # ----------------------------
        city_avg = df.groupby("City")["Rent"].mean().reset_index()

        fig1 = px.bar(
            city_avg,
            x="Rent",
            y="City",
            orientation='h',
            title="Average Rent by City",
            color="Rent"
        )

        fig1.update_layout(
            plot_bgcolor='#f1f5f9',
            paper_bgcolor='#f1f5f9'
        )

        graph1 = fig1.to_html(full_html=False)

        # ----------------------------
        # GRAPH 2 → BHK (CITY BASED)
        # ----------------------------
        selected_city = form_data.get('city')
        city_df = df[df["City"] == selected_city].copy()

        if city_df.empty:
            graph2 = "<h5 style='text-align:center'>No data available</h5>"
        else:
            city_df["BHK"] = pd.to_numeric(city_df["BHK"], errors='coerce')
            bhk_avg = city_df.groupby("BHK")["Rent"].mean().reset_index()

            fig2 = px.bar(
                bhk_avg,
                x="BHK",
                y="Rent",
                title=f"Rent vs BHK in {selected_city}",
                color="Rent"
            )

            fig2.update_layout(
                plot_bgcolor='#f1f5f9',
                paper_bgcolor='#f1f5f9'
            )

            graph2 = fig2.to_html(full_html=False)

        # ----------------------------
        # 💡 INSIGHT LOGIC
        # ----------------------------
        if result > 30000:
            insight = "💡 High rent — premium property or prime location."
        elif result > 15000:
            insight = "💡 Moderate rent — standard urban pricing."
        else:
            insight = "💡 Affordable rent — budget-friendly option."

        # ----------------------------
        # 🤖 RECOMMENDATION SYSTEM
        # ----------------------------
        recommended = df[
            (df["Rent"] >= result * 0.8) &
            (df["Rent"] <= result * 1.2)
        ]

        recommended = recommended.sort_values(by="Rent", ascending=False)

        recommendations = recommended[["City", "BHK", "Rent"]] \
            .head(5) \
            .to_dict(orient="records")

        # ----------------------------
        # RETURN
        # ----------------------------
        return render_template(
            'index.html',
            results=f"Predicted Rent: ₹{round(result, 2)}",
            graph1=graph1,
            graph2=graph2,
            insight=insight,
            recommendations=recommendations,
            form_data=form_data,
            theme=form_data.get('theme', 'light')   # theme support
        )

    except Exception as e:
        print("ERROR:", e)

        return render_template(
            'index.html',
            results=f"Error: {str(e)}",
            form_data=request.form,
            theme="light"
        )


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)