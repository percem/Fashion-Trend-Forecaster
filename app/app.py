import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/random_forest.pkl")

importance = joblib.load("models/importance.pkl")

# Load feature names
feature_names = joblib.load("models/features.pkl")

st.set_page_config(
    page_title="Fashion Trend Forecaster",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Fashion Trend Forecaster")

st.write("""
Welcome!

This application predicts whether a fashion product is likely to become a trend using a Machine Learning model.
""")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Prediction",
        "Dashboard",
        "About"
    ]
)

if page == "Prediction":

        st.header("Trend Prediction")

        st.write("Enter product information below.")

        left_col, right_col = st.columns(2)

        with left_col:
            category = st.selectbox(
                "👗 Category",
                ["Dress", "T-Shirt", "Sweater", "Jacket", "Pants"]
            )

        with right_col:
            price = st.number_input(
                "💰 Price",
                min_value=100,
                max_value=2000,
                value=500
            )

        with left_col:
            discount = st.slider(
                "🏷️ Discount (%)",
                0,
                40,
                10
            )

        with right_col:
            trend_score = st.slider(
                "📈 Trend Score",
                1.0,
                10.0,
                5.0
            )

        with left_col:
            stock = st.number_input(
                "📦 Stock",
                0,
                2000,
                500
            )

        with right_col:
            rating = st.slider(
                "⭐ Rating",
                3.5,
                5.0,
                4.2
            )

        with left_col:
            marketing = st.number_input(
                "📢 Marketing Budget",
                1000,
                100000,
                50000
            )

        with right_col:
            month = st.selectbox(
                "📅 Month",
                list(range(1, 13))
            )

        with left_col:
            quarter = st.selectbox(
                "🗓️ Quarter",
                [1, 2, 3, 4]
            )

        with right_col:
            day = st.selectbox(
                "📆 Day of Week",
                list(range(7))
            )


        predict = st.button("Predict")

        if predict:

            input_data = pd.DataFrame({

                "Price":[price],
                "Discount":[discount],
                "TrendScore":[trend_score],
                "Stock":[stock],
                "Rating":[rating],
                "MarketingBudget":[marketing],
                "Month":[month],
                "Quarter":[quarter],
                "DayOfWeek":[day],

                "Category_Jacket":[1 if category=="Jacket" else 0],
                "Category_Pants":[1 if category=="Pants" else 0],
                "Category_Sweater":[1 if category=="Sweater" else 0],
                "Category_T-Shirt":[1 if category=="T-Shirt" else 0]

                })

                       
            input_data = input_data[feature_names]


            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0][1]

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 1:
                st.success("🔥 This product is likely to become TREND!")
            else:
                st.error("❌ This product is NOT likely to become Trend.")

            st.metric(
                "Trend Probability",
                f"{probability*100:.2f}%"
            )

            st.divider()

            st.subheader("📋 Selected Product Summary")

            summary = pd.DataFrame({
                "Feature": [
                    "Category",
                    "Price",
                    "Discount",
                    "Trend Score",
                    "Stock",
                    "Rating",
                    "Marketing Budget",
                    "Month",
                    "Quarter",
                    "Day of Week"
                ],
                "Value": [
                    category,
                    f"₺{price}",
                    f"{discount}%",
                    trend_score,
                    stock,
                    rating,
                    f"₺{marketing}",
                    month,
                    quarter,
                    day
                ]
            })

            st.table(summary)

            st.progress(float(probability))

            if probability >= 0.80:
                st.success("Very High Trend Potential 🚀")

            elif probability >= 0.60:
                st.info("Moderate Trend Potential 📈")

            elif probability >= 0.40:
                st.warning("Uncertain Trend Potential ⚠️")

            else:
                st.error("Low Trend Potential 📉")
            
elif page == "Dashboard":

    st.header("📊 Fashion Analytics Dashboard")

    df = pd.read_csv("data/fashion_trend_dataset.csv")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📦 Total Products", len(df))

    with col2:
        st.metric("💰 Avg Price", f"₺{df['Price'].mean():.0f}")

    with col3:
        st.metric("⭐ Avg Rating", f"{df['Rating'].mean():.2f}")

    with col4:
        st.metric("📉 Avg Discount", f"{df['Discount'].mean():.1f}%")

    st.divider()

    st.subheader("📈 Average Price by Category")

    avg_price = df.groupby("Category")["Price"].mean()

    st.bar_chart(avg_price)

    st.divider()

    st.subheader("💰 Average Marketing Budget by Category")

    avg_marketing = (
    df.groupby("Category")["MarketingBudget"]
    .mean()
    .sort_values()
    )

    st.bar_chart(avg_marketing)

    st.divider()

    st.subheader("⭐ Average Rating by Category")

    avg_rating = (
    df.groupby("Category")["Rating"]
    .mean()
    .sort_values()
    )

    st.bar_chart(avg_rating)

    st.subheader("🔥 Trend vs Non-Trend Products")

    trend_counts = df["Trend"].value_counts()

    st.bar_chart(trend_counts)

    st.divider()

    st.subheader("🛍️ Product Count by Category")

    category_counts = df["Category"].value_counts()

    st.bar_chart(category_counts)

    st.divider()

    st.subheader("💰 Price Distribution")

    st.area_chart(df["Price"])

    st.divider()

    st.subheader("⭐ Rating Distribution")

    rating_counts = df["Rating"].value_counts().sort_index()

    st.line_chart(rating_counts)

    st.divider()

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head(10))

    st.divider()

    st.subheader("⭐ Average Trend Score by Category")

    trend_score = (
    df.groupby("Category")["TrendScore"]
      .mean()
      .sort_values()
    )

    st.bar_chart(trend_score)

    st.divider()

    st.subheader("📦 Stock by Category")

    stock = (
    df.groupby("Category")["Stock"]
      .mean()
    )

    st.bar_chart(stock)

    st.divider()

    st.subheader("⭐ Average Rating")

    rating = (
    df.groupby("Category")["Rating"]
      .mean()
    )

    st.bar_chart(rating)

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20))

    st.divider()

    st.subheader("📊 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )



elif page == "About":

    st.header("ℹ️ About Fashion Trend Forecaster")

    st.write("""
Fashion Trend Forecaster is a Machine Learning application developed to predict
whether a fashion product is likely to become a trend.

The project uses a Random Forest Classifier trained on a synthetic fashion dataset
containing product information such as price, discount, rating, stock level,
marketing budget and seasonal information.
""")

    st.divider()

    st.subheader("🎯 Project Features")

    st.markdown("""
- ✅ Trend Prediction using Machine Learning
- ✅ Probability Score
- ✅ Interactive Dashboard
- ✅ Feature Importance Analysis
- ✅ Category-based Analytics
- ✅ Streamlit Web Application
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Machine Learning")

        st.markdown("""
**Algorithm**
- Random Forest Classifier

**Dataset**
- Synthetic Fashion Dataset

**Records**
- 4,560 Products
        """)

    with col2:
        st.subheader("🛠 Technologies")

        st.markdown("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib
        """)

    st.divider()

    st.success("Developed by Ummuhan Percem Koklen")

