import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from wordcloud import WordCloud
import os
from fpdf import FPDF
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Telecom Customer Feedback Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Theme Colors
# ----------------------------
PRIMARY_RED = "#d54133"
DARK_TEAL = "#163447"
LIGHT_TEAL = "#4b8ca6"
HIGHLIGHT_YELLOW = "#f0c808"
NEUTRAL_GREY = "#888888"
TEXT_WHITE = "white"


# ----------------------------
# Utility: Save Chart
# ----------------------------
def save_chart(fig, filename):
    try:
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor="white")  # white bg
        plt.close(fig)
    except Exception as e:
        st.error(f"❌ Error saving {filename}: {e}")


# ----------------------------
# Insights Generator
# ----------------------------
def generate_insights(df):
    insights = []
    if "Location" in df.columns:
        top_region = df["Location"].mode()[0]
        top_region_count = df["Location"].value_counts().iloc[0]
        insights.append(f"- The region with the most feedback is {top_region} ({top_region_count} entries).")
    if "SentimentScore" in df.columns:
        avg_sentiment = df["SentimentScore"].mean()
        if avg_sentiment > 0.2:
            sentiment_summary = "Overall sentiment is Positive."
        elif avg_sentiment < -0.2:
            sentiment_summary = "Overall sentiment is Negative."
        else:
            sentiment_summary = "Overall sentiment is Neutral."
        insights.append(f"- Average sentiment score is {avg_sentiment:.2f}. {sentiment_summary}")
    if "CustomerFeedback" in df.columns:
        common_words = pd.Series(" ".join(df["CustomerFeedback"].astype(str)).split()).value_counts().head(5)
        keywords = ", ".join(common_words.index.tolist())
        insights.append(f"- The most common keywords mentioned by customers are: {keywords}")
    return "\n".join(insights)


# ----------------------------
# PDF Report Generator (Dark Corporate Theme)
# ----------------------------
def generate_pdf_report(df, charts):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover Page
    pdf.add_page()
    pdf.set_fill_color(22, 52, 71)  # Dark Teal Background
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_fill_color(213, 65, 51)  # Red Banner
    pdf.rect(20, 40, 170, 20, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.set_xy(20, 43)
    pdf.cell(170, 10, "Telecom Customer Feedback Report", align="C")

    pdf.set_text_color(240, 200, 8)  # Yellow
    pdf.set_font("Arial", '', 12)
    pdf.ln(30)
    pdf.cell(0, 10, "Generated Customer Feedback Insights", ln=True, align="C")

    # Chart Pages
    def add_chart_page(title, description, image_path):
        if not os.path.exists(image_path):
            return
        pdf.add_page()
        pdf.set_fill_color(22, 52, 71)  # Dark Teal BG
        pdf.rect(0, 0, 210, 297, "F")

        # Title
        pdf.set_text_color(213, 65, 51)  # Red
        pdf.set_font("Arial", 'B', 14)
        pdf.ln(10)
        pdf.cell(0, 10, title, ln=True, align="L")

        # Description
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 8, description)
        pdf.ln(5)

        # Image placement
        with Image.open(image_path) as im:
            w_px, h_px = im.size
        aspect = h_px / w_px
        max_w, max_h = 180, 120
        draw_w, draw_h = max_w, max_w * aspect
        if draw_h > max_h:
            draw_h = max_h
            draw_w = draw_h / aspect
        x = (210 - draw_w) / 2
        y = pdf.get_y()
        pdf.image(image_path, x=x, y=y, w=draw_w, h=draw_h)

    for chart in charts:
        add_chart_page(chart["title"], chart["desc"], chart["file"])

    # Summary Page
    pdf.add_page()
    pdf.set_fill_color(22, 52, 71)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_text_color(213, 65, 51)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Summary & Insights", ln=True)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, generate_insights(df))

    pdf_file = "Customer_Feedback_Report.pdf"
    pdf.output(pdf_file)
    return pdf_file


# ----------------------------
# Main App
# ----------------------------
st.title("📊 Telecom Customer Feedback Dashboard")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    for img in ["region.png", "gauge.png", "sentiment.png", "wordcloud.png"]:
        if os.path.exists(img):
            os.remove(img)

    st.subheader("🔎 Data Preview")
    st.dataframe(df.head())

    charts = []

    # Region-wise Chart
    if "Location" in df.columns:
        st.subheader("📍 Region-wise Customer Count")
        region_fig = px.histogram(df, x="Location", color_discrete_sequence=[PRIMARY_RED])
        region_fig.update_layout(plot_bgcolor=DARK_TEAL, paper_bgcolor=DARK_TEAL, font=dict(color=TEXT_WHITE))
        st.plotly_chart(region_fig, use_container_width=True)
        st.markdown("ℹ️ **Description:** Shows feedback volume by region to identify areas with the most engagement.")

        export_fig = go.Figure(region_fig)
        export_fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
        export_fig.write_image("region.png", width=1200, height=700, scale=2)

        charts.append({
            "title": "Region-wise Customer Distribution",
            "desc": "Shows feedback volume by region to identify engagement hotspots.",
            "file": "region.png"
        })

    # Sentiment Analysis
    if "CustomerFeedback" in df.columns:
        st.subheader("💬 Customer Sentiment Analysis")
        df["SentimentScore"] = df["CustomerFeedback"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        df["SentimentLabel"] = df["SentimentScore"].apply(
            lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

        avg_sentiment = df["SentimentScore"].mean()
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_sentiment,
            title={'text': "Customer Sentiment", 'font': {'color': PRIMARY_RED, 'size': 22}},
            number={'font': {'color': TEXT_WHITE, 'size': 18}},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': HIGHLIGHT_YELLOW},
                'steps': [
                    {'range': [-1, -0.2], 'color': "#800000"},
                    {'range': [-0.2, 0.2], 'color': NEUTRAL_GREY},
                    {'range': [0.2, 1], 'color': LIGHT_TEAL}
                ],
                'bordercolor': PRIMARY_RED,
            }
        ))
        fig_gauge.update_layout(paper_bgcolor=DARK_TEAL, font={'color': TEXT_WHITE})
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("ℹ️ **Description:** Gauge shows the average sentiment score (-1 = negative, +1 = positive).")

        export_gauge = go.Figure(fig_gauge)
        export_gauge.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
        export_gauge.write_image("gauge.png", width=1200, height=700, scale=2)

        charts.append({
            "title": "Average Customer Sentiment",
            "desc": "Gauge reflects average sentiment score from feedback (-1 to +1).",
            "file": "gauge.png"
        })

        sentiment_fig, ax = plt.subplots()
        sns.countplot(data=df, x="SentimentLabel",
                      order=["Positive", "Neutral", "Negative"],
                      palette=[PRIMARY_RED, NEUTRAL_GREY, DARK_TEAL], ax=ax)
        ax.set_facecolor(DARK_TEAL)
        ax.set_title("Sentiment Distribution", color=PRIMARY_RED)
        save_chart(sentiment_fig, "sentiment.png")
        st.pyplot(sentiment_fig)
        st.markdown("ℹ️ **Description:** Breaks down feedback into Positive, Neutral, and Negative categories.")

        charts.append({
            "title": "Sentiment Distribution",
            "desc": "Categorizes feedback into Positive, Neutral, and Negative sentiments.",
            "file": "sentiment.png"
        })

        st.subheader("☁️ Keywords in Feedback")
        text = " ".join(df["CustomerFeedback"].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(text)
        wc_fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        wc_fig.savefig("wordcloud.png", dpi=300, bbox_inches="tight", facecolor="white")
        st.pyplot(wc_fig)
        st.markdown("ℹ️ **Description:** Highlights the most frequent keywords in customer feedback.")

        charts.append({
            "title": "Feedback Word Cloud",
            "desc": "Highlights the most frequent words in customer feedback.",
            "file": "wordcloud.png"
        })

    # PDF Export
    if st.button("📑 Generate PDF Report"):
        pdf_file = generate_pdf_report(df, charts)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Download PDF Report", f, file_name=pdf_file, mime="application/pdf")
