import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

model_path = 'naive_bayes_model.pkl'
naive_bayes_model = joblib.load(model_path)

def classify_document(doc, model):
    prior_prob = model['prior_probabilities']
    prob_cond_positive = model['prob_cond_positive']
    prob_cond_neutral = model['prob_cond_neutral']
    prob_cond_negative = model['prob_cond_negative']
    total_positive_words = model['total_positive_words']
    total_neutral_words = model['total_neutral_words']
    total_negative_words = model['total_negative_words']
    feature_count = model['feature_count']

    words = doc.split()
    posterior_positive = prior_prob['positif']
    posterior_neutral = prior_prob['netral']
    posterior_negative = prior_prob['negatif']

    for word in words:
        posterior_positive *= prob_cond_positive.get(word, 1 / (total_positive_words + feature_count))
        posterior_neutral *= prob_cond_neutral.get(word, 1 / (total_neutral_words + feature_count))
        posterior_negative *= prob_cond_negative.get(word, 1 / (total_negative_words + feature_count))

    return max(
        {'positif': posterior_positive, 'netral': posterior_neutral, 'negatif': posterior_negative},
        key=lambda x: {'positif': posterior_positive, 'netral': posterior_neutral, 'negatif': posterior_negative}[x]
    )

st.title("Klasifikasi dan Analisis Sentimen Ulasan Hotel")

page = st.sidebar.selectbox("Choose", ['Pencarian', 'Sentimen'])

if page == 'Pencarian':
    st.write("**Pencarian Nama Hotel**")
    search_hotel = st.text_input("Masukkan nama hotel:")

    if st.button("Cari Hotel"):
        if search_hotel.strip():
            df = pd.read_csv('hotel_reviews.csv')
            filtered_data = df[df['Name'].str.contains(search_hotel, case=False, na=False)]

            if not filtered_data.empty:
                st.write("**Hasil Pencarian Ulasan untuk Hotel:**")
                filtered_data['Predicted_Sentiment'] = filtered_data['Review_Text'].apply(lambda x: classify_document(x, naive_bayes_model))
                st.dataframe(filtered_data[['Name', 'Review_Text', 'Predicted_Sentiment']])

                sentiment_counts = filtered_data['Predicted_Sentiment'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
                ax.set_title("Distribusi Sentimen")
                ax.set_xlabel("Sentimen")
                ax.set_ylabel("Jumlah")
                st.pyplot(fig)

                positive_reviews = " ".join(filtered_data[filtered_data['Predicted_Sentiment'] == 'positif']['Review_Text'])
                if positive_reviews:
                    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
                    st.subheader("Wordcloud Sentimen Positif")
                    st.image(wordcloud_positive.to_array())
                else:
                    st.subheader("Tidak ada ulasan positif untuk hotel ini.")

                neutral_reviews = " ".join(filtered_data[filtered_data['Predicted_Sentiment'] == 'netral']['Review_Text'].astype(str))
                if neutral_reviews:
                    wordcloud_neutral = WordCloud(width=800, height=400, background_color='white').generate(neutral_reviews)
                    st.subheader("Wordcloud Sentimen Netral")
                    st.image(wordcloud_neutral.to_array())
                else:
                    st.subheader("Tidak ada ulasan netral untuk hotel ini.")

                negative_reviews = " ".join(filtered_data[filtered_data['Predicted_Sentiment'] == 'negatif']['Review_Text'])
                if negative_reviews:
                    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
                    st.subheader("Wordcloud Sentimen Negatif")
                    st.image(wordcloud_negative.to_array())
                else:
                    st.subheader("Tidak ada ulasan negatif untuk hotel ini.")

            else:
                st.warning("Tidak ada ulasan ditemukan untuk nama hotel yang dimasukkan.")
        else:
            st.warning("Harap masukkan nama hotel untuk mencari ulasan.")

elif page == 'Sentimen':
    st.write("**Analisis Teks Individu**")
    input_text = st.text_area("Masukkan teks ulasan:")

    if st.button("Analisis Sentimen"):
        if input_text.strip():
            preprocessed_text = input_text.lower()
            preprocessed_text = ''.join([char for char in preprocessed_text if char.isalnum() or char.isspace()])

            sentiment = classify_document(preprocessed_text, naive_bayes_model)

            st.write("**Hasil Analisis Sentimen:**")
            st.write(f"Sentimen: **{sentiment.capitalize()}**")
        else:
            st.warning("Harap masukkan teks sebelum menganalisis.")

    st.write("**Analisis Batch Ulasan**")
    file_upload = st.file_uploader("Unggah file CSV dengan kolom 'Review_Text':", type=["csv"])

    if file_upload:
        df = pd.read_csv(file_upload)

        if 'Review_Text' in df.columns:
            df['Preprocessed_Text'] = df['Review_Text'].astype(str).str.lower()
            df['Preprocessed_Text'] = df['Preprocessed_Text'].apply(lambda x: ''.join([char for char in x if char.isalnum() or char.isspace()]))

            df['Predicted_Sentiment'] = df['Preprocessed_Text'].apply(lambda x: classify_document(x, naive_bayes_model))

            st.write("Hasil Analisis:")
            st.dataframe(df[['Review_Text', 'Predicted_Sentiment']])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Hasil sebagai CSV",
                data=csv,
                file_name='predicted_sentiments.csv',
                mime='text/csv',
            )
        else:
            st.error("File CSV harus memiliki kolom 'Review_Text'.")