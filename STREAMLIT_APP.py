#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:

pip install scikit-learn
import streamlit as st 
import pandas as pd
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier

def convert_to_df(sentiment):
    sentiment_dict = {"polarity": sentiment.polarity,"subjectivity":sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=["metric","value"])
    return sentiment_df
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= 0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
    result = {positives: "pos_list",negatives: "neg_list",neutral: "neutral"} 
    return result
def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Streamlit Project")
    
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        with st.form(key="nlpform"):
            raw_text = st.text_area("Enter text Here")
            submit_button = st.form_submit_button(label="Analyze")
        # layout
        col1, col2 = st.columns(2)
        if submit_button:
            
            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)
                
                #Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment::Positive:smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment::Negative:angry: ")
                else:
                    st.markdown("Sentiment::Neutral 😐")
                
                # DataFrame
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)
                
                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                x= "metric",
                y= "value",
                color="metric"
                )
                st.altair_chart(c,use_container_width =True)
                    
            with col2:
                st.info("Token Sentiment")
                
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)
                
    else:
        st.subheader("About")
if __name__ == '__':
    main()




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




