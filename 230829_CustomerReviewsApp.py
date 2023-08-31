import streamlit as st
import pandas as pd
# import spacy
from textblob import TextBlob
import plotly.express as px
import openai
import tiktoken
import numpy as np
import plotly

openai.api_key = ""


# nlp = spacy.load("en_core_web_sm")

def gptSentimentApi(df):
    
# =============================================================================
    encoding = tiktoken.get_encoding("cl100k_base")

    df['tokens']=df['Review'].apply(lambda x:len(encoding.encode(x)))
    df['tokensSum'] = df['tokens'].cumsum()
    df['tokenIterNum'] = df['tokensSum']//3500
    
    dfSent = pd.DataFrame()
    for i in df.tokenIterNum.unique():
        dfIter = df[df.tokenIterNum == i].reset_index(drop = True)
        df = df[df.tokenIterNum >= i]

# =============================================================================
        review_list = dfIter['Review'].tolist()
        review_str=get_input_reviews(review_list,len(dfIter))
        prompt="Provide a sentiment score for each of these reviews:\n" + review_str + "Give the output in a scale of 0-10 against the review text."
        
        Final_response=get_response(prompt)
        Final_response = Final_response[-len(dfIter):]
    
        dfIter['Sentiment Score'] = pd.Series([i[1] for i in pd.Series(Final_response).str.split('. ')]).astype(int)
    
        dfSent = pd.concat([dfSent, dfIter]).drop(['tokens', 'tokensSum', 'tokenIterNum'], 1)

    return dfSent.reset_index(drop = True)


def get_input_reviews(review_list,end):
    review_str = ''
    for i, review in enumerate(review_list, start=1):
        tweet_str = str(i) + '. "' + review + '"\n'
        review_str += tweet_str
        if i == end:
            break
    return review_str

          
def get_response(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=120,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return (response["choices"][0]["text"]).split('\n')


def plot_sentiment_graph(df):
    # graphData = df.groupby('Time Period', as_index = False)['Sentiment Score'].mean()

    # fig = px.line(graphData, x='Time Period', y='Sentiment Score', title='Average Sentiment Polarity Over Time')
    # fig.update_traces(mode='lines+markers', line_shape='spline')

    # st.plotly_chart(fig)

    lineChart = px.line(grouped_df, x = 'Date', y = ['Negative_avg', 'Neutral_avg', 'Positive_avg'], markers = True, line_shape = 'spline',
                  color_discrete_map = {'Negative_avg' : 'red', 'Neutral_avg' : 'blue', 'Positive_avg' : 'green'}).update_layout(
                  title = 'Average sentiment score (out of 10) over time', yaxis_title = 'Score'
                  ).update_traces(marker=dict(size=10, symbol = 'circle'))
                      
    areaChart = px.area(grouped_df[['Date', 'Negative_count%', 'Neutral_count%', 'Positive_count%']].melt(id_vars = 'Date', var_name = 'Sentiment', value_name = 'Percentage'),
                  x = 'Date', y = 'Percentage', color = 'Sentiment', line_shape = 'spline',
                  color_discrete_map = {'Negative_count%' : 'red', 'Neutral_count%' : 'lightblue', 'Positive_count%' : 'green'}).update_layout(title='Proportion of sentiments over time', yaxis_title = 'Percentage')


    return lineChart, areaChart
    

def analyze_reviews(df):
    def sentiment_analysis(text):
        if isinstance(text, str):
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            # subjectivity = analysis.sentiment.subjectivity
            return polarity#, subjectivity
        else:
            return None#, None
    
    # def named_entity_recognition(text):
    #     if isinstance(text, str):
    #         doc = nlp(text)
    #         entities = [ent.text for ent in doc.ents]
    #         return ", ".join(entities)
    #     else:
    #         return ""

    # df["Sentiment"] = df["Review"].apply(sentiment_analysis)
    df["Sentiment Score"] = df["Review"].apply(sentiment_analysis).apply(pd.Series)
    # df["Entities"] = df["Review"].apply(named_entity_recognition)
    
    return df


st.set_page_config(page_title='Customer Reviews Analysis', 
                   page_icon='https://upload.wikimedia.org/wikipedia/commons/2/2b/DeloitteNewSmall.png', 
                   layout="wide", initial_sidebar_state="auto", menu_items=None)

st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Deloitte.svg/1920px-Deloitte.svg.png', width = 150)
st.title("Customer Reviews Analysis")

col1, col2 = st.columns([3,1])
uploaded_file = col1.file_uploader("Upload the customer reviews file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(#r'C:\Users\shraha\OneDrive - Deloitte (O365D)\Desktop\Work\Gen AI\Capstone Text Analytics\hotelsampledata.csv', 
                         uploaded_file,
                         encoding = 'ISO-8859-1')
        
        if 'Product' in df.columns and 'Review Date' in df.columns and 'Review Title' in df.columns and 'Review' in df.columns:
            col2.write(''); col2.write('')
            col2.success("Data loaded successfully!")
            
            col1, col2, col3, col4 = st.columns([3, 4, 4, 3])

            selected_product = col1.selectbox("Select a Product", df['Product'].unique())
            filtered_df = df[df['Product'] == selected_product]
            
            col1, col2 = st.columns([3, 1])
                            
            # df = gptSentimentApi(df)
            filtered_df = analyze_reviews(filtered_df)
            filtered_df['Sentiment Score'] = round((filtered_df['Sentiment Score'] + 1) * 5)
            filtered_df['Sentiment'] = np.where(filtered_df['Sentiment Score'] < 4, 'Negative', np.where(filtered_df['Sentiment Score']>6, 'Positive', 'Neutral'))

            col1.success("Analysis complete!")
            
            col2.write('')
            if col2.checkbox("Show Sentiment Scores Data"):
                st.write(filtered_df)
                                
            time_period = col4.selectbox("Timeline:", ["Monthly", "Quaterly", "Yearly"], index= 2)

            if time_period == "Monthly":
                filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'])
                filtered_df['Time Period'] = filtered_df['Review Date'].dt.to_period('M').astype('str')
            elif time_period == 'Quaterly':
                filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'])
                filtered_df['Time Period'] = filtered_df['Review Date'].dt.to_period('Q').astype('str')
            elif time_period == 'Yearly':
                filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'])
                filtered_df['Time Period'] = filtered_df['Review Date'].dt.to_period('Y').astype('str')
                
            grouped_df = filtered_df.groupby(['Time Period', 'Sentiment'])['Sentiment Score'].agg(['mean', 'count']).reset_index().pivot(index='Time Period', columns='Sentiment').reset_index()
            grouped_df.columns = [''.join(col) for col in grouped_df.columns.values]
            colIntersect = [element for element in ['Time Period', 'meanNegative', 'meanNeutral', 'meanPositive', 'countNegative', 'countNeutral', 'countPositive'] if element not in grouped_df.columns]
            if len(colIntersect) > 0:
                grouped_df[colIntersect] = np.nan
            grouped_df.rename(columns = {'Time Period': 'Date', 'meanNegative': 'Negative_avg', 'meanNeutral': 'Neutral_avg', 'meanPositive': 'Positive_avg',
                                         'countNegative': 'Negative_count', 'countNeutral': 'Neutral_count', 'countPositive': 'Positive_count'},
                              inplace = True)
            # grouped_df.columns = ['Date', 'Negative_avg', 'Neutral_avg', 'Positive_avg', 'Negative_count', 'Neutral_count', 'Positive_count']
            grouped_df[['Negative_count%', 'Neutral_count%', 'Positive_count%']] = grouped_df[['Negative_count', 'Neutral_count', 'Positive_count']].div(grouped_df[['Negative_count', 'Neutral_count', 'Positive_count']].sum(axis=1), axis=0) * 100
            
            grouped_df[grouped_df.filter(like='count').columns] = grouped_df.filter(like='count').fillna(0)
            
            lineChart, areaChart = plot_sentiment_graph(grouped_df)
            
            col1, col2 = st.columns([1,1])
            col1.plotly_chart(lineChart)
            col2.plotly_chart(areaChart)

        else:
            st.error("Incorrect columns in the uploaded file. Required columns: 'Product', 'Review Date', 'Review Title', 'Review'")
        

    
    except Exception as e:
        st.error(f"Error loading data: {e}")

