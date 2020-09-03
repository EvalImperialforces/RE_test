import pandas as pd
import re
import string
#import nltk
#nltk.download('wordnet')
import streamlit as st
#from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
#from PIL import Image

st.write("""
# Recommendation Engine for Customer Goal Selection


This simple app demonstrates how open text can be used to suggest the most suitable Initiatives for a Customer Goal during OSP creation. 
\nFor HCE Goal Management, Value Drivers and KPIs linked to default goals will make the most relevant catalogue items (initiatives) accessible to the customer/ESA/CoE.

""")

#image = Image.open('pic.jpg')

#st.image(image, use_column_width=True)

#st.header('User Query')

#stopwords = nltk.corpus.stopwords.words('english')
#ps = LancasterStemmer() 

# Data import 
cols = ['Value Driver Display Name (T12)', 'KPI Name', 'Value Lever']
data = pd.read_excel("Value Driver Details.xlsx", sep='\t', usecols = cols)
data['VD_KPI'] = data[['Value Driver Display Name (T12)', 'KPI Name']].agg(' '.join, axis = 1)


def clean_text(txt):
    #txt = re.sub('<.+?>', "", txt) # Remove errors in SFSF description  
    #txt = re.sub('&.+?;', "", txt)
    txt = "".join([c for c in txt if c not in string.punctuation]) # Discount punctuation
    #tokens = re.split('\W+', txt) # Split words into a list of strings
    #txt = [ps.stem(word) for word in tokens if word not in stopwords] #Stem words
    return txt


#tfidf_vect = TfidfVectorizer(analyzer=clean_text)
#corpus = tfidf_vect.fit_transform(data['VD_KPI'])

st.sidebar.header('Description of Customer Goal')

query = st.sidebar.text_input('Please enter your description here:')

st.sidebar.header('Number of Results')
no_res = st.sidebar.slider('Number of top initiatives to be displayed:', min_value=1, max_value=30)

st.sidebar.header('Filter Results by Value Lever')

#@st.cache(suppress_st_warning=True)
def best_match(query, corpus, no_res):
    
    # Apply tf-idf and cosine similarity for query and corpus
    
    #query = tfidf_vect.transform([query])
    #cosineSimilarities = cosine_similarity(corpus, query, dense_output = False)
    #cos_df = cosineSimilarities.toarray()
    
    # Generate table of of top matches
    
    #Match_percent = [i*100 for i in cos_df] # calculate percentage of match 
    #matches = sorted([(x,i) for (i,x) in enumerate(Match_percent)], reverse=True)[:no_res] 
    # index and percentage from cos_df
    #idx = [item[1] for item in matches]
    
    #matches = [item[0] for item in matches] # get the percentage
    #matches = [int(float(x)) for x in matches] # convert to integer from np.array
    #matches = [str(i) for i in matches] # convert int to string for percentage
    #matches = list(map("{}%".format, matches))
    
    ### Must list of lists to list of integers
    
    
    #VD = [data.loc[i, 'Value Driver Display Name (T12)'] for i in idx] # Description of CD & KPI
    #KPI = [data.loc[i, 'KPI Name'] for i in idx]
    #Value_Lever = [data.loc[i, 'Value Lever'] for i in idx]

    VD = query
    KPI = [1,2,3]
    
    output = pd.DataFrame({'Value_Driver':VD, 'KPI':KPI})
    #result = pd.DataFrame(result, matches)

    if st.sidebar.checkbox('Effictiveness'):
        output = output.loc[output['Value Lever'] == 'Effectiveness']


    if st.sidebar.checkbox('Agility'):
        output = output.loc[output['Value Lever'] == 'Agility']

    if st.sidebar.checkbox('Efficiency'):
        output = output.loc[output['Value Lever'] == 'Efficiency']

    if st.sidebar.checkbox('Confidence'):
        output = output.loc[output['Value Lever'] == 'Confidence']

    if st.sidebar.checkbox('Experience'):
        output = output.loc[output['Value Lever'] == 'Experience']
    
    return(output)

st.header('Top Value Drivers')
output = best_match(query, corpus, no_res)
st.table(output) 



