import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

#cv_excel_data_file = 'cv_scan2021_11_02_11_09_36.xlsx'


@st.cache(allow_output_mutation=True)
def load_pretrained_model(model_name):
    model = SentenceTransformer(model_name)
    return model


@st.cache
def get_pred_matrix_data(file_name):
    #path = "data/output/cross_validation_intent_level/"
    file = file_name
    sheet = 'prediction_matrix'
    df = pd.read_excel(file, sheet_name=sheet)
    df = df.set_index("index")
    return df

@st.cache
def get_cross_validation_data(file_name):
    #path = "data/output/cross_validation_intent_level/"
    file = file_name
    sheet = 'current_all_data'
    df = pd.read_excel(file, sheet_name=sheet)
    return df

@st.cache
def get_flaged_data(file_name):
    #path = "data/output/cross_validation_intent_level/"
    file = file_name
    sheet = 'intent_summary'
    df = pd.read_excel(file, sheet_name=sheet, index_col=0)
    df = df.rename(columns={'f1-score_cur':'F1-Score','support_ratio':'Imbalanced Ratio', 'fp_cur':'False Positives', 'support':'Support', 'precision_cur': 'Precision', 'recall_cur':'Recall'})
    cols = ['F1-Score','Precision','Recall','Support','Imbalanced Ratio','False Positives']
    df = df[cols]
    df = df.reset_index(drop=False)
    df = df.rename(columns={'index':'Intent'})
    return df

def similar_phrases_compare(check_embed, compare_embed, check_text, compare_list, K_similar, similar_thres):
    cosine_scores = util.pytorch_cos_sim(check_embed, compare_embed)
    values,indices = cosine_scores.topk(K_similar)
    values = values.numpy()
    indices = indices.tolist()
    indices_list = [indices[0][i] for i in range(len(indices[0]))]
    compare_embed_text_list = [compare_list[i] for i in indices_list]
    similar_score_list = [values[0][i] for i in range(len(values[0]))]
    
    data = list(zip(compare_embed_text_list, similar_score_list))
    similar_df = pd.DataFrame(data, columns=['Similar Phrase','Similar Score'])
    similar_df = similar_df[similar_df['Similar Score'] > similar_thres]
    
    return similar_df[1:]

def filter_confused_intents(dataframe, max_confused_display):
    data = dataframe.loc[confused_intent]
    data = data.loc[~data.index.isin([confused_intent])]
    data = data[data > 0]
    data = data.sort_values(ascending=False)
    data = data[0:max_confused_display]
    data2 = data.reset_index(drop=False)
    data2 = data2.rename(columns={'index':'Confused Intent',confused_intent:'Count'})
    return data, data2
    

try:
    st.write("# NLU Confusion Analysis")
    uploaded_file = st.file_uploader("Upload Excel Data",type=['xlsx'])
    #if uploaded_file is not None:
        #file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        #st.write(file_details)

    #get initial data
    df = get_pred_matrix_data(uploaded_file)
    cross_df = get_cross_validation_data(uploaded_file)
    flag_df = get_flaged_data(uploaded_file)
    
    st.write("+ **High recall, but low precision**: the model is correctly predicting well, but also predicting when it shouldn't.")
    st.write("+ **Low recall, high precision**: the model is missing out on predicting when it should, but when it does predict it tends to do well.")
    st.write("+ **High recall, high precision**: the model is doing a good job. It is predicting when it should and it isn't missing out on predicting.")
    
    
    
    
    #display_flag_df = flag_df.reset_index(drop=False)
    specific = st.multiselect('Filter for a specific intent', list(flag_df['Intent']))

    if specific == []:
        st.dataframe(flag_df.style.format({"F1-Score":"{:.2f}","False Positives": "{:,}", "Imbalanced Ratio":"{:.2f}", "Precision":"{:.2f}", "Recall":"{:.2f}"}))
    else:
        b = flag_df[flag_df['Intent'].isin(specific)==True]
        st.dataframe(b.style.format({"F1-Score":"{:.2f}","False Positives": "{:,}", "Imbalanced Ratio":"{:.2f}", "Precision":"{:.2f}", "Recall":"{:.2f}"}))
    
    #st.dataframe(flag_df.reset_index(drop=False))
    
    
    #setup layout
    
    st.write("## Analyze Intent")
    confused_intent = st.selectbox('Choose an intent to analyze', list(df.index))
    
    max_confused_intents_display = st.sidebar.slider('Max Confused Intents to Display:', min_value=1, max_value=20, step=1, value=5)
    
    top_K_similar = st.sidebar.slider('Max Similar Phrases to Display:', min_value=1, max_value=20, step=1, value=5)
    similar_thres = st.sidebar.slider('Max Similar Similar Threshold:', min_value=0.00, max_value=1.00, step=0.05, value=0.5)
    
    nlu_model_choice = st.sidebar.selectbox('NLU Pre-trained Models',
                                     ('bert-base-uncased', 'all-mpnet-base-v2', 'all-distilroberta-v1'))
    
    model = load_pretrained_model(nlu_model_choice)

    #intents = st.multiselect("Choose Intents", list(df.index), [])
    
    if not confused_intent:
        st.error("Please select at least one intent.")
        
    
    else:
        st.write("### False Negatives")
        data, data2 = filter_confused_intents(df, max_confused_intents_display)
        st.write("Utterances incorrectly predicted as another intent when the intent should have been **{}**".format(confused_intent))
        
        st.dataframe(data2)
        
        similar_intent = st.selectbox('Choose confused intent', list(data.index))
        st.write("## ")
        column_1, column_2 = st.columns(2)
        
        confused_phrases_df = cross_df[['text','pred_conf']][(cross_df['actual_intent']==confused_intent) & (cross_df['pred_intent']==similar_intent)].set_index('text')
        confused_phrases_df = confused_phrases_df.sort_values(by='pred_conf', ascending=False)
        similar_phrases_df = cross_df[['text','pred_conf']][(cross_df['actual_intent']==similar_intent)].set_index('text')
        
        
        with column_1:
            st.write('##### Confused Phrases')
            st.write('Phrases from **{}** that got confused with **{}**'.format(confused_intent, similar_intent))
            
            #st.write(confused_phrases_df.reset_index(drop=False))
            display_confused = confused_phrases_df.reset_index(drop=False)
            st.dataframe(display_confused.style.format({"pred_conf": "{:.2f}"}))
            
        
        with column_2:
            st.write('##### Similar Phrases')
            check_text = st.selectbox('Choose a confused phrase see a similar phrases in {}'.format(similar_intent), list(confused_phrases_df.index))
            check_embed = model.encode(check_text)
            compare_embed = model.encode(list(similar_phrases_df.index.values))
            #st.write(similar_phrases_df.index.values)
            
            a = similar_phrases_compare(check_embed, compare_embed, check_text, list(similar_phrases_df.index.values), top_K_similar, similar_thres)
            
            #st.write(a.style.format("{:.2f}"))
            #st.dataframe(a.style.format({"E":.2%}"))
            #st.dataframe(a)
            st.table(a.style.format({"Similar Score": "{:.2f}"}))
            
    st.write("### False Positives")


                    
except Exception as e:
    print(e)
    print('error')