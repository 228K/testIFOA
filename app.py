import numpy as np
import streamlit as st
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb


def pairplot(df, feat, hue):
    if (feat is not None or hue is not None) and len(feat)>1:
        fig, ax = plt.subplots()
        sns.pairplot(data=df[feat], hue=hue, height=3.5)
        return fig

def heatmap(df):
    fig, ax = plt.subplots() #??
    sns.heatmap(data=df.corr(numeric_only=True), ax=ax, annot=True, cmap='Blues')
    return fig

if __name__ == '__main__':
    st.title('Analisi di IRIS')
    upload_file= st.file_uploader('sceglie il file')#, type={'csv','xlsx'})
    if upload_file is not None:
        if upload_file.name.endswith('.csv'):
            df = pd.read_csv(upload_file,header=None)
        elif upload_file.name.endswith('.xlsx'):
            df=pd.read_excel(upload_file,header=None)
        
        df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
        cate= [x for x,t in zip(df.columns.to_list(), df.dtypes) if t.kind not in {'i','f'}]

        st.header('Prime 5 righe')
        st.dataframe(df.head())
        st.header('statistiche del DataFrame')
        st.dataframe(df.describe())

        st.header('Scegli la target')
        selec_target = st.selectbox('',cate + [None])

        st.header('Opzioni di visualizzazione')
        features= st.multiselect('Scegli le feauture',
                                 options=df.columns.to_list(),
                                 help='feature che vuoi visualizzare nel pairplot',
                                 default=df.columns.to_list()[:2])

        #hue

        features.append(selec_target)
        hue= st.selectbox('Scegli come differenziare i dati',
                                 cate + [None],
                                 help='feature che vuoi visualizzare nel pairplot')
        st.header('pairplot')
        st.pyplot(sns.pairplot(df,hue=hue))

        st.header('heatmap')
        st.pyplot(heatmap(df))


        st.title('ML')
        rfc = jb.load(filename='model_rfc')
        #SIDEBAR
        # max_sepl_len= float(df['sepal length'].max())
        # x1= st.slider('Inserisci il valore della lunghezza del sepalo',0.0,max_sepl_len+1.5, .5) #testo, start, stop, default
        # max_sepl_witd= float(df['sepal width'].max())
        # x2= st.slider('Inserisci il valore della lunghezza del sepalo',0.0,max_sepl_witd+1.5, .5) #testo, start, stop, default
        # max_petal_len= float(df['petal length'].max())
        # x3= st.slider('Inserisci il valore della lunghezza del sepalo',0.0,max_petal_len+1.5, .5) #testo, start, stop, default
        # max_petal_witd= float(df['petal width'].max())
        # x4= st.slider('Inserisci il valore della lunghezza del sepalo',0.0,max_petal_witd+1.5, .5) #testo, start, stop, default
        # st.text(max_petal_witd)
        # st.text(max_petal_len)
        # st.text( max_sepl_witd)
        # st.text(max_sepl_len)
        #
        # if st.button('predict'):
        #     data = {
        #         'sepal length': [x1],
        #         'sepal width': [x2],
        #         'petal length': [x3],
        #         'petal width': [x4]
        #     }
        #     inpute_df = pd.DataFrame(data)
        #     res = rfc.predict(inpute_df)[0]
        #     st.success(res)


        X= df.drop(columns=selec_target)
        target= df[selec_target]
        inpute_df = pd.DataFrame(X)
        res = rfc.predict(inpute_df)
        data=pd.DataFrame(zip(target,res), columns=['reale', 'predetto'])
        st.dataframe(data)
        st.balloons()
        #creare un excel per


