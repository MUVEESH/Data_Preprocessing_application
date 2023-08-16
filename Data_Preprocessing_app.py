import streamlit as st
import pandas as pd
import base64
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV

def main():
    st.title('DATA PREPROCESSING')

    # Upload DataFrame
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        feature_engineering(df)
        anomaly=['none','z_score','iqr','isolation_forest']
        st.header("Outlier_Detection")
        outlier_detection=st.selectbox("Select a anomaly_detector", anomaly)
        if outlier_detection==z_score:
            z_score(df)
        elif outlier_detection==iqr:
            iqr(df)
        elif outlier_detection==isolation_forest:
            isolation_forest(df)
        missing_values_imputation(df)
        
        # Display modified DataFrame
        st.write("Modified DataFrame:")
        st.dataframe(df)

        # Provide download link for modified DataFrame
        st.markdown(get_download_link(df), unsafe_allow_html=True)

def feature_engineering(df):
    st.header("Feature Engineering")
    encoders=['none','OneHotEncoder','TargetEncoder']
    encoder=st.selectbox("Select a Encoder", encoders)
    if encoder=='OneHotEncoder':
        df1=pd.get_dummies(df)
        df.drop(df.columns, axis=1, inplace=True)
        df2=df+df1
        st.write(df2)
    elif encoder=='TargetEncoder':
        column = st.selectbox("Select a column for TargetEncoder", df.columns)
        for j,i in enumerate(df[column].unique()):
            cat={i:j+1}
            df[column].replace(cat,inplace=True)
    scalers=['none','MinMaxScaler']
    scaler=st.selectbox("Select a scaler", scalers)
    def minmax_scaler(data):
        min_val = data.min()
        max_val = data.max()
    
        scaled_data = (data - min_val) / (max_val - min_val)
    
        return scaled_data
    if scaler=='MinMaxScaler':
       df=df.apply(minmax_scaler)
    st.write('Done')

def z_score(df):
    column = st.selectbox("Select a column for Z-Score", df.columns)
    outliers=[]
    skewness = df[column].skew()
    kurt = df[column].kurtosis()
    mean=df[column].mean()
    std=df[column].std()
    for x in df[column]:
        z=(x-mean)/std
        if z>3 or z<-3:
            outliers.append(x)
    out=0
    for i in outliers:
        out+=1
    st.write(f'count of outliers: {out}')
    st.write(f'skewness: {skewness}\nKurtosis: {kurt}')
    st.write(f'Outliers: {outliers}')
    remove=st.radio('Do you want to remove outliers?', ['Yes', 'No'])
    if remove=='Yes':
        df.drop(outliers, inplace=True)
    st.write('Done')

def iqr(df):
    column = st.selectbox("Select a column for IQR", df.columns)
    outliers=[]
    q1=df[column].quantile(0.25)
    q3=df[column].quantile(0.75)
    IQR=q3-q1
    inner=q1-1.5*IQR
    outer=q3+1.5*IQR
    for x in df[column]:
        if x<inner or x>outer:
            outliers.append(x)
    out=0
    for i in outliers:
        out+=1
    st.write(f'count of outliers: {out}')
    remove=st.radio('Do you want to remove outliers?', ['Yes', 'No'])
    if remove=='Yes':
        df.drop(outliers, inplace=True)
    st.write('Done')

def isolation_forest(df):
    model = IsolationForest()
    grid_params = {
        'contamination': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'n_estimators': [100, 200],
        'max_features': [1],
        'max_samples': ['auto', 0.7, 0.8],
        'n_jobs': [-1]
    }
    grid_search = GridSearchCV(model, grid_params, scoring='roc_auc')
    grid_search.fit(df)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    # Train the best model
    best_model.fit(df)
    best_model.params=best_params
    # Get scores
    df['scores'] = best_model.decision_function(df)
    df['anomaly_score'] = best_model.predict(df)
    # Print
    outliers=df[df['anomaly_score']==-1]
    st.write(outliers)
    remove=st.radio('Do you want to remove outliers?', ['Yes', 'No'])
    if remove=='Yes':
        df.drop(outliers,axis=1,inplace=True)
        df.drop(['scores','anomaly_score'],axis=1,inplace=True)
    else:
        print('Done')
    return isolation_forest
def missing_values_imputation(df):
    st.header("Missing Values Imputation")
    # Your missing values imputation code here
    st.write(f'{df.isna().sum()}')
    option=['none','yes','no']
    impute=st.selectbox('select yes or no to impute ',option)
    
    try:
        if impute=='yes':
            imputor_names=['none','median','mode',iterative]
            imputor=st.selectbox('select imputer name')
            column = st.selectbox("Select a column for simple imputation", df.columns)
            if imputor==median:
                df.fillna(df[column].median())
            elif imputor==mode:
                df.fillna(df[column].mode())
            elif imputor==iterative:
                from sklearn.impute import IterativeImputer
                imputer=IterativeImputer(max_iter=10,random_state=0)
                imputer.fit_transform(df)
        else:
            print('Done')
    except NameError:
        print('Error:You should type only listed - kindly rerun')

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="modified_data.csv">Download Modified CSV</a>'
    return href

if __name__ == '__main__':
    main()
