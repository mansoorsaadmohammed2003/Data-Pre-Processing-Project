from django.shortcuts import render

# Create your views here.

import numpy as np
import pandas as pd
from django.shortcuts import render
import pandas as pd
from django.shortcuts import render,HttpResponse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import skew
import os
from django.conf import settings


from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to my Django app!")



def perform_quality_check(df):
    # Ensure all numeric columns are converted properly
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
    
    # 1. Calculating missing value percentage
    missing = df.isnull().sum()
    missing_values = missing[(missing > 0)]
    sum_missing_values = sum(missing_values)
    rows = df.shape[0]
    columns = df.shape[1]
    total_data = rows * columns
    missing_value_percentage = round((sum_missing_values / total_data) * 100, 2)

    # 2. Calculating percentage of Outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5
    outliers = (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))
    outliers_count_per_column = outliers.sum()
    total_outliers = sum(outliers_count_per_column)
    outliers_percentage = round((total_outliers / total_data) * 100, 2)

    # 3. Calculating Skewness of the data
    skew_values = df.skew()
    positive_high_skew = skew_values[(skew_values > 1)]
    negative_high_skew = skew_values[(skew_values < -1)]
    no_skew_col = len(positive_high_skew) + len(negative_high_skew)
    percentage_skew = round((no_skew_col / columns) * 100, 2)

    # 4. Total data quality
    total_data_quality = round(100 - (missing_value_percentage + outliers_percentage + percentage_skew), 2)

    return {
        'missing_value_percentage': missing_value_percentage,
        'outliers_percentage': outliers_percentage,
        'skew_percentage': percentage_skew,
        'total_data_quality': total_data_quality
    }


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import skew
import os



def clean_data(df):
    # Make a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Imputation: fill missing values
    imputer = SimpleImputer(strategy='mean')
    cleaned_df[numerical_cols] = imputer.fit_transform(cleaned_df[numerical_cols])

    if not categorical_cols.empty:
        imputer = SimpleImputer(strategy='most_frequent')
        cleaned_df[categorical_cols] = imputer.fit_transform(cleaned_df[categorical_cols])

        # Encoding categorical variables
        encoder = OneHotEncoder(drop='first', sparse_output=False)  # Updated parameter name
        df_encoded = pd.DataFrame(encoder.fit_transform(cleaned_df[categorical_cols]))
        df_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    
        cleaned_df = pd.concat([cleaned_df.drop(columns=categorical_cols), df_encoded], axis=1)

    # Handling outliers using IQR
    Q1 = cleaned_df.quantile(0.25)
    Q3 = cleaned_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (cleaned_df < (Q1 - 1.5 * IQR)) | (cleaned_df > (Q3 + 1.5 * IQR))
    
    for col in outliers.columns:
        cleaned_df[col] = np.where(outliers[col], np.clip(cleaned_df[col], Q1[col], Q3[col]), cleaned_df[col])

    # Removing skewness using log transformation
    skewness = cleaned_df.apply(lambda x: skew(x.dropna()))
    skewed_cols = skewness[abs(skewness) > 1].index
    cleaned_df[skewed_cols] = cleaned_df[skewed_cols].apply(
        lambda x: np.log1p(x) if x.min() >= 0 else x
    )

    # Standardizing numerical variables
    num_cols_after_encoding = cleaned_df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    cleaned_df[num_cols_after_encoding] = scaler.fit_transform(cleaned_df[num_cols_after_encoding])

    return cleaned_df

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
def upload_file(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        global target_variable
        target_variable = request.POST.get('target_variable')
        drop_index = request.POST.get('drop_index')

        # Read the CSV file using pandas
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            error_message = f"Error reading CSV file: {str(e)}"
            return render(request, 'upload_form.html', {'error_message': error_message})
        
        if drop_index:
            index_col = df.columns[0]
            df.drop(index_col, axis=1, inplace=True)
        
        # Perform data quality assessment
        assessment_results = perform_quality_check(df)
        
        # Calculate complementary percentages
        assessment_results['complementary_missing_value_percentage'] = 100 - assessment_results['missing_value_percentage']
        assessment_results['complementary_outliers_percentage'] = 100 - assessment_results['outliers_percentage']
        assessment_results['complementary_skew_percentage'] = 100 - assessment_results['skew_percentage']
        
        # Clean the data
        global cleaned_df
        cleaned_df = clean_data(df)
        
        # Get the first five rows of cleaned data
        cleaned_df_first_five_rows = cleaned_df.head(5)

        df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns
        correlation = df_numeric.corr()[target_variable].sort_values()

        if(target_variable in df.columns):
            correlation = df.corr()[target_variable].sort_values()
            positive_high_correlation = correlation[(correlation > 0.6) & (correlation.index != target_variable)].index.tolist()
            negative_high_correlation = correlation[(correlation<-0.6)].index.tolist()
            slope_dic = {}
            covariance_matrix = df.cov()
            high_correlation = positive_high_correlation + negative_high_correlation
            for i in high_correlation:
                covariance = covariance_matrix.loc[i,target_variable]
                variance = df[i].var()
                slope = covariance / variance 
                slope_dic[i] = slope

        summary = cleaned_df.describe()    
        
        # Create media directory if it doesn't exist
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        
        # Save the cleaned data to a CSV file
        cleaned_file_path = os.path.join(settings.MEDIA_ROOT, 'cleaned_data.csv')
        cleaned_df.to_csv(cleaned_file_path, index=False)
        
        # Provide the file path to the template
        cleaned_file_url = os.path.join(settings.MEDIA_URL, 'cleaned_data.csv')
        
        return render(request, 'assessment_results.html', {
            'assessment_results': assessment_results, 
            'cleaned_file_url': cleaned_file_url,
            'cleaned_df_first_five_rows': cleaned_df_first_five_rows,
            'target_variable': target_variable,
            'positive_high_correlation': positive_high_correlation,
            'negative_high_correlation': negative_high_correlation,
            'slope_dic': slope_dic,
            'summary': summary
        })
    
    return render(request, 'upload_form.html')



from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def train_model(request):
    # Assuming cleaned data is stored in a DataFrame called `cleaned_df`
    
    # Split the data into features (X) and target variable (y)
    X = cleaned_df.drop(columns=[target_variable])
    y = cleaned_df[target_variable]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize a dictionary to store model performance
    model_performance = {}
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        # Add more models here if needed
    }
    
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        
        # Store model performance
        model_performance[model_name] = mse
    
    # Select the best model based on performance
    best_model_name = min(model_performance, key=model_performance.get)
    best_model = models[best_model_name]
    best_metrics = model_performance[best_model_name]
    
    # Create media directory if it doesn't exist
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    
    # Save the best model with a proper filename
    model_filename = f'best_model_{best_model_name.replace(" ", "_")}.joblib'
    model_filepath = os.path.join(settings.MEDIA_ROOT, model_filename)
    joblib.dump(best_model, model_filepath)
    
    return render(request, 'model_evaluation.html', {
        'best_model_name': best_model_name,
        'model_filename': model_filename,  # Pass just the filename
        'best_metrics': best_metrics
    })
















from django.http import HttpResponse, FileResponse

def download_cleaned_data(request):
    cleaned_file_path = os.path.join(settings.MEDIA_ROOT, 'cleaned_data.csv')

    if os.path.exists(cleaned_file_path):
        response = FileResponse(open(cleaned_file_path, 'rb'))
        response['Content-Disposition'] = 'attachment; filename="cleaned_data.csv"'
        return response
    else:
        return Http404("The cleaned data file is not available.")
    
def download_model(request, filename):
    model_path = os.path.join(settings.MEDIA_ROOT, filename)
    
    if os.path.exists(model_path):
        response = FileResponse(open(model_path, 'rb'))
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    else:
        raise Http404("Model file not found")


