
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

GLOBAL_SEED = 42

def enforce_absolute_determinism(seed=42):
    """Locks the Python environment, NumPy, and random module states for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    st.session_state['random_seed'] = seed # Store in session state for Streamlit

enforce_absolute_determinism(GLOBAL_SEED)
# 2. DATA AND PREPROCESSOR INITIALIZATION
@st.cache_resource # Cache the preprocessor to avoid re-fitting on every rerun
def load_data_and_initialize_preprocessor():
    # Re-define load_custom_dataset to ensure X_raw is available within the Streamlit context
    def load_custom_dataset():
        file_path = "/content/Master datasheet.xlsx"
        target_column = "FOS"

        if not os.path.exists(file_path):
            st.warning(f"[Warning] File '{file_path}' not found. Generating a structural placeholder dataset for testing...")
            df = pd.DataFrame(np.random.rand(200, 6), columns=['Cohesion (kN/m2)', ' Phi (deg)', 'Unit Weight (kN/m3)', 'Overall Bench Height', 'Overall Slope angle', ' Natural Moisture content'])
            df[target_column] = np.random.rand(200)
        else:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
        df = df.reset_index(drop=True)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    X_raw_app, _ = load_custom_dataset()

    numeric_features = X_raw_app.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Assuming no categorical features were explicitly handled in the original notebook for this CT
    categorical_features = [] # Keep empty if not used, or add if present in X_raw_app

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Recreate the ColumnTransformer using the original logic
    preprocessor_app = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough' # Keeps columns that don't need transformation
    )
    
    # Fit the preprocessor on the (re)loaded raw data
    preprocessor_app.fit(X_raw_app)
    st.success("Preprocessor initialized and fitted successfully!")
    
    return X_raw_app, preprocessor_app, numeric_features

X_raw_app, preprocessor_app, numeric_features_app = load_data_and_initialize_preprocessor()
# 3. LOAD OPTIMIZED MODELS
@st.cache_resource # Cache the loaded models to avoid reloading on every rerun
def load_optimized_models():
    models_filename = 'optimized_gradient_boosting_models.pkl'
    if os.path.exists(models_filename):
        with open(models_filename, 'rb') as file:
            loaded_models = pickle.load(file)
        st.success(f"Optimized models loaded successfully from '{models_filename}'")
        return loaded_models
    else:
        st.error(f"Error: Model file '{models_filename}' not found. Please ensure it's in the same directory.")
        st.stop() # Stop the app if models aren't found


loaded_models = load_optimized_models()

final_model_cdo = loaded_models.get('CDO_Optimized_Model')
final_model_bbo = loaded_models.get('BBO_Optimized_Model')
final_model_ewo = loaded_models.get('EWO_Optimized_Model')
final_model_gto = loaded_models.get('GTO_Optimized_Model')

ALL_MODELS = {
    'CDO_Optimized_Model': final_model_cdo,
    'BBO_Optimized_Model': final_model_bbo,
    'EWO_Optimized_Model': final_model_ewo,
    'GTO_Optimized_Model': final_model_gto
}

# 4. HELPER FUNCTIONS: PREPROCESSING AND PREDICTION
def preprocess_new_data(new_raw_data: pd.DataFrame) -> np.ndarray:
    """
    Applies the same preprocessing (imputation, scaling) to new raw data
    as was applied to the training data, using the cached preprocessor.

    Args:
        new_raw_data (pd.DataFrame): A DataFrame containing the raw new data,
                                     with columns matching the original X_raw features.

    Returns:
        np.ndarray: The preprocessed data ready for model prediction.
    """
    # Ensure the input DataFrame has the same columns as the original raw data
    if not all(col in new_raw_data.columns for col in X_raw_app.columns):
        # This check should be more robust for production, potentially reordering columns
        st.error("New data columns do not match the original training data columns.")
        st.stop()

    processed_data = preprocessor_app.transform(new_raw_data)
    return processed_data

def make_predictions(new_raw_data: pd.DataFrame) -> dict:
    """
    Preprocesses new raw data and makes predictions using all optimized models.

    Args:
        new_raw_data (pd.DataFrame): A DataFrame containing the raw new data,
                                     with columns matching the original X_raw features.

    Returns:
        dict: A dictionary where keys are model names and values are their predictions.
    """
    try:
        processed_input = preprocess_new_data(new_raw_data)
        predictions = {
            model_name: model.predict(processed_input)[0]
            for model_name, model in ALL_MODELS.items()
        }
        return predictions
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return {"error": str(e)}

# 5. FEATURE IMPORTANCE VISUALIZATION FUNCTION
def get_feature_names(column_transformer):
    """
    Extracts feature names from a ColumnTransformer, including those from pipelines.
    """
    output_features = []
    for name, estimator, features in column_transformer.transformers_:
        if name == 'num':
            if hasattr(estimator, 'get_feature_names_out'):
                output_features.extend(estimator.get_feature_names_out(features))
            else:
                output_features.extend(features)
        elif name == 'remainder':
            output_features.extend(list(features)) # 'features' here is a list of column indices, need to map to original names if using 'passthrough'
    # For 'passthrough' or 'drop', the feature names are often the original column names if not dropped
    # Assuming simple numeric features as per the original CT, so original names are preserved after scaling
    if len(output_features) != column_transformer.transform(X_raw_app.head(1)).shape[1]:
        # Fallback if original column names aren't properly captured by get_feature_names_out
        st.warning("Adjusting feature names: count mismatch with preprocessed data dimensions.")
        return [f'feature_{i}' for i in range(column_transformer.transform(X_raw_app.head(1)).shape[1])]
    return output_features

feature_names_app = get_feature_names(preprocessor_app)
def plot_feature_importances(models_dict, feature_names):
    """
    Generates and returns a matplotlib figure with comparative feature importance plots.
    """
    all_feature_importances = {
        model_name: model.feature_importances_ for model_name, model in models_dict.items()
    }

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flatten()

    for i, (model_name, importances) in enumerate(all_feature_importances.items()):
        df_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        sns.barplot(x='Importance', y='Feature', data=df_importances, palette='viridis', ax=axs[i])
        axs[i].set_title(f'Feature Importance: {model_name}', fontsize=14)
        axs[i].set_xlabel('Relative Importance', fontsize=12)
        axs[i].set_ylabel('Feature Name', fontsize=12)
        axs[i].tick_params(axis='x', labelsize=10)
        axs[i].tick_params(axis='y', labelsize=10)

    plt.suptitle('Comparative Feature Importance Across Optimized Models', y=1.02, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent suptitle overlap
    return fig


# 6. STREAMLIT APPLICATION UI

st.set_page_config(layout="wide", page_title="Hyperparameter Optimized GBR Predictor")

st.title("🚀 Hyperparameter Optimized Gradient Boosting Regressor Predictor")
st.markdown("--- ")

st.sidebar.header("Input Features")

input_data = {}
for feature in X_raw_app.columns:
    default_value = X_raw_app[feature].mean()
    min_val = X_raw_app[feature].min()
    max_val = X_raw_app[feature].max()

    if pd.api.types.is_float_dtype(X_raw_app[feature]):
        input_data[feature] = st.sidebar.number_input(
            f"Enter {feature}",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_value),
            step=0.01,
            format="%.4f"
        )
    else:
        input_data[feature] = st.sidebar.number_input(
            f"Enter {feature}",
            min_value=int(min_val),
            max_value=int(max_val),
            value=int(default_value),
            step=1
        )

new_data_df = pd.DataFrame([input_data])

st.sidebar.markdown("--- ")

if st.sidebar.button("✨ Get Predictions"):
    st.header("📊 Predictions")
    with st.spinner('Making predictions...'):
        predictions = make_predictions(new_data_df)

        if "error" in predictions:
            st.error(f"Prediction Error: {predictions['error']}")
        else:
            st.write("Here are the predictions from each optimized model:")
            predictions_df = pd.DataFrame([predictions]).T.reset_index()
            predictions_df.columns = ['Model', 'Predicted Value']
            st.dataframe(predictions_df, hide_index=True)

            best_model_name = min(predictions, key=predictions.get)
            st.success(f"The **{best_model_name}** predicts the lowest value for this input.")

st.markdown("--- ")
st.header("📈 Model Insights")

if ALL_MODELS and feature_names_app:
    st.subheader("Feature Importance Across Models")
    try:
        fig_importance = plot_feature_importances(ALL_MODELS, feature_names_app)
        st.pyplot(fig_importance)
        st.markdown("Each plot shows the relative importance of features for each optimized Gradient Boosting Regressor model. A higher bar indicates a more significant contribution to the model's predictions.")
    except Exception as e:
        st.error(f"Could not generate feature importance plots: {e}")
else:
    st.warning("Models or feature names are not available to display feature importance.")

st.markdown("--- ")
st.info("To run this as a standalone Streamlit app: save all code cells into a file named `app.py`, then run `streamlit run app.py` in your terminal.")
