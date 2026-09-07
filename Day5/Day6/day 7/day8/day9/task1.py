import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Autonomous Data Scientist",
    page_icon="🤖"
)

st.title("🤖 Autonomous Data Scientist")

st.write(
    "Upload your dataset and let the system analyze it automatically."
)


# =========================================================
# STEP 1 - UPLOAD DATASET
# =========================================================

st.header("📂 Step 1 - Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # -----------------------------------------------------
    # READ DATASET
    # -----------------------------------------------------

    data = pd.read_csv(uploaded_file)

    st.success("✅ Dataset uploaded successfully!")


    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.header("👀 Dataset Preview")

    st.dataframe(
        data.head()
    )


    # =====================================================
    # STEP 2 - AUTOMATIC DATA ANALYSIS
    # =====================================================

    st.header("📊 Step 2 - Automatic Data Analysis")


    # -----------------------------------------------------
    # Dataset Shape
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            data.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            data.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            data.isnull().sum().sum()
        )


    # -----------------------------------------------------
    # Numerical Columns
    # -----------------------------------------------------

    numerical_columns = data.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


    # -----------------------------------------------------
    # Categorical Columns
    # -----------------------------------------------------

    categorical_columns = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


    st.subheader("🔢 Numerical Columns")

    if len(numerical_columns) > 0:
        st.write(numerical_columns)
    else:
        st.info("No numerical columns found.")


    st.subheader("🔤 Categorical Columns")

    if len(categorical_columns) > 0:
        st.write(categorical_columns)
    else:
        st.info("No categorical columns found.")


    # -----------------------------------------------------
    # Column Information
    # -----------------------------------------------------

    st.subheader("🔍 Column Information")

    info = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str),
        "Missing Values": data.isnull().sum().values,
        "Unique Values": data.nunique().values
    })

    st.dataframe(info)


    # -----------------------------------------------------
    # Duplicate Rows
    # -----------------------------------------------------

    st.subheader("📋 Duplicate Rows")

    duplicate_count = data.duplicated().sum()

    if duplicate_count > 0:

        st.warning(
            f"⚠️ Found {duplicate_count} duplicate rows."
        )

    else:

        st.success(
            "✅ No duplicate rows found."
        )


    # -----------------------------------------------------
    # Missing Values
    # -----------------------------------------------------

    st.subheader("❓ Missing Values")

    missing = data.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) > 0:

        st.dataframe(missing)

    else:

        st.success(
            "✅ No missing values found."
        )


    # -----------------------------------------------------
    # Statistical Summary
    # -----------------------------------------------------

    st.subheader("📈 Statistical Summary")

    if len(numerical_columns) > 0:

        st.dataframe(
            data[numerical_columns].describe()
        )

    else:

        st.info(
            "No numerical columns available."
        )


    # =====================================================
    # STEP 3 - AUTOMATIC DATA CLEANING
    # =====================================================

    st.header("🧹 Step 3 - Automatic Data Cleaning")


    cleaned_data = data.copy()


    # -----------------------------------------------------
    # Count duplicates
    # -----------------------------------------------------

    duplicates_before = cleaned_data.duplicated().sum()


    # -----------------------------------------------------
    # Remove duplicates
    # -----------------------------------------------------

    cleaned_data = cleaned_data.drop_duplicates()


    # -----------------------------------------------------
    # Fill numerical missing values
    # -----------------------------------------------------

    for column in cleaned_data.select_dtypes(
        include=["int64", "float64"]
    ).columns:

        if cleaned_data[column].isnull().sum() > 0:

            median_value = cleaned_data[column].median()

            cleaned_data[column] = cleaned_data[column].fillna(
                median_value
            )


    # -----------------------------------------------------
    # Fill categorical missing values
    # -----------------------------------------------------

    for column in cleaned_data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns:

        if cleaned_data[column].isnull().sum() > 0:

            mode_value = cleaned_data[column].mode()

            if len(mode_value) > 0:

                cleaned_data[column] = cleaned_data[column].fillna(
                    mode_value[0]
                )


    # -----------------------------------------------------
    # Cleaning Results
    # -----------------------------------------------------

    st.subheader("🔎 Cleaning Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Original Rows",
            data.shape[0]
        )

    with col2:

        st.metric(
            "Rows After Cleaning",
            cleaned_data.shape[0]
        )

    with col3:

        st.metric(
            "Duplicates Removed",
            duplicates_before
        )


    remaining_missing = cleaned_data.isnull().sum().sum()


    if remaining_missing == 0:

        st.success(
            "✅ Data cleaning completed successfully!"
        )

    else:

        st.warning(
            f"⚠️ {remaining_missing} missing values remain."
        )


    st.subheader("✨ Cleaned Dataset")

    st.dataframe(
        cleaned_data.head(20)
    )


    # =====================================================
    # STEP 4 - TARGET COLUMN DETECTION
    # =====================================================

    st.header("🎯 Step 4 - Target Column Detection")


    st.write(
        "Select the column that you want the system to predict."
    )


    columns = cleaned_data.columns.tolist()


    target_column = st.selectbox(
        "Select Target Column",
        columns
    )


    st.success(
        f"🎯 Selected Target Column: {target_column}"
    )


    # -----------------------------------------------------
    # Target Information
    # -----------------------------------------------------

    target_data = cleaned_data[target_column]


    st.subheader("📊 Target Information")


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Data Type",
            str(target_data.dtype)
        )

    with col2:

        st.metric(
            "Unique Values",
            target_data.nunique()
        )

    with col3:

        st.metric(
            "Missing Values",
            target_data.isnull().sum()
        )


    # -----------------------------------------------------
    # Problem Type Detection
    # -----------------------------------------------------

    st.subheader("🤖 Problem Type Detection")


    if (
        target_data.dtype == "object"
        or target_data.dtype == "category"
        or target_data.dtype == "bool"
        or target_data.nunique() <= 10
    ):

        problem_type = "Classification"

        st.success(
            "🟢 Classification problem detected"
        )

    else:

        problem_type = "Regression"

        st.success(
            "🔵 Regression problem detected"
        )


    st.write(
        f"Detected Problem Type: **{problem_type}**"
    )


    # =====================================================
    # STEP 5 - AUTOMATIC MODEL SELECTION
    # =====================================================

    st.header("🤖 Step 5 - Automatic Model Selection")


    # -----------------------------------------------------
    # Separate X and y
    # -----------------------------------------------------

    X = cleaned_data.drop(
        columns=[target_column]
    )

    y = cleaned_data[target_column]


    # -----------------------------------------------------
    # Convert categorical features
    # -----------------------------------------------------

    X = pd.get_dummies(
        X,
        drop_first=True
    )


    # Convert boolean columns
    for column in X.columns:

        if X[column].dtype == "bool":

            X[column] = X[column].astype(int)


    # -----------------------------------------------------
    # Convert target
    # -----------------------------------------------------

    label_encoder = None


    if problem_type == "Classification":

        if (
            y.dtype == "object"
            or y.dtype == "category"
            or y.dtype == "bool"
        ):

            label_encoder = LabelEncoder()

            y = label_encoder.fit_transform(y)


    # -----------------------------------------------------
    # Remove invalid rows
    # -----------------------------------------------------

    valid_rows = pd.notna(y)

    X = X.loc[valid_rows]

    if hasattr(y, "loc"):

        y = y.loc[valid_rows]


    # -----------------------------------------------------
    # Fill missing feature values
    # -----------------------------------------------------

    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)


    # -----------------------------------------------------
    # Train/Test Split
    # -----------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    except Exception as e:

        st.error(
            f"❌ Train-test split failed: {e}"
        )

        st.stop()


    # -----------------------------------------------------
    # Select Models
    # -----------------------------------------------------

    if problem_type == "Classification":

        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
        }

    else:

        models = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
        }


    # -----------------------------------------------------
    # Train Models
    # -----------------------------------------------------

    results = []


    for name, model in models.items():

        try:

            model.fit(
                X_train,
                y_train
            )


            model_prediction = model.predict(
                X_test
            )


            if problem_type == "Classification":

                score = accuracy_score(
                    y_test,
                    model_prediction
                )


                results.append({
                    "Model": name,
                    "Accuracy": score
                })


            else:

                mse = mean_squared_error(
                    y_test,
                    model_prediction
                )


                results.append({
                    "Model": name,
                    "MSE": mse
                })


        except Exception as e:

            st.warning(
                f"⚠️ {name} could not be trained: {e}"
            )


    # -----------------------------------------------------
    # Model Comparison
    # -----------------------------------------------------

    if len(results) == 0:

        st.error(
            "❌ No model could be trained."
        )

        st.stop()


    results_df = pd.DataFrame(
        results
    )


    st.subheader("🏆 Model Comparison")

    st.dataframe(
        results_df
    )


    # -----------------------------------------------------
    # Select Best Model
    # -----------------------------------------------------

    if problem_type == "Classification":

        best_index = results_df[
            "Accuracy"
        ].idxmax()

    else:

        best_index = results_df[
            "MSE"
        ].idxmin()


    best_model_name = results_df.loc[
        best_index,
        "Model"
    ]


    st.success(
        f"🏆 Best Model: **{best_model_name}**"
    )


    # =====================================================
    # STEP 6 - AUTOMATIC MODEL EVALUATION
    # =====================================================

    st.header("📊 Step 6 - Automatic Model Evaluation")


    # -----------------------------------------------------
    # Create Best Model
    # -----------------------------------------------------

    if problem_type == "Classification":

        if best_model_name == "Logistic Regression":

            best_model = LogisticRegression(
                max_iter=1000
            )

        elif best_model_name == "Decision Tree":

            best_model = DecisionTreeClassifier(
                random_state=42
            )

        else:

            best_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )


    else:

        if best_model_name == "Linear Regression":

            best_model = LinearRegression()

        elif best_model_name == "Decision Tree":

            best_model = DecisionTreeRegressor(
                random_state=42
            )

        else:

            best_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )


    # -----------------------------------------------------
    # Train Best Model
    # -----------------------------------------------------

    best_model.fit(
        X_train,
        y_train
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = best_model.predict(
        X_test
    )


    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    if problem_type == "Classification":

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        precision = precision_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.2%}"
            )


        with col2:

            st.metric(
                "Precision",
                f"{precision:.2%}"
            )


        with col3:

            st.metric(
                "Recall",
                f"{recall:.2%}"
            )


        with col4:

            st.metric(
                "F1 Score",
                f"{f1:.2%}"
            )


    else:

        mae = mean_absolute_error(
            y_test,
            prediction
        )


        mse = mean_squared_error(
            y_test,
            prediction
        )


        rmse = mse ** 0.5


        r2 = r2_score(
            y_test,
            prediction
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "MAE",
                f"{mae:.3f}"
            )


        with col2:

            st.metric(
                "MSE",
                f"{mse:.3f}"
            )


        with col3:

            st.metric(
                "RMSE",
                f"{rmse:.3f}"
            )


        with col4:

            st.metric(
                "R² Score",
                f"{r2:.3f}"
            )


    st.success(
        f"✅ Evaluation completed using {best_model_name}"
    )


    # =====================================================
    # STEP 7 - AUTOMATIC PREDICTION
    # =====================================================

    st.header("🔮 Step 7 - Make a Prediction")


    st.write(
        "Enter new values and the selected model "
        "will generate a prediction."
    )


    prediction_inputs = {}


    # -----------------------------------------------------
    # Create Input Fields
    # -----------------------------------------------------

    original_features = cleaned_data.drop(
        columns=[target_column]
    )


    for column in original_features.columns:

        if pd.api.types.is_numeric_dtype(
            cleaned_data[column]
        ):

            median_value = cleaned_data[column].median()

            if pd.isna(median_value):

                median_value = 0.0


            prediction_inputs[column] = st.number_input(
                f"Enter {column}",
                value=float(median_value)
            )


        else:

            options = (
                cleaned_data[column]
                .dropna()
                .unique()
                .tolist()
            )


            if len(options) > 0:

                prediction_inputs[column] = st.selectbox(
                    f"Select {column}",
                    options
                )


    # -----------------------------------------------------
    # Prediction Button
    # -----------------------------------------------------

    if st.button("🔮 Predict"):

        new_data = pd.DataFrame(
            [prediction_inputs]
        )


        # Convert categorical variables
        new_data = pd.get_dummies(
            new_data,
            drop_first=True
        )


        # Convert boolean columns
        for column in new_data.columns:

            if new_data[column].dtype == "bool":

                new_data[column] = new_data[column].astype(int)


        # Match training columns
        new_data = new_data.reindex(
            columns=X.columns,
            fill_value=0
        )


        # Prediction
        new_prediction = best_model.predict(
            new_data
        )


        # Decode classification
        if (
            problem_type == "Classification"
            and label_encoder is not None
        ):

            result = label_encoder.inverse_transform(
                new_prediction.astype(int)
            )[0]

        else:

            result = new_prediction[0]


        st.success(
            f"🎯 Prediction: **{result}**"
        )


    # =====================================================
    # STEP 8 - AUTOMATIC EDA VISUALIZATIONS
    # =====================================================

    st.header("📈 Step 8 - Automatic EDA Visualizations")


    st.write(
        "The system automatically generates visualizations "
        "to understand the dataset."
    )


    # -----------------------------------------------------
    # Numerical Distribution
    # -----------------------------------------------------

    if len(numerical_columns) > 0:

        st.subheader(
            "📊 Numerical Data Distribution"
        )


        selected_column = st.selectbox(
            "Select numerical column",
            numerical_columns,
            key="distribution_column"
        )


        fig, ax = plt.subplots()


        ax.hist(
            cleaned_data[selected_column].dropna(),
            bins=20
        )


        ax.set_xlabel(
            selected_column
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.set_title(
            f"Distribution of {selected_column}"
        )


        st.pyplot(fig)

        plt.close(fig)


    # -----------------------------------------------------
    # Box Plot
    # -----------------------------------------------------

    if len(numerical_columns) > 0:

        st.subheader(
            "📦 Outlier Detection"
        )


        selected_box_column = st.selectbox(
            "Select column for outlier analysis",
            numerical_columns,
            key="box_plot_column"
        )


        fig, ax = plt.subplots()


        ax.boxplot(
            cleaned_data[
                selected_box_column
            ].dropna()
        )


        ax.set_ylabel(
            selected_box_column
        )

        ax.set_title(
            f"Box Plot - {selected_box_column}"
        )


        st.pyplot(fig)

        plt.close(fig)


    # -----------------------------------------------------
    # Correlation Heatmap
    # -----------------------------------------------------

    if len(numerical_columns) >= 2:

        st.subheader(
            "🔥 Correlation Heatmap"
        )


        correlation = cleaned_data[
            numerical_columns
        ].corr()


        fig, ax = plt.subplots(
            figsize=(10, 6)
        )


        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )


        ax.set_title(
            "Feature Correlation"
        )


        st.pyplot(fig)

        plt.close(fig)


    else:

        st.info(
            "At least two numerical columns are required "
            "for correlation analysis."
        )


    # -----------------------------------------------------
    # EDA Summary
    # -----------------------------------------------------

    st.subheader(
        "🤖 EDA Summary"
    )


    st.write(
        f"• Numerical features detected: "
        f"**{len(numerical_columns)}**"
    )


    st.write(
        f"• Categorical features detected: "
        f"**{len(categorical_columns)}**"
    )


    if len(numerical_columns) >= 2:

        correlation = cleaned_data[
            numerical_columns
        ].corr()


        correlation_values = (
            correlation
            .where(
                ~correlation.eq(1)
            )
            .abs()
            .stack()
        )


        if len(correlation_values) > 0:

            highest_correlation = (
                correlation_values.max()
            )


            st.write(
                f"• Strongest feature correlation: "
                f"**{highest_correlation:.2f}**"
            )


    st.success(
        "📊 Automatic EDA completed!"
    )


    # =====================================================
    # STEP 9 - AUTOMATIC FEATURE ENGINEERING
    # =====================================================

    st.header(
        "⚙️ Step 9 - Automatic Feature Engineering"
    )


    st.write(
        "The system automatically prepares features "
        "for machine learning."
    )


    # Create feature dataset
    feature_data = cleaned_data.copy()


    # Remove target
    features = feature_data.drop(
        columns=[target_column]
    )


    target = feature_data[target_column]


    # Convert categorical features
    features = pd.get_dummies(
        features,
        drop_first=True
    )


    # Convert boolean columns
    for column in features.columns:

        if features[column].dtype == "bool":

            features[column] = features[column].astype(int)


    # Replace infinite values
    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )


    # Fill missing values
    features = features.fillna(
        features.median(numeric_only=True)
    )


    features = features.fillna(0)


    # Display
    st.subheader(
        "🔧 Engineered Features"
    )


    st.write(
        f"Original features: "
        f"**{len(cleaned_data.columns) - 1}**"
    )


    st.write(
        f"Features after engineering: "
        f"**{features.shape[1]}**"
    )


    st.dataframe(
        features.head()
    )


    st.success(
        "✅ Feature engineering completed successfully!"
    )


    # =====================================================
    # STEP 10 - AUTOMATIC FEATURE SELECTION
    # =====================================================

    st.header(
        "🎯 Step 10 - Automatic Feature Selection"
    )


    st.write(
        "The system automatically identifies the most "
        "important features for prediction."
    )


    total_features = features.shape[1]


    # -----------------------------------------------------
    # Select Features
    # -----------------------------------------------------

    if total_features == 0:

        st.error(
            "❌ No usable features found."
        )

        st.stop()


    elif total_features <= 5:

        selected_features = (
            features.columns.tolist()
        )


        st.info(
            f"Only {total_features} features detected. "
            "All features will be used."
        )


    else:

        k = min(
            5,
            total_features
        )


        if problem_type == "Classification":

            selector = SelectKBest(
                score_func=f_classif,
                k=k
            )

        else:

            selector = SelectKBest(
                score_func=f_regression,
                k=k
            )


        try:

            selector.fit(
                features,
                target
            )


            selected_features = (
                features.columns[
                    selector.get_support()
                ].tolist()
            )

        except Exception:

            # If feature selection fails,
            # use all features
            selected_features = (
                features.columns.tolist()
            )


            st.warning(
                "⚠️ Automatic feature selection "
                "could not be completed. "
                "All features will be used."
            )


    # -----------------------------------------------------
    # Create Selected Dataset
    # -----------------------------------------------------

    selected_features_data = features[
        selected_features
    ]


    # -----------------------------------------------------
    # Display Selected Features
    # -----------------------------------------------------

    st.subheader(
        "🏆 Selected Features"
    )


    for feature in selected_features:

        st.write(
            f"✅ {feature}"
        )


    # -----------------------------------------------------
    # Feature Selection Summary
    # -----------------------------------------------------

    st.subheader(
        "📊 Feature Selection Summary"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Before Selection",
            total_features
        )


    with col2:

        st.metric(
            "After Selection",
            len(selected_features)
        )


    st.success(
        "🎯 Automatic feature selection completed!"
    )


    # =====================================================
    # FINAL DATASET
    # =====================================================

    st.header(
        "🤖 Final ML Dataset"
    )


    st.dataframe(
        selected_features_data.head(20)
    )


    # =====================================================
    # FINAL REPORT
    # =====================================================

    st.header(
        "📋 Autonomous Data Scientist Report"
    )


    st.subheader(
        "🔍 Dataset Overview"
    )


    st.write(
        f"• Dataset contains "
        f"**{data.shape[0]} rows** and "
        f"**{data.shape[1]} columns**."
    )


    st.write(
        f"• **{len(numerical_columns)} numerical** "
        f"and **{len(categorical_columns)} categorical** "
        f"columns detected."
    )


    st.write(
        f"• **{duplicates_before} duplicate rows** "
        f"were removed."
    )


    st.write(
        f"• Target column: **{target_column}**"
    )


    st.write(
        f"• Problem type: **{problem_type}**"
    )


    st.subheader(
        "🤖 Model Analysis"
    )


    st.write(
        "• Multiple machine learning models were tested."
    )


    st.write(
        f"• Best model: **{best_model_name}**"
    )


    if problem_type == "Classification":

        st.write(
            f"• Accuracy: **{accuracy:.2%}**"
        )

        st.write(
            f"• Precision: **{precision:.2%}**"
        )

        st.write(
            f"• Recall: **{recall:.2%}**"
        )

        st.write(
            f"• F1 Score: **{f1:.2%}**"
        )


        if accuracy >= 0.90:

            st.success(
                "🟢 Excellent model performance."
            )

        elif accuracy >= 0.75:

            st.info(
                "🟡 Good model performance."
            )

        else:

            st.warning(
                "🔴 Model performance is relatively low."
            )


    else:

        st.write(
            f"• MAE: **{mae:.3f}**"
        )

        st.write(
            f"• RMSE: **{rmse:.3f}**"
        )

        st.write(
            f"• R² Score: **{r2:.3f}**"
        )


        if r2 >= 0.90:

            st.success(
                "🟢 Excellent regression performance."
            )

        elif r2 >= 0.70:

            st.info(
                "🟡 Good regression performance."
            )

        else:

            st.warning(
                "🔴 Regression performance could be improved."
            )


    st.subheader(
        "💡 Autonomous Recommendation"
    )


    st.write(
        f"Based on the automatic analysis, "
        f"**{best_model_name}** is currently the "
        f"recommended model for predicting "
        f"**{target_column}**."
    )


    st.success(
        "🤖 Autonomous analysis completed successfully!"
    )
    # =====================================================
# STEP 11 - AUTOMATIC FEATURE IMPORTANCE
# =====================================================

st.header("🔥 Step 11 - Automatic Feature Importance")

st.write(
    "The system automatically identifies which features "
    "have the greatest influence on the prediction."
)


# -----------------------------------------------------
# Check whether model supports feature importance
# -----------------------------------------------------

if hasattr(best_model, "feature_importances_"):

    importance_values = best_model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance_values
    })

    # Sort from highest to lowest
    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )


    # -------------------------------------------------
    # Display Feature Importance
    # -------------------------------------------------

    st.subheader("🏆 Feature Importance Ranking")

    st.dataframe(
        feature_importance
    )


    # -------------------------------------------------
    # Top Features
    # -------------------------------------------------

    st.subheader("🥇 Most Important Features")

    top_features = feature_importance.head(5)

    for index, row in top_features.iterrows():

        st.write(
            f"🔹 **{row['Feature']}** "
            f"→ Importance: **{row['Importance']:.3f}**"
        )


    # -------------------------------------------------
    # Feature Importance Graph
    # -------------------------------------------------

    st.subheader("📊 Feature Importance Graph")

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Top 5 Important Features"
    )

    ax.invert_yaxis()

    st.pyplot(fig)

    plt.close(fig)


    # -------------------------------------------------
    # Autonomous Insight
    # -------------------------------------------------

    most_important_feature = (
        feature_importance.iloc[0]["Feature"]
    )

    highest_importance = (
        feature_importance.iloc[0]["Importance"]
    )


    st.subheader("🤖 Autonomous Insight")

    st.success(
        f"💡 The most important feature is "
        f"**{most_important_feature}** "
        f"with an importance score of "
        f"**{highest_importance:.3f}**."
    )


else:

    st.info(
        "ℹ️ Feature importance is not directly available "
        "for this selected model."
    )
    # =====================================================
# STEP 12 - AUTONOMOUS INSIGHTS
# =====================================================

st.header("🧠 Step 12 - Autonomous Insights")

st.write(
    "The system automatically analyzes the dataset and "
    "generates meaningful insights."
)


# -----------------------------------------------------
# Dataset Insights
# -----------------------------------------------------

st.subheader("📊 Dataset Insights")


# Dataset size
st.write(
    f"🔹 The dataset contains **{data.shape[0]} rows** "
    f"and **{data.shape[1]} columns**."
)


# Numerical and categorical features
st.write(
    f"🔹 The dataset contains **{len(numerical_columns)} "
    f"numerical features** and **"
    f"{len(categorical_columns)} categorical features**."
)


# Missing values
total_missing = data.isnull().sum().sum()

if total_missing == 0:

    st.success(
        "✅ No missing values were detected."
    )

else:

    st.warning(
        f"⚠️ The dataset contains "
        f"**{total_missing} missing values**."
    )


# Duplicate values
if duplicates_before == 0:

    st.success(
        "✅ No duplicate rows were detected."
    )

else:

    st.warning(
        f"⚠️ **{duplicates_before} duplicate rows** "
        f"were removed during cleaning."
    )


# -----------------------------------------------------
# Target Insights
# -----------------------------------------------------

st.subheader("🎯 Target Insights")

st.write(
    f"🔹 Target column selected: **{target_column}**"
)

st.write(
    f"🔹 Problem type detected: **{problem_type}**"
)

st.write(
    f"🔹 Target contains **"
    f"{target_data.nunique()} unique values**."
)


# -----------------------------------------------------
# Model Insights
# -----------------------------------------------------

st.subheader("🤖 Model Insights")

st.write(
    f"🔹 Best performing model: **{best_model_name}**"
)


if problem_type == "Classification":

    if accuracy >= 0.90:

        st.success(
            f"🟢 Model accuracy is **{accuracy:.2%}**, "
            "which indicates excellent performance."
        )

    elif accuracy >= 0.75:

        st.info(
            f"🟡 Model accuracy is **{accuracy:.2%}**, "
            "which indicates reasonable performance."
        )

    else:

        st.warning(
            f"🔴 Model accuracy is only **{accuracy:.2%}**. "
            "The model may need better features or more data."
        )


else:

    if r2 >= 0.90:

        st.success(
            f"🟢 R² score is **{r2:.3f}**, "
            "indicating excellent regression performance."
        )

    elif r2 >= 0.70:

        st.info(
            f"🟡 R² score is **{r2:.3f}**, "
            "indicating reasonable regression performance."
        )

    else:

        st.warning(
            f"🔴 R² score is **{r2:.3f}**. "
            "The regression model could be improved."
        )


# -----------------------------------------------------
# Feature Insight
# -----------------------------------------------------

st.subheader("🔥 Feature Insight")

if "feature_importance" in locals():

    most_important_feature = (
        feature_importance.iloc[0]["Feature"]
    )

    most_important_score = (
        feature_importance.iloc[0]["Importance"]
    )

    st.write(
        f"🔹 The most important feature identified by "
        f"the model is **{most_important_feature}**."
    )

    st.write(
        f"🔹 Its importance score is "
        f"**{most_important_score:.3f}**."
    )

else:

    st.info(
        "Feature importance is not available for "
        "the selected model."
    )


# -----------------------------------------------------
# Autonomous Recommendation
# -----------------------------------------------------

st.subheader("💡 Autonomous Recommendation")


if problem_type == "Classification":

    if accuracy >= 0.90:

        recommendation = (
            "The selected model performs very well. "
            "It can be considered a strong candidate "
            "for prediction."
        )

    elif accuracy >= 0.75:

        recommendation = (
            "The model performs reasonably well. "
            "Additional feature engineering and "
            "data improvement may increase performance."
        )

    else:

        recommendation = (
            "The model performance is low. "
            "Consider collecting more data, "
            "improving features, and trying additional models."
        )

else:

    if r2 >= 0.90:

        recommendation = (
            "The regression model performs very well "
            "and explains most of the variation in the target."
        )

    elif r2 >= 0.70:

        recommendation = (
            "The regression model performs reasonably well. "
            "Further feature engineering may improve results."
        )

    else:

        recommendation = (
            "The regression model needs improvement. "
            "Consider better features, more data, "
            "and additional machine learning algorithms."
        )


st.info(
    f"🤖 **System Recommendation:** {recommendation}"
)


st.success(
    "🧠 Autonomous insights generated successfully!"
)
# =====================================================
# STEP 13 - AUTOMATIC PROFESSIONAL REPORT
# =====================================================

st.header("📋 Step 13 - Autonomous Data Scientist Report")

st.write(
    "The system automatically generates a professional "
    "summary of the complete analysis."
)


# -----------------------------------------------------
# REPORT - DATASET
# -----------------------------------------------------

st.subheader("📊 1. Dataset Analysis")

st.write(
    f"Dataset size: **{data.shape[0]} rows × {data.shape[1]} columns**"
)

st.write(
    f"Numerical features: **{len(numerical_columns)}**"
)

st.write(
    f"Categorical features: **{len(categorical_columns)}**"
)

st.write(
    f"Duplicate rows removed: **{duplicates_before}**"
)

st.write(
    f"Remaining missing values: **{cleaned_data.isnull().sum().sum()}**"
)


# -----------------------------------------------------
# REPORT - TARGET
# -----------------------------------------------------

st.subheader("🎯 2. Target Analysis")

st.write(
    f"Target column: **{target_column}**"
)

st.write(
    f"Problem type: **{problem_type}**"
)

st.write(
    f"Target unique values: **{target_data.nunique()}**"
)


# -----------------------------------------------------
# REPORT - MODEL
# -----------------------------------------------------

st.subheader("🤖 3. Machine Learning Analysis")

st.write(
    f"Models tested: **{len(results_df)}**"
)

st.write(
    f"Best model: **{best_model_name}**"
)


if problem_type == "Classification":

    st.write(
        f"Accuracy: **{accuracy:.2%}**"
    )

    st.write(
        f"Precision: **{precision:.2%}**"
    )

    st.write(
        f"Recall: **{recall:.2%}**"
    )

    st.write(
        f"F1 Score: **{f1:.2%}**"
    )

else:

    st.write(
        f"MAE: **{mae:.3f}**"
    )

    st.write(
        f"RMSE: **{rmse:.3f}**"
    )

    st.write(
        f"R² Score: **{r2:.3f}**"
    )


# -----------------------------------------------------
# REPORT - FEATURE ENGINEERING
# -----------------------------------------------------

st.subheader("⚙️ 4. Feature Engineering")

st.write(
    f"Features before engineering: "
    f"**{len(cleaned_data.columns) - 1}**"
)

st.write(
    f"Features after engineering: "
    f"**{features.shape[1]}**"
)

st.write(
    f"Selected features: "
    f"**{len(selected_features)}**"
)


# -----------------------------------------------------
# REPORT - IMPORTANT FEATURES
# -----------------------------------------------------

st.subheader("🔥 5. Important Features")

if "feature_importance" in locals():

    for index, row in feature_importance.head(5).iterrows():

        st.write(
            f"🔹 **{row['Feature']}** — "
            f"Importance: **{row['Importance']:.3f}**"
        )

else:

    st.info(
        "Feature importance is not available "
        "for this model."
    )


# -----------------------------------------------------
# REPORT - FINAL RECOMMENDATION
# -----------------------------------------------------

st.subheader("💡 6. Final Recommendation")

if problem_type == "Classification":

    if accuracy >= 0.90:

        final_recommendation = (
            "The selected classification model shows "
            "excellent performance and is a strong candidate "
            "for future predictions."
        )

    elif accuracy >= 0.75:

        final_recommendation = (
            "The classification model shows good performance. "
            "Additional feature engineering may improve it further."
        )

    else:

        final_recommendation = (
            "The classification model needs improvement. "
            "More data, better features, or additional models "
            "should be considered."
        )

else:

    if r2 >= 0.90:

        final_recommendation = (
            "The regression model shows excellent performance "
            "and explains most of the variation in the target."
        )

    elif r2 >= 0.70:

        final_recommendation = (
            "The regression model shows good performance. "
            "Further feature engineering may improve predictions."
        )

    else:

        final_recommendation = (
            "The regression model needs improvement. "
            "More data and better features should be considered."
        )


st.info(
    f"🤖 {final_recommendation}"
)


# -----------------------------------------------------
# FINAL STATUS
# -----------------------------------------------------

st.success(
    "📋 Professional autonomous report generated successfully!"
)

# =====================================================
# STEP 14 - DOWNLOAD REPORT
# =====================================================

st.header("💾 Step 14 - Download Analysis Report")

st.write(
    "Download the complete autonomous data science "
    "analysis as a text report."
)


# -----------------------------------------------------
# Create Report Text
# -----------------------------------------------------

report = f"""
========================================================
        AUTONOMOUS DATA SCIENTIST REPORT
========================================================

DATASET ANALYSIS
--------------------------------------------------------
Rows                 : {data.shape[0]}
Columns              : {data.shape[1]}
Numerical Features   : {len(numerical_columns)}
Categorical Features : {len(categorical_columns)}
Duplicate Rows       : {duplicates_before}
Missing Values       : {cleaned_data.isnull().sum().sum()}


TARGET ANALYSIS
--------------------------------------------------------
Target Column        : {target_column}
Problem Type         : {problem_type}
Unique Target Values : {target_data.nunique()}


MODEL ANALYSIS
--------------------------------------------------------
Best Model           : {best_model_name}
"""


# -----------------------------------------------------
# Add Classification Metrics
# -----------------------------------------------------

if problem_type == "Classification":

    report += f"""
Accuracy             : {accuracy:.2%}
Precision            : {precision:.2%}
Recall               : {recall:.2%}
F1 Score             : {f1:.2%}
"""


# -----------------------------------------------------
# Add Regression Metrics
# -----------------------------------------------------

else:

    report += f"""
MAE                  : {mae:.3f}
MSE                  : {mse:.3f}
RMSE                 : {rmse:.3f}
R2 Score             : {r2:.3f}
"""


# -----------------------------------------------------
# Feature Engineering
# -----------------------------------------------------

report += f"""

FEATURE ENGINEERING
--------------------------------------------------------
Original Features    : {len(cleaned_data.columns) - 1}
Engineered Features  : {features.shape[1]}
Selected Features    : {len(selected_features)}


SELECTED FEATURES
--------------------------------------------------------
"""

for feature in selected_features:

    report += f"- {feature}\n"


# -----------------------------------------------------
# Feature Importance
# -----------------------------------------------------

report += """

TOP IMPORTANT FEATURES
--------------------------------------------------------
"""

if "feature_importance" in locals():

    for index, row in feature_importance.head(5).iterrows():

        report += (
            f"- {row['Feature']} : "
            f"{row['Importance']:.3f}\n"
        )

else:

    report += (
        "Feature importance is not available "
        "for the selected model.\n"
    )


# -----------------------------------------------------
# Recommendation
# -----------------------------------------------------

report += f"""

FINAL RECOMMENDATION
--------------------------------------------------------
{final_recommendation}


========================================================
        END OF REPORT
========================================================
"""


# -----------------------------------------------------
# Display Report
# -----------------------------------------------------

st.subheader("📋 Generated Report")

st.text_area(
    "Report Preview",
    report,
    height=500
)


# -----------------------------------------------------
# Download Button
# -----------------------------------------------------

st.download_button(
    label="⬇️ Download Report",
    data=report,
    file_name="autonomous_data_scientist_report.txt",
    mime="text/plain"
)


st.success(
    "💾 Report is ready to download!"
)