# ============================================================
# TELECOM CUSTOMER CHURN PREDICTION WEB APPLICATION
# Framework: Streamlit
# Models: Logistic Regression, SVM, KNN
# ============================================================

import json
import joblib
import streamlit as st
from pathlib import Path
import pandas as pd
# import page.py from app/features/model_consensus

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CONFIG_PATH = MODELS_DIR / "app_config.json"

MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "logistic_regression_pipeline.pkl",
    "Support Vector Machine (SVM)": MODELS_DIR / "svm_pipeline.pkl",
    "K-Nearest Neighbors (KNN)": MODELS_DIR / "knn_pipeline.pkl",
}


# ------------------------------------------------------------
# 2. MODEL EVALUATION RESULTS
# These are the exact results supplied from the project notebook.
# ------------------------------------------------------------

MODEL_METRICS = pd.DataFrame(
    [
        {
            "Model": "Logistic Regression",
            "Accuracy": 0.8384,
            "Precision": 0.7193,
            "Recall": 0.7059,
            "F1 Score": 0.7126,
            "ROC-AUC": 0.9059,
        },
        {
            "Model": "SVM",
            "Accuracy": 0.8414,
            "Precision": 0.7448,
            "Recall": 0.6711,
            "F1 Score": 0.7060,
            "ROC-AUC": 0.9030,
        },
        {
            "Model": "KNN",
            "Accuracy": 0.8187,
            "Precision": 0.6901,
            "Recall": 0.6551,
            "F1 Score": 0.6722,
            "ROC-AUC": 0.8561,
        },
    ]
).set_index("Model")

# Confusion matrix format:
# Rows = Actual class, Columns = Predicted class
# Class 0 = Stayed, Class 1 = Churned
CONFUSION_MATRICES = {
    "Logistic Regression": [[841, 103], [110, 264]],
    "SVM": [[858, 86], [123, 251]],
    "KNN": [[834, 110], [129, 245]],
}


# ------------------------------------------------------------
# 3. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
)


# ------------------------------------------------------------
# 4. CUSTOM CSS
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #8B949E;
        font-size: 15px;
        margin-bottom: 25px;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.10);
        border: 1px solid rgba(128, 128, 128, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 5. LOAD APPLICATION CONFIGURATION
# ------------------------------------------------------------

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ------------------------------------------------------------
# 6. LOAD TRAINED MODELS
# ------------------------------------------------------------

@st.cache_resource
def load_models():
    loaded_models = {}

    for model_name, model_path in MODEL_PATHS.items():
        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


# ------------------------------------------------------------
# 7. CHECK REQUIRED FILES AND LOAD RESOURCES
# ------------------------------------------------------------

required_files = [CONFIG_PATH, *MODEL_PATHS.values()]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error("Required project files are missing.")
    for file in missing_files:
        st.write(file)
    st.stop()

try:
    config = load_config()
    models = load_models()
except Exception as error:
    st.error(f"Unable to load application files: {error}")
    st.stop()


# ------------------------------------------------------------
# 8. GET DATASET OPTIONS
# ------------------------------------------------------------

categorical_options = config["categorical_options"]
numerical_defaults = config["numerical_defaults"]
numerical_min = config["numerical_min"]
numerical_max = config["numerical_max"]


# ------------------------------------------------------------
# 9. HELPER FUNCTIONS
# ------------------------------------------------------------

def get_options(column):
    return categorical_options[column]


def get_number_input(column, label, integer=False):
    minimum = numerical_min[column]
    maximum = numerical_max[column]
    default = numerical_defaults[column]

    if integer:
        minimum = int(minimum)
        maximum = int(maximum)
        default = int(round(default))

        return st.number_input(
            label,
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=1,
        )

    return st.number_input(
        label,
        min_value=float(minimum),
        max_value=float(maximum),
        value=float(default),
        step=0.5,
        format="%.2f",
    )


def show_confusion_matrix(model_name):
    """Display the saved test-set confusion matrix for a model."""
    matrix = CONFUSION_MATRICES[model_name]

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Stayed", "Actual Churned"],
        columns=["Predicted Stayed", "Predicted Churned"],
    )

    st.dataframe(matrix_df, use_container_width=True)


# ------------------------------------------------------------
# 10. APPLICATION HEADER
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">Telecom Customer Churn Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Machine Learning Based Customer Churn Prediction '
    'and Retention Recommendation System'
    '</div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 11. SIDEBAR NAVIGATION
# ------------------------------------------------------------

with st.sidebar:
    st.title("Navigation")

    page = st.radio(
        "Select Page",
        [
            "Customer Prediction",
            "Model Comparison Dashboard",
            "Customer Churn Risk Analysis",
            "About Project",
        ],
    )

    st.divider()
    st.subheader("Available ML Models")
    st.write("1. Logistic Regression")
    st.write("2. Support Vector Machine")
    st.write("3. K-Nearest Neighbors")
    st.divider()
    st.caption("Telecom Customer Churn ML Project")


# ============================================================
# PAGE 1: MODEL COMPARISON DASHBOARD
# ============================================================

if page == "Model Comparison Dashboard":

    st.title("Model Comparison Dashboard")
    st.write(
        "Compare the performance of Logistic Regression, SVM, and KNN "
        "using the evaluation results from the held-out test dataset."
    )

    st.info(
        "All metric values shown below are from your model evaluation "
        "results. Higher values are generally better for these metrics."
    )

    # --------------------------------------------------------
    # A. PERFORMANCE HIGHLIGHTS
    # --------------------------------------------------------

    st.subheader("Performance Highlights")

    best_accuracy_model = MODEL_METRICS["Accuracy"].idxmax()
    best_precision_model = MODEL_METRICS["Precision"].idxmax()
    best_recall_model = MODEL_METRICS["Recall"].idxmax()
    best_f1_model = MODEL_METRICS["F1 Score"].idxmax()
    best_auc_model = MODEL_METRICS["ROC-AUC"].idxmax()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Highest Accuracy",
            f"{MODEL_METRICS.loc[best_accuracy_model, 'Accuracy'] * 100:.2f}%",
            best_accuracy_model,
        )

    with col2:
        st.metric(
            "Highest Recall",
            f"{MODEL_METRICS.loc[best_recall_model, 'Recall'] * 100:.2f}%",
            best_recall_model,
        )

    with col3:
        st.metric(
            "Highest F1 Score",
            f"{MODEL_METRICS.loc[best_f1_model, 'F1 Score'] * 100:.2f}%",
            best_f1_model,
        )

    col4, col5 = st.columns(2)

    with col4:
        st.metric(
            "Highest Precision",
            f"{MODEL_METRICS.loc[best_precision_model, 'Precision'] * 100:.2f}%",
            best_precision_model,
        )

    with col5:
        st.metric(
            "Highest ROC-AUC",
            f"{MODEL_METRICS.loc[best_auc_model, 'ROC-AUC'] * 100:.2f}%",
            best_auc_model,
        )

    st.divider()

    # --------------------------------------------------------
    # B. COMPLETE MODEL COMPARISON TABLE
    # --------------------------------------------------------

    st.subheader("Complete Model Performance")

    formatted_metrics = MODEL_METRICS.copy()

    for column in formatted_metrics.columns:
        formatted_metrics[column] = formatted_metrics[column].map(
            lambda value: f"{value * 100:.2f}%"
        )

    st.dataframe(
        formatted_metrics,
        use_container_width=True,
    )

    st.caption(
        "Accuracy = overall correct predictions. Precision = proportion "
        "of predicted churners that were actually churners. Recall = "
        "proportion of actual churners detected. F1 Score balances "
        "Precision and Recall. ROC-AUC measures ranking/discrimination "
        "across classification thresholds."
    )

    st.divider()

    # --------------------------------------------------------
    # C. METRIC COMPARISON CHART
    # --------------------------------------------------------

    st.subheader("Metric Comparison Chart")

    selected_metrics = st.multiselect(
        "Select metrics to compare",
        options=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
        ],
        default=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
        ],
    )

    if selected_metrics:
        chart_data = MODEL_METRICS[selected_metrics] * 100
        st.bar_chart(chart_data, use_container_width=True)
        st.caption("Metric values are displayed as percentages.")
    else:
        st.warning("Select at least one metric to display the chart.")

    st.divider()

    # --------------------------------------------------------
    # D. CONFUSION MATRIX COMPARISON
    # --------------------------------------------------------

    st.subheader("Confusion Matrix Analysis")

    selected_matrix_model = st.selectbox(
        "Select model for confusion matrix",
        list(CONFUSION_MATRICES.keys()),
    )

    show_confusion_matrix(selected_matrix_model)

    selected_matrix = CONFUSION_MATRICES[selected_matrix_model]

    true_stayed = selected_matrix[0][0]
    false_churn = selected_matrix[0][1]
    missed_churn = selected_matrix[1][0]
    detected_churn = selected_matrix[1][1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Correctly Stayed", true_stayed)
    c2.metric("False Churn Alerts", false_churn)
    c3.metric("Missed Churners", missed_churn)
    c4.metric("Correctly Detected Churn", detected_churn)

    st.caption(
        "Rows represent actual customer status. Columns represent "
        "the model's predicted status. A missed churner is an actual "
        "churned customer predicted as stayed."
    )

    st.divider()

    # --------------------------------------------------------
    # E. MODEL SELECTION RECOMMENDATION
    # --------------------------------------------------------

    st.subheader("Model Selection Analysis")

    st.markdown(
        """
        **Recommended primary model for customer-retention focus: Logistic Regression**

        Logistic Regression has the highest Recall (70.59%), F1 Score
        (71.26%), and ROC-AUC (90.59%) among the three evaluated models.
        Its higher Recall means it detected a larger share of actual
        churners in this test set.

        **When SVM may be preferred**

        SVM has the highest Accuracy (84.14%) and Precision (74.48%).
        Its higher Precision means a larger share of customers it labels
        as churners were actually churners in this test set.

        **KNN**

        KNN has the lowest reported scores among the three models in this
        evaluation.
        """
    )

    st.warning(
        "This recommendation is based on the supplied test-set metrics. "
        "The best model for deployment should also be selected using "
        "business costs, validation results, threshold selection, and "
        "probability calibration where needed."
    )

    st.subheader("Confusion Matrices - All Models")

    for model_name in CONFUSION_MATRICES:
        with st.expander(f"{model_name}"):
            show_confusion_matrix(model_name)

    st.stop()



# ============================================================
# PAGE 2: CUSTOMER CHURN RISK FACTOR ANALYSIS
# ============================================================

if page == "Customer Churn Risk Analysis":

    st.title("Customer Churn Risk Factor Analysis")
    st.write(
        "Explore how churn rates differ across customer and service "
        "groups in the original telecom dataset."
    )

    DATA_PATH = BASE_DIR / "data" / "telecom_customer_churn.csv"

    if not DATA_PATH.exists():
        st.error("The original dataset was not found.")
        st.write(f"Expected file: {DATA_PATH}")
        st.info(
            "Place telecom_customer_churn.csv inside the project's "
            "data folder, then refresh this page."
        )
        st.stop()

    @st.cache_data
    def load_analysis_data(path_string):
        data = pd.read_csv(path_string)

        required_columns = [
            "Customer Status",
            "Age",
            "Married",
            "Number of Dependents",
            "Number of Referrals",
            "Tenure in Months",
            "Internet Service",
            "Internet Type",
            "Online Security",
            "Online Backup",
            "Device Protection Plan",
            "Premium Tech Support",
            "Contract",
            "Paperless Billing",
            "Payment Method",
            "Monthly Charge",
        ]

        missing_columns = [
            column for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing_columns)
            )

        return data

    try:
        raw_analysis_data = load_analysis_data(str(DATA_PATH))
    except Exception as error:
        st.error(f"Unable to load or validate the dataset: {error}")
        st.stop()

    # Keep only the two classes used for model training.
    analysis_data = raw_analysis_data[
        raw_analysis_data["Customer Status"].isin(["Stayed", "Churned"])
    ].copy()

    if analysis_data.empty:
        st.error(
            "No rows with Customer Status 'Stayed' or 'Churned' "
            "were found in the dataset."
        )
        st.stop()

    # Recreate the preprocessing treatment for structural missingness:
    # internet add-on fields are not applicable when Internet Service = No.
    internet_dependent_columns = [
        "Internet Type",
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
    ]

    no_internet_mask = analysis_data["Internet Service"].eq("No")

    for column in internet_dependent_columns:
        analysis_data.loc[no_internet_mask, column] = (
            analysis_data.loc[no_internet_mask, column]
            .fillna("No Internet Service")
        )

    analysis_data["Customer Status"] = (
        analysis_data["Customer Status"].astype(str).str.strip()
    )
    analysis_data["Churn Flag"] = (
        analysis_data["Customer Status"].eq("Churned").astype(int)
    )

    total_customers = len(analysis_data)
    churned_customers = int(analysis_data["Churn Flag"].sum())
    stayed_customers = total_customers - churned_customers
    overall_churn_rate = churned_customers / total_customers

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers Analyzed", f"{total_customers:,}")
    k2.metric("Churned Customers", f"{churned_customers:,}")
    k3.metric("Stayed Customers", f"{stayed_customers:,}")
    k4.metric("Overall Churn Rate", f"{overall_churn_rate * 100:.2f}%")

    st.caption(
        "Joined customers are excluded, matching the binary modeling "
        "dataset. Rates below are descriptive historical group rates."
    )

    st.divider()
    st.subheader("Explore Churn by Customer Factor")

    factor_options = [
        "Tenure in Months",
        "Contract",
        "Internet Type",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Married",
        "Paperless Billing",
        "Payment Method",
        "Number of Referrals",
        "Number of Dependents",
        "Age",
        "Monthly Charge",
    ]

    selected_factor = st.selectbox(
        "Select a factor",
        factor_options,
        index=0,
    )

    numeric_factors = [
        "Age",
        "Number of Dependents",
        "Number of Referrals",
        "Tenure in Months",
        "Monthly Charge",
    ]

    factor_data = analysis_data[
        [selected_factor, "Churn Flag"]
    ].copy()

    if selected_factor in numeric_factors:
        factor_data[selected_factor] = pd.to_numeric(
            factor_data[selected_factor],
            errors="coerce",
        )
        factor_data = factor_data.dropna(subset=[selected_factor])

        if factor_data.empty:
            st.warning("No valid numeric values are available for this factor.")
            st.stop()

        try:
            factor_data["Factor Group"] = pd.qcut(
                factor_data[selected_factor],
                q=5,
                duplicates="drop",
            ).astype(str)
        except ValueError:
            factor_data["Factor Group"] = (
                factor_data[selected_factor].astype(str)
            )

        group_column = "Factor Group"

    else:
        factor_data["Factor Group"] = (
            factor_data[selected_factor]
            .fillna("Missing / Not Available")
            .astype(str)
        )
        group_column = "Factor Group"

    summary = (
        factor_data.groupby(group_column, dropna=False)
        .agg(
            Customers=("Churn Flag", "size"),
            Churned=("Churn Flag", "sum"),
            Churn_Rate=("Churn Flag", "mean"),
        )
        .reset_index()
        .rename(columns={group_column: "Factor Group"})
    )

    summary["Stayed"] = summary["Customers"] - summary["Churned"]
    summary["Churn Rate"] = summary["Churn_Rate"] * 100
    summary = summary.sort_values(
        "Churn_Rate",
        ascending=False,
    )

    minimum_group_size = st.slider(
        "Minimum customers required in a group",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
    )

    visible_summary = summary[
        summary["Customers"] >= minimum_group_size
    ].copy()

    if visible_summary.empty:
        st.warning(
            "No groups meet this minimum size. Reduce the slider value."
        )
    else:
        display_table = visible_summary[
            ["Factor Group", "Customers", "Churned", "Stayed", "Churn Rate"]
        ].copy()
        display_table["Churn Rate"] = display_table["Churn Rate"].map(
            lambda value: f"{value:.2f}%"
        )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Churn Rate by Selected Factor")
        chart_data = visible_summary.set_index("Factor Group")[
            ["Churn Rate"]
        ]
        st.bar_chart(chart_data, use_container_width=True)

        highest_group = visible_summary.iloc[0]
        lowest_group = visible_summary.iloc[-1]

        r1, r2 = st.columns(2)
        r1.metric(
            "Highest Churn Group",
            str(highest_group["Factor Group"]),
            f'{highest_group["Churn Rate"]:.2f}% churn rate',
        )
        r2.metric(
            "Lowest Churn Group",
            str(lowest_group["Factor Group"]),
            f'{lowest_group["Churn Rate"]:.2f}% churn rate',
        )

    st.divider()
    st.subheader("How to Interpret This Analysis")
    st.markdown(
        """
        - **Customers** is the number of customers in that group.
        - **Churned** is the number whose recorded status is Churned.
        - **Stayed** is the number whose recorded status is Stayed.
        - **Churn Rate** is `Churned / Customers × 100`.
        - Numeric factors are grouped into quantile-based ranges so that
          their churn rates can be compared more easily.
        """
    )

    st.warning(
        "Important: These are historical associations in this dataset, "
        "not proof that a factor causes churn. This page does not calculate "
        "model feature importance, does not change the trained models, and "
        "does not guarantee an individual customer's outcome. Small groups "
        "can have unstable rates; use the minimum-group-size filter."
    )

    with st.expander("View analysis dataset summary"):
        st.write(
            f"Original dataset rows: {len(raw_analysis_data):,}"
        )
        st.write(
            f"Rows used here (Stayed + Churned): {total_customers:,}"
        )
        st.write(
            f"Joined rows excluded: "
            f"{int(raw_analysis_data['Customer Status'].eq('Joined').sum()):,}"
        )
        st.write(
            "Dataset path: data/telecom_customer_churn.csv"
        )

    st.stop()


# ============================================================
# PAGE 3: ABOUT PROJECT
# ============================================================

if page == "About Project":

    st.title("About This Project")

    st.write(
        """
        This project predicts whether an existing telecom customer
        is likely to stay or churn using machine learning.
        """
    )

    st.subheader("Machine Learning Algorithms")
    st.write("- Logistic Regression")
    st.write("- Support Vector Machine (SVM)")
    st.write("- K-Nearest Neighbors (KNN)")

    st.subheader("Technology Stack")
    st.write(
        """
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Streamlit
        - Joblib
        """
    )

    st.subheader("Prediction Classes")
    st.write("0 = Stayed")
    st.write("1 = Churned")

    st.subheader("Model Evaluation")
    st.dataframe(
        (MODEL_METRICS * 100).round(2).astype(str) + "%",
        use_container_width=True,
    )

    st.info(
        "This application provides model-based predictions and general "
        "retention suggestions. It does not guarantee that a customer "
        "will churn or that a suggested action will prevent churn."
    )

    st.stop()


# ============================================================
# PAGE 4: CUSTOMER PREDICTION
# ============================================================

st.title("Customer Prediction")
st.write("Enter the customer's information to predict churn.")


# ------------------------------------------------------------
# 12. SELECT MACHINE LEARNING MODEL
# ------------------------------------------------------------

selected_model_name = st.selectbox(
    "Select Machine Learning Model",
    list(MODEL_PATHS.keys()),
)

internet_service = st.selectbox(
    "Internet Service",
    get_options("Internet Service"),
    key="customer_internet_service",
)

st.divider()


# ------------------------------------------------------------
# 13. CUSTOMER INPUT FORM
# ------------------------------------------------------------

with st.form("customer_prediction_form"):

    st.subheader("Section A - Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        age = get_number_input(
            "Age",
            "Customer Age",
            integer=True,
        )

        married = st.selectbox(
            "Married",
            get_options("Married"),
        )

    with col2:
        dependents = get_number_input(
            "Number of Dependents",
            "Number of Dependents",
            integer=True,
        )

        referrals = get_number_input(
            "Number of Referrals",
            "Number of Referrals",
            integer=True,
        )

    st.divider()
    st.subheader("Section B - Subscription Information")

    col1, col2 = st.columns(2)

    with col1:
        tenure = get_number_input(
            "Tenure in Months",
            "Tenure in Months",
            integer=True,
        )

    with col2:
        # Internet-related categories depend on Internet Service.
        if internet_service == "No":
            internet_type = "No Internet Service"

            st.text_input(
                "Internet Type",
                value=internet_type,
                disabled=True,
            )
        else:
            internet_type_options = [
                value
                for value in get_options("Internet Type")
                if value != "No Internet Service"
            ]

            internet_type = st.selectbox(
                "Internet Type",
                internet_type_options,
            )

    # Internet service options
    if internet_service == "No":

        online_security = "No Internet Service"
        online_backup = "No Internet Service"
        device_protection = "No Internet Service"
        tech_support = "No Internet Service"

        st.info(
            "Internet service is disabled. Internet-related services "
            "are automatically set to No Internet Service."
        )

    else:
        col1, col2 = st.columns(2)

        with col1:
            online_security = st.selectbox(
                "Online Security",
                [
                    value
                    for value in get_options("Online Security")
                    if value != "No Internet Service"
                ],
            )

            online_backup = st.selectbox(
                "Online Backup",
                [
                    value
                    for value in get_options("Online Backup")
                    if value != "No Internet Service"
                ],
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection Plan",
                [
                    value
                    for value in get_options("Device Protection Plan")
                    if value != "No Internet Service"
                ],
            )

            tech_support = st.selectbox(
                "Premium Tech Support",
                [
                    value
                    for value in get_options("Premium Tech Support")
                    if value != "No Internet Service"
                ],
            )

    st.divider()
    st.subheader("Section C - Contract and Billing")

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox(
            "Contract",
            get_options("Contract"),
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            get_options("Paperless Billing"),
        )

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            get_options("Payment Method"),
        )

        monthly_charge = get_number_input(
            "Monthly Charge",
            "Monthly Charge",
        )

    st.divider()

    submit_button = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True,
    )


# ------------------------------------------------------------
# 14. PREPARE CUSTOMER DATA
# ------------------------------------------------------------

if submit_button:

    customer_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Married": married,
                "Number of Dependents": dependents,
                "Number of Referrals": referrals,
                "Tenure in Months": tenure,
                "Internet Service": internet_service,
                "Internet Type": internet_type,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection Plan": device_protection,
                "Premium Tech Support": tech_support,
                "Contract": contract,
                "Paperless Billing": paperless_billing,
                "Payment Method": payment_method,
                "Monthly Charge": monthly_charge,
            }
        ]
    )

    # --------------------------------------------------------
    # 15. PREDICT CUSTOMER CHURN
    # --------------------------------------------------------

    model = models[selected_model_name]
    prediction = int(model.predict(customer_data)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        churn_index = classes.index(1)

        probability = float(
            model.predict_proba(customer_data)[0][churn_index]
        )

    # --------------------------------------------------------
    # 16. DISPLAY PREDICTION RESULT
    # --------------------------------------------------------

    st.divider()
    st.title("Prediction Result")
    st.write(f"Selected Model: **{selected_model_name}**")

    if prediction == 1:
        st.error("Prediction: Customer is likely to CHURN")
    else:
        st.success("Prediction: Customer is likely to STAY")

    if probability is not None:
        st.subheader("Model-estimated Churn Score")

        st.metric(
            "Churn Score",
            f"{probability * 100:.2f}%",
        )

        st.progress(probability)

        st.caption(
            "This is the model's estimated probability score, "
            "not necessarily a calibrated real-world probability."
        )

    # --------------------------------------------------------
    # 17. RETENTION RECOMMENDATIONS
    # --------------------------------------------------------

    st.divider()
    st.subheader("Retention Recommendation")

    if prediction == 1:

        recommendations = []

        if contract == "Month-to-Month":
            recommendations.append(
                "Review whether a longer-term contract would provide "
                "better value."
            )

        if tenure <= 12:
            recommendations.append(
                "Contact the customer and check whether they need "
                "additional onboarding support."
            )

        if tech_support == "No":
            recommendations.append(
                "Explain available technical support options."
            )

        if online_security == "No" and internet_service == "Yes":
            recommendations.append(
                "Explain the available online security options."
            )

        if not recommendations:
            recommendations.append(
                "Contact the customer to understand their service "
                "experience and concerns."
            )

        st.warning(
            "This customer was classified as likely to churn. "
            "Consider the following retention actions:"
        )

        for recommendation in recommendations:
            st.write(f"- {recommendation}")

    else:
        st.success(
            "The model classified this customer as likely to stay."
        )

        st.write("- Continue providing good customer service.")
        st.write("- Monitor customer satisfaction.")
        st.write("- Maintain regular customer engagement.")

    # --------------------------------------------------------
    # 18. CUSTOMER INPUT SUMMARY
    # --------------------------------------------------------

    with st.expander("View Customer Information"):
        st.dataframe(
            customer_data,
            use_container_width=True,
        )

    st.caption(
        "Predictions are decision-support estimates, not guarantees. "
        "Retention recommendations are rule-based suggestions, "
        "not proven churn-prevention interventions."
    )
