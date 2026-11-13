"""Compare Logistic Regression, SVM, and KNN for one customer."""

import pandas as pd
import streamlit as st


MODEL_LABELS = {
    "Logistic Regression": "Logistic Regression",
    "Support Vector Machine (SVM)": "SVM",
    "K-Nearest Neighbors (KNN)": "KNN",
}

FEATURES = [
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


def _number_input(column, label, numerical_defaults, numerical_min, numerical_max, integer=False):
    minimum = numerical_min[column]
    maximum = numerical_max[column]
    default = numerical_defaults[column]

    if integer:
        return st.number_input(
            label,
            min_value=int(minimum),
            max_value=int(maximum),
            value=int(round(default)),
            step=1,
            key=f"consensus_{column}",
        )

    return st.number_input(
        label,
        min_value=float(minimum),
        max_value=float(maximum),
        value=float(default),
        step=0.5,
        format="%.2f",
        key=f"consensus_{column}",
    )


def _get_churn_probability(model, customer_data):
    """Return probability for class 1 (Churned), when supported."""
    if not hasattr(model, "predict_proba"):
        return None

    classes = list(model.classes_)
    if 1 not in classes:
        return 0.0

    churn_index = classes.index(1)
    return float(model.predict_proba(customer_data)[0][churn_index])


def render_page(models, categorical_options, numerical_defaults, numerical_min, numerical_max):
    """Render the independent three-model comparison page."""
    st.title("Three-Model Customer Prediction")
    st.write(
        "Enter a customer's information once to compare the predictions "
        "from Logistic Regression, SVM, and KNN."
    )
    st.info(
        "All three saved models evaluate the same customer inputs. "
        "Probability scores are model-estimated scores and may not be "
        "calibrated or directly comparable between different algorithms."
    )

    def options(column):
        return categorical_options[column]

    with st.form("three_model_comparison_form"):
        st.subheader("A. Customer Information")
        col1, col2 = st.columns(2)
        with col1:
            age = _number_input("Age", "Customer Age", numerical_defaults, numerical_min, numerical_max, True)
            married = st.selectbox("Married", options("Married"), key="consensus_married")
            dependents = _number_input("Number of Dependents", "Number of Dependents", numerical_defaults, numerical_min, numerical_max, True)
            referrals = _number_input("Number of Referrals", "Number of Referrals", numerical_defaults, numerical_min, numerical_max, True)
        with col2:
            tenure = _number_input("Tenure in Months", "Tenure in Months", numerical_defaults, numerical_min, numerical_max, True)
            monthly_charge = _number_input("Monthly Charge", "Monthly Charge", numerical_defaults, numerical_min, numerical_max)

        st.divider()
        st.subheader("B. Internet and Services")
        internet_service = st.selectbox(
            "Internet Service", options("Internet Service"), key="consensus_internet_service"
        )

        if internet_service == "No":
            internet_type = "No Internet Service"
            online_security = "No Internet Service"
            online_backup = "No Internet Service"
            device_protection = "No Internet Service"
            tech_support = "No Internet Service"
            st.info("Internet Service is No. Internet-dependent fields are set automatically to 'No Internet Service'.")
        else:
            internet_type_values = [v for v in options("Internet Type") if v != "No Internet Service"]
            security_values = [v for v in options("Online Security") if v != "No Internet Service"]
            backup_values = [v for v in options("Online Backup") if v != "No Internet Service"]
            protection_values = [v for v in options("Device Protection Plan") if v != "No Internet Service"]
            support_values = [v for v in options("Premium Tech Support") if v != "No Internet Service"]

            c1, c2 = st.columns(2)
            with c1:
                internet_type = st.selectbox("Internet Type", internet_type_values, key="consensus_internet_type")
                online_security = st.selectbox("Online Security", security_values, key="consensus_online_security")
                online_backup = st.selectbox("Online Backup", backup_values, key="consensus_online_backup")
            with c2:
                device_protection = st.selectbox("Device Protection Plan", protection_values, key="consensus_device_protection")
                tech_support = st.selectbox("Premium Tech Support", support_values, key="consensus_tech_support")

        st.divider()
        st.subheader("C. Contract and Billing")
        c1, c2 = st.columns(2)
        with c1:
            contract = st.selectbox("Contract", options("Contract"), key="consensus_contract")
            paperless_billing = st.selectbox("Paperless Billing", options("Paperless Billing"), key="consensus_paperless")
        with c2:
            payment_method = st.selectbox("Payment Method", options("Payment Method"), key="consensus_payment")

        submitted = st.form_submit_button("Compare All Three Models", use_container_width=True)

    if not submitted:
        st.caption("Complete the form and select Compare All Three Models to view the results.")
        return

    customer_data = pd.DataFrame([{
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
    }], columns=FEATURES)

    results = []
    try:
        for model_key, model in models.items():
            prediction = int(model.predict(customer_data)[0])
            probability = _get_churn_probability(model, customer_data)
            results.append({
                "Model": MODEL_LABELS.get(model_key, model_key),
                "Prediction": "Churned" if prediction == 1 else "Stayed",
                "Churn Score": probability,
                "Class": prediction,
            })
    except Exception as error:
        st.error(f"Unable to compare the models: {error}")
        st.exception(error)
        return

    results_df = pd.DataFrame(results)
    churn_votes = int(results_df["Class"].sum())
    stay_votes = len(results_df) - churn_votes

    st.divider()
    st.subheader("Side-by-Side Results")
    result_columns = st.columns(len(results_df))

    for column, (_, result) in zip(result_columns, results_df.iterrows()):
        with column:
            st.markdown(f"**{result['Model']}**")
            if result["Class"] == 1:
                st.error(f"Prediction: {result['Prediction']}")
            else:
                st.success(f"Prediction: {result['Prediction']}")
            if pd.notna(result["Churn Score"]):
                st.metric("Estimated Churn Score", f"{result['Churn Score'] * 100:.2f}%")
            else:
                st.metric("Estimated Churn Score", "Not available")

    st.subheader("Prediction Comparison Table")
    display_results = results_df[["Model", "Prediction", "Churn Score"]].copy()
    display_results["Churn Score"] = display_results["Churn Score"].apply(
        lambda value: f"{value * 100:.2f}%" if pd.notna(value) else "Not available"
    )
    st.dataframe(display_results, use_container_width=True, hide_index=True)

    st.subheader("Three-Model Voting Summary")
    vote1, vote2, vote3 = st.columns(3)
    vote1.metric("Churn Predictions", f"{churn_votes} / 3")
    vote2.metric("Stay Predictions", f"{stay_votes} / 3")

    if churn_votes > stay_votes:
        vote3.metric("Majority Result", "CHURNED")
        st.warning("The majority of the three models predict that this customer may churn. Review the customer context before taking retention action.")
    elif stay_votes > churn_votes:
        vote3.metric("Majority Result", "STAYED")
        st.success("The majority of the three models predict that this customer may stay.")
    else:
        vote3.metric("Majority Result", "TIE")
        st.warning("The models disagree. Consider reviewing the customer details and model performance before making a decision.")

    st.caption(
        "The majority result is a simple vote across the three model classes. "
        "It is not a separately trained ensemble model. A prediction is not a guarantee of future customer behavior."
    )

    with st.expander("View the exact customer data sent to the models"):
        st.dataframe(customer_data, use_container_width=True, hide_index=True)
