import gradio as gr
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Load the trained model (Ensure this is the model where we dropped grade_subgrade!)
try:
    model = joblib.load('xgboost_loan_model.pkl')
except FileNotFoundError:
    print("Error: Could not find 'xgboost_loan_model.pkl'. Please ensure it's in the same folder.")
    exit()

# 2. Define the core prediction function
def predict_loan(age, gender, marital_status, education_level, employment_status,
                 annual_income, credit_score, dti, open_accounts, total_credit_limit, 
                 current_balance, delinquency_history, public_records, num_delinquencies,
                 loan_amount, loan_purpose, interest_rate, loan_term):
    
    # Auto-calculate derived financial metrics
    monthly_income = annual_income / 12
    
    # Standard loan installment formula
    monthly_interest_rate = (interest_rate / 100) / 12
    if monthly_interest_rate > 0:
        installment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate)**loan_term) / ((1 + monthly_interest_rate)**loan_term - 1)
    else:
        installment = loan_amount / loan_term

    # Map inputs directly to the columns our model expects
    # NOTE: 'grade_subgrade' is completely removed, making this a pure AI decision!
    input_data = {
        'age': age,
        'gender': gender,
        'marital_status': marital_status,
        'education_level': education_level,
        'annual_income': annual_income,
        'monthly_income': monthly_income,
        'employment_status': employment_status,
        'debt_to_income_ratio': dti,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'loan_purpose': loan_purpose,
        'interest_rate': interest_rate,
        'loan_term': loan_term,
        'installment': installment,
        'num_of_open_accounts': open_accounts,
        'total_credit_limit': total_credit_limit,
        'current_balance': current_balance,
        'delinquency_history': delinquency_history,
        'public_records': public_records,
        'num_of_delinquencies': num_delinquencies
    }

    input_df = pd.DataFrame([input_data])

    # Make Prediction
    prediction = model.predict(input_df)[0]
    prob_default = model.predict_proba(input_df)[0][0] # Probability of Class 0 (Default)

    # Format the Output beautifully
    if prediction == 1: # Paid Back
        confidence = (1 - prob_default) * 100
        return f"""
        <div style="background-color:#d4edda; color:#155724; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="margin-top:0;">✅ LOAN APPROVED</h2>
            <p style="font-size:16px;">The AI model classifies this application as <b>Low Risk</b>.</p>
            <p style="font-size:18px;"><b>Confidence Score: {confidence:.1f}%</b> probability of successful repayment.</p>
        </div>
        """
    else: # Default
        confidence = prob_default * 100
        return f"""
        <div style="background-color:#f8d7da; color:#721c24; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="margin-top:0;">❌ LOAN REJECTED</h2>
            <p style="font-size:16px;">The AI model classifies this application as <b>High Risk of Default</b>.</p>
            <p style="font-size:18px;"><b>Risk Score: {confidence:.1f}%</b> probability of default.</p>
            <p style="font-size:14px;"><i>Recommendation: Manually review applicant's Debt-to-Income Ratio.</i></p>
        </div>
        """

# 3. Build the UI using Gradio Blocks
with gr.Blocks(theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# 🏦 Unbiased AI Underwriting Engine")
    gr.Markdown("Fill out the applicant details below. This AI makes pure decisions based on financial health metrics without relying on manual bank grades.")

    with gr.Tabs():
        # TAB 1: Demographics & Financials
        with gr.TabItem("👤 Applicant Profile"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Personal Information")
                    age = gr.Slider(minimum=18, maximum=100, value=35, step=1, label="Age")
                    gender = gr.Radio(choices=["Male", "Female"], value="Male", label="Gender")
                    marital_status = gr.Dropdown(choices=["Single", "Married", "Divorced"], value="Single", label="Marital Status")
                    education_level = gr.Dropdown(choices=["High School", "Bachelor's", "Master's", "PhD"], value="Bachelor's", label="Education Level")
                    employment_status = gr.Radio(choices=["Employed", "Unemployed", "Self-Employed"], value="Employed", label="Employment Status")
                
                with gr.Column():
                    gr.Markdown("### Financial Health")
                    annual_income = gr.Number(value=65000, label="Annual Income ($)")
                    credit_score = gr.Slider(minimum=300, maximum=850, value=710, step=1, label="Credit Score")
                    dti = gr.Slider(minimum=0.0, maximum=1.0, value=0.22, step=0.01, label="Debt-to-Income Ratio (DTI)")
                    total_credit_limit = gr.Number(value=50000, label="Total Credit Limit ($)")
                    current_balance = gr.Number(value=15000, label="Current Total Balance ($)")

        # TAB 2: Credit History & Loan Request
        with gr.TabItem("📊 Loan Request & History"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Credit History")
                    open_accounts = gr.Slider(minimum=0, maximum=50, value=6, step=1, label="Number of Open Accounts")
                    delinquency_history = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Years of Delinquency History")
                    public_records = gr.Slider(minimum=0, maximum=5, value=0, step=1, label="Derogatory Public Records (Bankruptcies)")
                    num_delinquencies = gr.Slider(minimum=0, maximum=20, value=0, step=1, label="Number of Past Delinquencies")

                with gr.Column():
                    gr.Markdown("### Current Loan Request")
                    loan_amount = gr.Number(value=20000, label="Requested Loan Amount ($)")
                    loan_purpose = gr.Dropdown(choices=["Debt consolidation", "Credit card", "Home", "Car", "Medical", "Business", "Other"], value="Debt consolidation", label="Loan Purpose")
                    interest_rate = gr.Slider(minimum=5.0, maximum=35.0, value=11.5, step=0.1, label="Interest Rate (%)")
                    loan_term = gr.Radio(choices=[36, 60], value=36, label="Loan Term (Months)")

    # Execute Button
    with gr.Row():
        submit_btn = gr.Button("🔍 Run AI Underwriter", variant="primary", scale=1)
    
    with gr.Row():
        output_html = gr.HTML(label="AI Decision")

    # Inputs list (NO grade_subgrade!)
    inputs_list = [
        age, gender, marital_status, education_level, employment_status,
        annual_income, credit_score, dti, open_accounts, total_credit_limit, 
        current_balance, delinquency_history, public_records, num_delinquencies,
        loan_amount, loan_purpose, interest_rate, loan_term
    ]
    
    submit_btn.click(fn=predict_loan, inputs=inputs_list, outputs=output_html)

if __name__ == "__main__":
    app.launch(share=True)