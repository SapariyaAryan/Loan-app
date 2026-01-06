import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Initialize Google Sheets connection
@st.cache_resource
def get_gsheet_connection():
    """Connect to Google Sheets using service account credentials"""
    try:
        # Get credentials from Streamlit secrets
        gcp_service_account = st.secrets["gcp_service_account"]
        
        # Define the scope
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Authenticate with Google
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            gcp_service_account, scopes=scope
        )
        
        # Connect to Google Sheets
        gc = gspread.authorize(credentials)
        
        # Open your Google Sheet by ID
        SHEET_ID = "1le0a4eqwZtXfT_3PUHmv0dokAGu0mh7Wx1Vp35T9rwU"
        sh = gc.open_by_key(SHEET_ID)
        
        # Access the first worksheet (default)
        return sh.sheet1
    
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

def save_to_google_sheets(data):
    """Save customer data to Google Sheets"""
    try:
        worksheet = get_gsheet_connection()
        
        if worksheet is None:
            st.error("Could not connect to Google Sheets")
            return False
        
        # Add timestamp
        data_with_timestamp = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ] + data
        
        # Append to sheet
        worksheet.append_row(data_with_timestamp)
        
        return True
    
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}")
        return False


st.set_page_config(page_title="Deutsche Kreditbank - Loan Management", layout="wide")
st.title("🏦 Deutsche Kreditbank - Loan Management System")
st.markdown("---")
st.header("💳 Loan Eligibility Checker")

def extract_text_from_pdf_ocr(uploaded_file):
    """Extract text from scanned PDF using OCR"""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        
        pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()
        images = convert_from_bytes(pdf_bytes, dpi=300)
        text = ""
        
        for image in images:
            ocr_text = pytesseract.image_to_string(image, lang="deu+eng", config="--psm 6")
            if ocr_text:
                text += ocr_text + " "
        
        return text.lower().strip() if text.strip() else None
    except Exception as e:
        return None

def extract_text_from_pdf_direct(uploaded_file):
    """Extract text directly from PDF"""
    try:
        import pdfplumber
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return text.lower().strip() if text.strip() else None
    except:
        return None

def extract_text_from_pdf(uploaded_file):
    """Try direct extraction first, then OCR"""
    text = extract_text_from_pdf_direct(uploaded_file)
    if text and len(text) > 10:
        return text, "direct"
    
    text = extract_text_from_pdf_ocr(uploaded_file)
    if text and len(text) > 10:
        return text, "ocr"
    
    return None, None

def is_valid_payslip(text):
    """🚨 FRAUD DETECTION - Only accept PAYSLIPS. Reject contracts & blocked accounts!"""
    if text is None or len(text) < 50:
        return False, 0, False
    
    text_lower = text.lower()
    
    # 🚫 REJECT BLOCKED ACCOUNTS (Sperrkonto) - FIRST CHECK!
    blocked_account_keywords = [
        'sperrkonto', 'blocked account', 'sperrfrist', 'einzahlungsnachweis',
        'blockade', 'international student', 'student visa', 'visum',
        'sperrkontonachweis', 'blocked account statement', 'sperrung'
    ]
    
    is_blocked_account = any(kw in text_lower for kw in blocked_account_keywords)
    if is_blocked_account:
        return False, 0, True
    
    # 🚫 REJECT EMPLOYMENT CONTRACTS - NOT PAYSLIPS!
    contract_keywords = [
        'arbeitsvertrag', 'employment contract', 'vertragsinhalt', 'vertragsbedingungen',
        'arbeitgeber', 'arbeitnehmer', 'beschäftigung', 'anstellung',
        'dienstverhältnis', 'probezeit', 'kündigung', 'kündigungsfrist',
        'arbeitsvertragliche', 'contractual terms', 'employment relationship',
        'termination', 'probationary period', 'contract period'
    ]
    
    is_contract = sum(1 for kw in contract_keywords if kw in text_lower) >= 2
    if is_contract:
        return False, 0, False
    
    # PAYSLIP KEYWORDS - must have at least 3
    payslip_keywords = [
        'gehalt', 'lohn', 'brutto', 'netto', 'abrechnung', 'verdienst',
        'auszahlung', 'einkommen', 'entgelt', 'salary', 'wage', 'payment',
        'arbeitsentgelt', 'vergütung', 'honorar', 'bezug', 
        'azb', 'gsn', 'bsl', 'brg', 'steuer', 'tax', 'kranken', 'sozial',
        'personalnummer', 'employee', 'monatlich', 'monthly'
    ]
    
    keyword_count = sum(1 for kw in payslip_keywords if kw in text_lower)
    
    # Must have at least 3 payslip keywords
    if keyword_count < 3:
        return False, keyword_count, False
    
    # REJECT obvious non-payslips
    suspicious = ['invoice', 'rechnung', 'quittung', 'receipt', 'bestellung', 'order']
    has_suspicious = any(s in text_lower for s in suspicious)
    
    if has_suspicious and keyword_count < 6:
        return False, keyword_count, False
    
    return True, keyword_count, False

def extract_salary_from_text(text):
    """SUPER AGGRESSIVE salary extraction - handles all formats"""
    if text is None or len(text) < 2:
        return None
    
    try:
        original_text = text
        text = re.sub(r'\s+', ' ', text.lower())
        candidates = []
        
        # PRIORITY 1: AZB (Auszahlungsbetrag) - most reliable
        azb_pattern = r'azb[^0-9]*?([\d.,]+)'
        azb_matches = re.findall(azb_pattern, text)
        for num_str in azb_matches:
            clean_num = num_str.replace(' ', '')
            if ',' in clean_num and '.' in clean_num:
                if clean_num.rfind(',') > clean_num.rfind('.'):
                    clean_num = clean_num.replace('.', '').replace(',', '.')
                else:
                    clean_num = clean_num.replace(',', '')
            elif ',' in clean_num:
                if len(clean_num.split(',')[-1]) <= 2:
                    clean_num = clean_num.replace(',', '.')
                else:
                    clean_num = clean_num.replace(',', '')
            elif '.' in clean_num and clean_num.count('.') > 1:
                clean_num = clean_num.replace('.', '')
            try:
                amount = float(clean_num)
                if 50 <= amount <= 25000:
                    candidates.append(amount)
            except:
                pass
        
        # PRIORITY 2: GSN (Gesetzliches Netto)
        if len(candidates) < 2:
            gsn_pattern = r'gsn[^0-9]*?([\d.,]+)'
            gsn_matches = re.findall(gsn_pattern, text)
            for num_str in gsn_matches:
                clean_num = num_str.replace(' ', '')
                if ',' in clean_num and '.' in clean_num:
                    if clean_num.rfind(',') > clean_num.rfind('.'):
                        clean_num = clean_num.replace('.', '').replace(',', '.')
                    else:
                        clean_num = clean_num.replace(',', '')
                elif ',' in clean_num:
                    if len(clean_num.split(',')[-1]) <= 2:
                        clean_num = clean_num.replace(',', '.')
                try:
                    amount = float(clean_num)
                    if 50 <= amount <= 25000:
                        candidates.append(amount)
                except:
                    pass
        
        # PRIORITY 3: Generic keyword-based
        if len(candidates) < 2:
            keywords = [
                r'nettobetrag', r'nettolohn', r'nettoentgelt', r'netto\s*zahlung',
                r'auszahlungsbetrag', r'auszahlung', r'ausgezahlt',
                r'verdienst', r'gehalt', r'lohn', r'brutto', r'einkommen',
                r'entgelt', r'salary', r'wage', r'net\s*pay', r'monthly',
                r'monatlich', r'netto'
            ]
            
            for keyword in keywords:
                for match in re.finditer(rf'({keyword})[^\d]*?([\d.,]+)', text):
                    num_str = match.group(2)
                    clean_num = num_str.replace(' ', '')
                    
                    if ',' in clean_num and '.' in clean_num:
                        if clean_num.rfind(',') > clean_num.rfind('.'):
                            clean_num = clean_num.replace('.', '').replace(',', '.')
                        else:
                            clean_num = clean_num.replace(',', '')
                    elif ',' in clean_num:
                        if len(clean_num.split(',')[-1]) <= 2:
                            clean_num = clean_num.replace(',', '.')
                        else:
                            clean_num = clean_num.replace(',', '')
                    elif '.' in clean_num and clean_num.count('.') > 1:
                        clean_num = clean_num.replace('.', '')
                    
                    try:
                        amount = float(clean_num)
                        if 50 <= amount <= 25000:
                            candidates.append(amount)
                    except:
                        pass
        
        # PRIORITY 4: € symbol
        if len(candidates) < 2:
            euro_pattern = r'€\s*([\d.,]+)'
            euro_matches = re.findall(euro_pattern, text)
            for num_str in euro_matches:
                clean_num = num_str.replace(' ', '')
                if ',' in clean_num and '.' in clean_num:
                    if clean_num.rfind(',') > clean_num.rfind('.'):
                        clean_num = clean_num.replace('.', '').replace(',', '.')
                    else:
                        clean_num = clean_num.replace(',', '')
                elif ',' in clean_num:
                    if len(clean_num.split(',')[-1]) <= 2:
                        clean_num = clean_num.replace(',', '.')
                try:
                    amount = float(clean_num)
                    if 50 <= amount <= 25000:
                        candidates.append(amount)
                except:
                    pass
        
        # PRIORITY 5: All large numbers in text (last resort)
        if len(candidates) < 2:
            all_nums = re.findall(r'[\d]{3,5}[.,]?[\d]{0,2}', text)
            for num_str in all_nums[-100:]:
                clean_num = num_str.replace(' ', '')
                if ',' in clean_num and '.' in clean_num:
                    if clean_num.rfind(',') > clean_num.rfind('.'):
                        clean_num = clean_num.replace('.', '').replace(',', '.')
                    else:
                        clean_num = clean_num.replace(',', '')
                elif ',' in clean_num:
                    if len(clean_num.split(',')[-1]) <= 2:
                        clean_num = clean_num.replace(',', '.')
                elif '.' in clean_num and clean_num.count('.') > 1:
                    clean_num = clean_num.replace('.', '')
                
                try:
                    amount = float(clean_num)
                    if 50 <= amount <= 25000:
                        candidates.append(amount)
                except:
                    pass
        
        if not candidates:
            return None
        
        # Return median (most stable for multiple salaries)
        return float(np.median(candidates))
    
    except:
        return None

def validate_and_extract_payslip(uploaded_file):
    """Validate and extract salary with FRAUD DETECTION"""
    try:
        filename = uploaded_file.name.lower()
        
        if not filename.endswith('.pdf'):
            return False, 0.0, "❌ Only PDF", "", "", 0
        
        extracted_text, method = extract_text_from_pdf(uploaded_file)
        
        if extracted_text is None or len(extracted_text) < 5:
            return False, 0.0, "❌ Cannot read", "", "", 0
        
        # 🚨 FRAUD CHECK - must be a real payslip by scanning PDF CONTENT
        is_valid, keyword_count, is_blocked = is_valid_payslip(extracted_text)
        
        # Check if it's a blocked account
        if is_blocked:
            return False, 0.0, "🚫 BLOCKED ACCOUNT! (Not a payslip)", extracted_text, method, 0
        
        if not is_valid:
            if 'arbeitsvertrag' in extracted_text.lower() or 'employment contract' in extracted_text.lower():
                return False, 0.0, "❌ EMPLOYMENT CONTRACT! (Not a payslip)", extracted_text, method, 0
            return False, 0.0, f"❌ NOT A PAYSLIP! (Only {keyword_count} keywords)", extracted_text, method, keyword_count
        
        extracted_salary = extract_salary_from_text(extracted_text)
        
        if extracted_salary is None or extracted_salary <= 0:
            return False, 0.0, "❌ No salary found", extracted_text, method, keyword_count
        
        if not (50 <= extracted_salary <= 25000):
            return False, 0.0, f"❌ Out of range: €{extracted_salary:.2f}", extracted_text, method, keyword_count
        
        return True, extracted_salary, f"✅ €{extracted_salary:.2f}", extracted_text, method, keyword_count
    
    except:
        return False, 0.0, "❌ Error", "", "", 0

# Train ML model
if not os.path.exists('loan_model.pkl'):
    st.info("⏳ Training ML model...")
    np.random.seed(42)
    data = {
        'Monthly_Income': np.random.uniform(500, 10000, 1000),
        'Credit_Score': np.random.randint(300, 900, 1000),
        'Age': np.random.randint(22, 65, 1000),
        'Employment_Years': np.random.randint(1, 30, 1000),
        'Existing_Loans': np.random.randint(0, 5, 1000),
        'Monthly_Expenses': np.random.uniform(200, 5000, 1000),
    }
    df = pd.DataFrame(data)
    # More flexible approval criteria
    df['Loan_Approved'] = (
        ((df['Monthly_Income'] > 800) & (df['Credit_Score'] > 600) & (df['Monthly_Expenses'] < df['Monthly_Income'] * 0.8)) |  # Standard criteria
        ((df['Credit_Score'] > 750) & (df['Monthly_Income'] > 500))  # High credit score override
    ).astype(int)
    df['Debt_to_Income_Ratio'] = df['Monthly_Expenses'] / df['Monthly_Income']
    df['Monthly_Savings'] = df['Monthly_Income'] - df['Monthly_Expenses']
    df['Income_per_Loan'] = df['Monthly_Income'] / (df['Existing_Loans'] + 1)
    X = df[['Monthly_Income', 'Credit_Score', 'Age', 'Employment_Years', 'Existing_Loans', 'Monthly_Expenses', 'Debt_to_Income_Ratio', 'Monthly_Savings', 'Income_per_Loan']]
    y = df['Loan_Approved']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    pickle.dump(model, open('loan_model.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    st.success("✅ Model trained!")

model = pickle.load(open('loan_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.subheader("👤 Your Details")
col1, col2 = st.columns(2)
with col1:
    customer_name = st.text_input("Name:")
    age = st.number_input("Age:", 18, 100, 30)
with col2:
    employment_years = st.number_input("Employment (yrs):", 0, 50, 5)
    existing_loans = st.number_input("Loans:", 0, 10, 0)

st.markdown("---")
st.subheader("📄 Upload 4 Payslips")
st.info("⚠️ Employment contracts & blocked accounts will be rejected!")

uploaded_files = st.file_uploader("Select 4 PDFs", type=["pdf"], accept_multiple_files=True, key="main")

all_valid, avg_salary, manual_salary = False, 0.0, None

if uploaded_files:
    if len(uploaded_files) != 4:
        st.error(f"{len(uploaded_files)}/4 files")
    else:
        extracted_salaries = []
        for i, file in enumerate(uploaded_files, 1):
            is_valid, salary, msg, _, _, _ = validate_and_extract_payslip(file)
            col1, col2, col3 = st.columns([0.3, 2, 1.5])
            with col1:
                st.write(f"{i}.")
            with col2:
                st.write(f"`{file.name}`")
            with col3:
                if is_valid:
                    st.success(msg)
                    extracted_salaries.append(salary)
                else:
                    st.error(msg)
        
        if len(extracted_salaries) == 4:
            all_valid = True
            avg_salary = np.mean(extracted_salaries)
            st.success("✅ All verified!")
            st.info(f"Average: €{avg_salary:,.2f}")
            
            manual_salary = st.number_input("Confirm (€):", avg_salary * 0.9, avg_salary * 1.1, avg_salary, 10.0)
            if abs(manual_salary - avg_salary) / avg_salary > 0.1:
                st.error("❌ >10% difference")
                all_valid = False

st.markdown("---")
col3, col4 = st.columns(2)
with col3:
    monthly_income = st.number_input("Income (€):", 100.0, 100000.0, float(manual_salary) if manual_salary else 2000.0, disabled=True)
    expenses = st.number_input("Expenses (€):", 100.0, 100000.0, 1000.0, 50.0)
with col4:
    credit_score = st.slider("Credit Score:", 300, 900, 650)

st.markdown("---")
st.subheader("💰 Loan Request")

# Calculate max loan available based on income
available_savings = monthly_income - expenses
max_monthly_payment = available_savings * 0.7  # Max 70% of available income
max_loan_amount = max_monthly_payment * 60  # Assuming 5-year repayment

st.info(f"📊 **Max Affordable Loan:** €{max_loan_amount:,.0f} (based on €{max_monthly_payment:,.0f}/month budget)")

requested_loan = st.number_input(
    "How much would you like to borrow? (€):", 
    min_value=100.0, 
    max_value=100000.0, 
    value=min(5000.0, max(100.0, max_loan_amount)),
    step=100.0
)

st.markdown("---")

if st.button("✅ CHECK LOAN", use_container_width=True, type="primary"):
    if not customer_name:
        st.error("Enter name")
    elif not all_valid:
        st.error("Verify 4 payslips")
    elif monthly_income <= 0 or expenses >= monthly_income:
        st.error("Invalid income/expenses")
    elif requested_loan <= 0:
        st.error("Enter loan amount")
    else:
        dti = expenses / monthly_income
        savings = monthly_income - expenses
        income_per_loan = monthly_income / (existing_loans + 1)
        
        input_data = np.array([[monthly_income, credit_score, age, employment_years, existing_loans, expenses, dti, savings, income_per_loan]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]
        
        # Dynamic loan limits based on credit score
        if credit_score >= 800:
            max_amt = monthly_income * 30
            rate = 6.5
            risk = "🟢 EXCELLENT"
        elif credit_score >= 750:
            max_amt = monthly_income * 24
            rate = 8.5
            risk = "🟢 LOW"
        elif credit_score >= 700:
            max_amt = monthly_income * 18
            rate = 9.5
            risk = "🟡 MEDIUM"
        elif credit_score >= 650:
            max_amt = monthly_income * 12
            rate = 11.5
            risk = "🟡 MEDIUM"
        elif credit_score >= 600:
            max_amt = monthly_income * 6
            rate = 13.5
            risk = "🟠 HIGH"
        else:
            max_amt = 0.0
            rate = 0.0
            risk = "🔴 VERY HIGH"
        
        # Calculate monthly payment for 5-year loan
        loan_term_months = 60
        monthly_rate = (rate / 100) / 12
        if monthly_rate > 0 and requested_loan > 0:
            monthly_payment = (monthly_rate * requested_loan) / (1 - (1 + monthly_rate) ** (-loan_term_months))
        else:
            monthly_payment = requested_loan / loan_term_months if requested_loan > 0 else 0
        
        # Check if user can afford the loan
        available_for_payment = savings * 0.7
        can_afford = monthly_payment <= available_for_payment
        requested_within_limit = requested_loan <= max_amt
        
        st.markdown("---")
        st.subheader("📊 DECISION")
        
        # Approval logic - based on affordability + credit score
        if credit_score < 600:
            st.error("❌ DENIED - Credit score too low (minimum 600)")
        elif not requested_within_limit:
            st.error(f"❌ DENIED - Requested loan (€{requested_loan:,.0f}) exceeds max approved amount (€{max_amt:,.0f})")
        elif not can_afford:
            st.error(f"❌ DENIED - Monthly payment (€{monthly_payment:,.0f}) exceeds affordable budget (€{available_for_payment:,.0f})")
        else:
            st.success("✅ APPROVED!")
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Credit Score", credit_score)
        with col2:
            st.metric("Risk Level", risk)
        with col3:
            st.metric("Confidence", f"{proba:.0%}")
        
        st.markdown("---")
        st.subheader("💳 Loan Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Requested", f"€{requested_loan:,.0f}")
        with col2:
            st.metric("Max Allowed", f"€{max_amt:,.0f}")
        with col3:
            st.metric("Interest Rate", f"{rate}%")
        with col4:
            st.metric("Term", "5 years")
        
        st.markdown("---")
        st.subheader("📈 Affordability Check")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Monthly Payment", f"€{monthly_payment:,.0f}")
        with col2:
            st.metric("Budget Available", f"€{available_for_payment:,.0f}")
        with col3:
            ratio = (monthly_payment / available_for_payment * 100) if available_for_payment > 0 else 0
            color = "🟢" if ratio <= 100 else "🔴"
            st.metric("Payment Ratio", f"{color} {ratio:.0f}%")
        with col4:
            st.metric("Monthly Income", f"€{monthly_income:,.0f}")

st.markdown("---")

st.markdown("<p style='text-align:center;font-size:11px;color:gray;'>🏦 Deutsche Kreditbank © 2025 | Smart Loan Approval System</p>", unsafe_allow_html=True)
