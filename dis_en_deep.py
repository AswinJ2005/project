import os
import streamlit as st
import pandas as pd
import bcrypt
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import re
import random
import time
import plotly.express as px
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import smtplib
from email.mime.text import MIMEText
import asyncio
import python_weather

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name=os.getenv("AWS_REGION"))
s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION"))
cognito = boto3.client('cognito-idp', region_name=os.getenv("AWS_REGION"))
kms = boto3.client('kms', region_name=os.getenv("AWS_REGION"))

# AWS Resources
USERS_TABLE = dynamodb.Table(os.getenv("USERS_TABLE"))
DISEASES_TABLE = dynamodb.Table(os.getenv("DISEASES_TABLE"))
MEDICATIONS_BUCKET = os.getenv("MEDICATIONS_BUCKET")
APPOINTMENTS_BUCKET = os.getenv("APPOINTMENTS_BUCKET")
MEDICAL_REPORTS_BUCKET = os.getenv("MEDICAL_REPORTS_BUCKET")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID")
KMS_KEY_ID = os.getenv("KMS_KEY_ID")

# Configure Streamlit page
st.set_page_config(
    page_title="Chronic Disease Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DRUG_INTERACTIONS = {
    'warfarin': ['aspirin', 'ibuprofen', 'omeprazole'],
    'metformin': ['contrast_dye', 'hydrochlorothiazide'],
}


# Security functions
def encrypt_data(data):
    try:
        response = kms.encrypt(KeyId=KMS_KEY_ID, Plaintext=data.encode())
        return response['CiphertextBlob']
    except ClientError as e:
        st.error(f"Encryption error: {e}")
        return None


def verify_password(hashed_password, input_password):
    try:
        decrypted = kms.decrypt(CiphertextBlob=hashed_password)['Plaintext']
        return decrypted.decode() == input_password
    except ClientError as e:
        st.error(f"Decryption error: {e}")
        return False


# Data management functions
def load_users():
    response = USERS_TABLE.scan()
    return pd.DataFrame(response['Items'])


def save_user(user_data):
    try:
        USERS_TABLE.put_item(Item=user_data)
    except ClientError as e:
        st.error(f"Error saving user: {e}")


def load_diseases():
    response = DISEASES_TABLE.scan()
    return pd.DataFrame(response['Items'])


def save_disease(disease_data):
    try:
        DISEASES_TABLE.put_item(Item=disease_data)
    except ClientError as e:
        st.error(f"Error saving disease: {e}")


def load_medications(username):
    try:
        obj = s3.get_object(Bucket=MEDICATIONS_BUCKET, Key=f"{username}/medications.csv")
        return pd.read_csv(obj['Body'])
    except ClientError:
        return pd.DataFrame(columns=['name', 'dosage', 'frequency', 'start_date', 'end_date'])


def save_medications(username, df):
    csv_buffer = df.to_csv(index=False)
    s3.put_object(Bucket=MEDICATIONS_BUCKET, Key=f"{username}/medications.csv", Body=csv_buffer)


def load_appointments(username):
    try:
        obj = s3.get_object(Bucket=APPOINTMENTS_BUCKET, Key=f"{username}/appointments.csv")
        return pd.read_csv(obj['Body'], parse_dates=['datetime'])
    except ClientError:
        return pd.DataFrame(columns=['doctor', 'datetime', 'reason', 'confirmed'])


def save_appointments(username, df):
    csv_buffer = df.to_csv(index=False)
    s3.put_object(Bucket=APPOINTMENTS_BUCKET, Key=f"{username}/appointments.csv", Body=csv_buffer)


def get_medical_reports(username):
    """List all medical reports for a user"""
    try:
        response = s3.list_objects_v2(
            Bucket=MEDICAL_REPORTS_BUCKET,
            Prefix=f"{username}/"
        )
        return [obj['Key'] for obj in response.get('Contents', [])]
    except ClientError as e:
        st.error(f"Error fetching reports: {e}")
        return []


# Authentication functions
def cognito_signup(username, password, email):
    try:
        response = cognito.sign_up(
            ClientId=COGNITO_CLIENT_ID,
            Username=username,
            Password=password,
            UserAttributes=[{'Name': 'email', 'Value': email}]
        )
        return True
    except ClientError as e:
        st.error(f"Signup error: {e.response['Error']['Message']}")
        return False


def cognito_login(username, password):
    try:
        response = cognito.initiate_auth(
            ClientId=COGNITO_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={'USERNAME': username, 'PASSWORD': password}
        )
        return True
    except ClientError as e:
        st.error(f"Login error: {e.response['Error']['Message']}")
        return False


# Initialize session state
if "username" not in st.session_state:
    st.session_state.update({
        "username": "",
        "messages": [],
        "otp": None,
        "password_reset_user": None,
        "symptom_suggestions": [],
        "medications": [],
        "doctor_chat": [],
        "appointments": []
    })


# Medical functions
def get_symptom_suggestions():
    diseases_df = load_diseases()
    all_symptoms = set()
    for symptoms in diseases_df['Symptoms']:
        all_symptoms.update([s.strip().lower() for s in symptoms.split(",")])
    return sorted(all_symptoms)


def handle_symptom_analysis(symptoms):
    diseases_df = load_diseases()
    symptoms = [s.strip().lower() for s in symptoms.split(",") if s.strip()]
    matches = []

    for _, row in diseases_df.iterrows():
        disease_symptoms = [s.strip().lower() for s in row['Symptoms'].split(",")]
        matches_count = sum(1 for s in symptoms if s in disease_symptoms)
        if matches_count > 0:
            matches.append((row['Disease'], row['Recommendations'], matches_count))

    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


@st.cache_resource
def train_model():
    diseases_df = load_diseases()
    if not diseases_df.empty:
        X = diseases_df['Symptoms'].str.lower()
        y = diseases_df['Disease']
        model = make_pipeline(
            CountVectorizer(analyzer=lambda x: [s.strip() for s in x.split(',')]),
            MultinomialNB()
        )
        model.fit(X, y)
        return model
    return None


# New Features ==============================================================
def check_interactions(current_meds):
    """Check for drug interactions"""
    interactions = []
    for med1 in current_meds:
        for med2 in current_meds:
            if med1 != med2 and med1.lower() in DRUG_INTERACTIONS.get(med2.lower(), []):
                interactions.append(f"{med1} ↔ {med2}")
    return interactions


async def get_weather(location):
    """Get weather data"""
    async with python_weather.Client(unit=python_weather.METRIC) as client:
        return await client.get(location)


def update_user_history(username, symptoms, recommendations):
    """Update user medical history"""
    try:
        users_table = dynamodb.Table(os.getenv("USERS_TABLE"))
        response = users_table.update_item(
            Key={'username': username},
            UpdateExpression="SET symptoms_history = list_append(if_not_exists(symptoms_history, :empty_list), recommendations_history = list_append(if_not_exists(recommendations_history, :empty_list)",
            ExpressionAttributeValues={
                ':empty_list': [],
                ':s': [symptoms],
                ':r': [recommendations]
            }
        )
    except ClientError as e:
        st.error(f"History update error: {e}")


# Admin Panel
def show_admin_panel():
    st.title("Admin Dashboard")
    tab1, tab2, tab3 = st.tabs(["Patients", "Diseases", "System Health"])

    with tab1:
        st.subheader("Patient Records")
        users_df = load_users()

        with st.expander("Filter Patients"):
            col1, col2 = st.columns(2)
            filter_email = col1.text_input("Filter by Email")
            filter_phone = col2.text_input("Filter by Phone")

        if filter_email:
            users_df = users_df[users_df['email'].str.contains(filter_email, case=False)]
        if filter_phone:
            users_df = users_df[users_df['phone'].str.contains(filter_phone)]

        st.dataframe(users_df)

        selected_user = st.selectbox("Select user to manage", users_df['username'])
        if st.button("Delete User"):
            try:
                USERS_TABLE.delete_item(Key={'username': selected_user})
                st.success("User deleted successfully!")
                st.rerun()
            except ClientError as e:
                st.error(f"Deletion error: {e}")

    with tab2:
        st.subheader("Manage Diseases")
        diseases_df = load_diseases()

        with st.expander("Add New Disease"):
            with st.form("disease_form"):
                disease = st.text_input("Disease Name")
                symptoms = st.text_area("Symptoms (comma-separated)")
                recommendations = st.text_area("Recommendations")

                if st.form_submit_button("Add Disease"):
                    new_disease = {
                        'Disease': disease,
                        'Symptoms': symptoms,
                        'Recommendations': recommendations
                    }
                    save_disease(new_disease)
                    st.success("Disease added successfully!")

        with st.expander("Edit Diseases"):
            selected_disease = st.selectbox("Select disease", diseases_df['Disease'])
            if selected_disease:
                disease_data = diseases_df[diseases_df['Disease'] == selected_disease].iloc[0]

                with st.form("edit_disease"):
                    new_name = st.text_input("Name", value=disease_data['Disease'])
                    new_symptoms = st.text_area("Symptoms", value=disease_data['Symptoms'])
                    new_rec = st.text_area("Recommendations", value=disease_data['Recommendations'])

                    if st.form_submit_button("Update"):
                        DISEASES_TABLE.delete_item(Key={'Disease': selected_disease})
                        save_disease({
                            'Disease': new_name,
                            'Symptoms': new_symptoms,
                            'Recommendations': new_rec
                        })
                        st.success("Disease updated!")
                        st.rerun()

                if st.button("Delete Disease"):
                    DISEASES_TABLE.delete_item(Key={'Disease': selected_disease})
                    st.success("Disease deleted!")
                    st.rerun()

        st.dataframe(diseases_df)


# Main Application
st.title("🏥 Chronic Disease Diagnosis System")

if not st.session_state["username"]:
    col1, col2 = st.columns(2)

    with col1:
        st.header("User Authentication")
        auth_tab = st.radio("Choose action", ["Login", "Sign Up", "Forgot Password"], horizontal=True)

        if auth_tab == "Login":
            with st.form("login_form"):
                username = st.text_input("Username").strip()
                password = st.text_input("Password", type="password")

                if st.form_submit_button("Login"):
                    if cognito_login(username, password):
                        st.session_state.username = username
                        st.rerun()

        elif auth_tab == "Sign Up":
            with st.form("signup_form"):
                username = st.text_input("Username").strip()
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                email = st.text_input("Email")
                phone = st.text_input("Phone Number")
                age = st.number_input("Age", min_value=0, max_value=120)
                gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
                medical_history = st.text_area("Medical History")
                emergency_name = st.text_input("Emergency Contact Name")
                emergency_phone = st.text_input("Emergency Contact Phone")
                emergency_email = st.text_input("Emergency Contact Email")
                files = st.file_uploader("Upload Medical Reports", accept_multiple_files=True)

                if st.form_submit_button("Register"):
                    if password == confirm_password:
                        if cognito_signup(username, password, email):
                            # Upload files to S3
                            for file in files:
                                try:
                                    s3.upload_fileobj(
                                        file,
                                        MEDICAL_REPORTS_BUCKET,
                                        f"{username}/{file.name}"
                                    )
                                except ClientError as e:
                                    st.error(f"Error uploading {file.name}: {e}")

                            new_user = {
                                'username': username,
                                'email': email,
                                'phone': phone,
                                'age': age,
                                'gender': gender,
                                'medical_history': medical_history,
                                'emergency_name': emergency_name,
                                'emergency_phone': emergency_phone,
                                'emergency_email': emergency_email,
                                'last_login': datetime.now().isoformat()
                            }
                            save_user(new_user)
                            st.success("Registration successful! Please login.")

        elif auth_tab == "Forgot Password":
            with st.form("forgot_password"):
                username = st.text_input("Username")
                if st.form_submit_button("Send OTP"):
                    users_df = load_users()
                    user = users_df[users_df['username'] == username]
                    if not user.empty:
                        st.session_state.otp = str(random.randint(100000, 999999))
                        st.session_state.password_reset_user = username
                        st.success(f"Demo OTP: {st.session_state.otp}")

            if st.session_state.password_reset_user:
                with st.form("reset_password"):
                    otp = st.text_input("Enter OTP")
                    new_password = st.text_input("New Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")

                    if st.form_submit_button("Reset Password"):
                        if otp == st.session_state.otp:
                            if new_password == confirm_password:
                                try:
                                    cognito.admin_set_user_password(
                                        UserPoolId=os.getenv("COGNITO_USER_POOL_ID"),
                                        Username=username,
                                        Password=new_password,
                                        Permanent=True
                                    )
                                    st.success("Password updated successfully!")
                                except ClientError as e:
                                    st.error(f"Password reset error: {e}")
                            else:
                                st.error("Passwords do not match")
                        else:
                            st.error("Invalid OTP")

else:
    if st.session_state.username == "admin":
        show_admin_panel()
    else:
        users_df = load_users()
        user_data = users_df[users_df['username'] == st.session_state.username].iloc[0]
        med_df = load_medications(st.session_state.username)
        appointments_df = load_appointments(st.session_state.username)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.header(f"👤 Welcome, {st.session_state.username}")

            with st.expander("📝 Update Profile", expanded=False):
                with st.form("profile_form"):
                    new_email = st.text_input("Email", value=user_data['email'])
                    new_phone = st.text_input("Phone Number", value=user_data['phone'])
                    new_medical_history = st.text_area("Medical History", value=user_data['medical_history'])
                    new_files = st.file_uploader("Upload New Reports", accept_multiple_files=True)

                    if st.form_submit_button("Update Profile"):
                        # Upload new files
                        for file in new_files:
                            try:
                                s3.upload_fileobj(
                                    file,
                                    MEDICAL_REPORTS_BUCKET,
                                    f"{st.session_state.username}/{file.name}"
                                )
                            except ClientError as e:
                                st.error(f"Error uploading {file.name}: {e}")

                        updated_user = {
                            'username': st.session_state.username,
                            'email': new_email,
                            'phone': new_phone,
                            'medical_history': new_medical_history,
                            'age': user_data['age'],
                            'gender': user_data['gender'],
                            'last_login': datetime.now().isoformat()
                        }
                        save_user(updated_user)
                        st.success("Profile updated!")
                        st.rerun()

            st.header("💬 Symptom Analysis Chatbot")
            if not st.session_state.symptom_suggestions:
                st.session_state.symptom_suggestions = get_symptom_suggestions()

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if symptoms := st.chat_input("Describe your symptoms (comma-separated)"):
                st.session_state.messages.append({"role": "user", "content": symptoms})
                matches = handle_symptom_analysis(symptoms)

                if matches:
                    response = "## Possible Conditions:\n\n"
                    for disease, rec, _ in matches[:3]:
                        response += f"### {disease}\n{rec}\n\n"
                    # Update user history
                    update_user_history(
                        st.session_state.username,
                        symptoms,
                        "\n".join([d[0] for d in matches]))
                else:
                    response = "No matching conditions found. Please consult a doctor."

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

            st.subheader("🤖 AI-Powered Diagnosis")
            symptoms_text = st.text_area("Enter symptoms (comma-separated):")
            if st.button("Analyze with AI"):
                model = train_model()
                if model:
                    try:
                        prediction = model.predict([symptoms_text.lower()])[0]
                        proba = model.predict_proba([symptoms_text.lower()]).max()
                        st.success(f"Predicted Disease: {prediction} (Confidence: {proba:.2%})")
                        update_user_history(
                            st.session_state.username,
                            symptoms_text,
                            prediction
                        )
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                else:
                    st.error("Diagnosis model not available")

            # MediLocker Section
            with st.expander("📁 MediLocker - Medical Document Storage", expanded=True):
                st.subheader("Medical Document Storage")

                # File Upload
                with st.form("upload_report"):
                    uploaded_files = st.file_uploader(
                        "Upload Medical Reports",
                        type=['pdf', 'jpg', 'png', 'docx', 'txt'],
                        accept_multiple_files=True
                    )
                    if st.form_submit_button("Upload Documents"):
                        for file in uploaded_files:
                            try:
                                s3.upload_fileobj(
                                    file,
                                    MEDICAL_REPORTS_BUCKET,
                                    f"{st.session_state.username}/{file.name}"
                                )
                                st.success(f"Uploaded {file.name} successfully!")
                            except ClientError as e:
                                st.error(f"Error uploading {file.name}: {e}")

                # Document List
                st.subheader("Your Medical Documents")
                documents = get_medical_reports(st.session_state.username)

                if documents:
                    for doc in documents:
                        col1, col2 = st.columns([4, 1])
                        filename = doc.split("/")[-1]

                        with col1:
                            st.write(f"📄 {filename}")

                        with col2:
                            if st.button(f"Download {filename}", key=f"dl_{filename}"):
                                try:
                                    url = s3.generate_presigned_url(
                                        'get_object',
                                        Params={
                                            'Bucket': MEDICAL_REPORTS_BUCKET,
                                            'Key': doc
                                        },
                                        ExpiresIn=3600
                                    )
                                    st.markdown(f"[Download Link]({url})")
                                except ClientError as e:
                                    st.error(f"Error generating download link: {e}")
                else:
                    st.info("No medical documents found in your MediLocker")

        with col2:
            st.header("🩺 Health Overview")
            st.metric("Age", user_data['age'])
            st.metric("Gender", user_data['gender'])

            # Emergency Alert Button
            if st.button("🚨 Emergency Alert"):
                try:
                    msg = MIMEText(f"Emergency alert from {st.session_state.username}!")
                    msg['Subject'] = 'MEDICAL EMERGENCY'
                    msg['From'] = os.getenv("SMTP_USER")
                    msg['To'] = user_data['emergency_email']

                    with smtplib.SMTP(os.getenv("SMTP_SERVER"), os.getenv("SMTP_PORT")) as server:
                        server.starttls()
                        server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
                        server.send_message(msg)
                    st.success("Alert sent to emergency contact!")
                except Exception as e:
                    st.error(f"Failed to send alert: {str(e)}")

            # Weather Integration
            with st.expander("🌤️ Weather Advisory"):
                location = st.text_input("Enter location for weather updates")
                if location:
                    try:
                        weather = asyncio.run(get_weather(location))
                        st.write(f"""
                        **Temperature**: {weather.current.temperature}°C  
                        **Humidity**: {weather.current.humidity}%  
                        **Conditions**: {weather.current.description}
                        """)
                    except Exception as e:
                        st.error(f"Weather data error: {str(e)}")

            with st.expander("📅 Appointments"):
                if not appointments_df.empty:
                    for idx, row in appointments_df.iterrows():
                        st.write(f"""
                        **Doctor**: {row['doctor']}  
                        **When**: {row['datetime'].strftime("%Y-%m-%d %H:%M")}  
                        **Reason**: {row['reason']}  
                        **Confirmed**: {'✅' if row['confirmed'] else '❌'}
                        """)
                else:
                    st.info("No upcoming appointments")

                with st.form("new_appointment"):
                    doctor = st.text_input("Doctor Name")
                    appointment_date = st.date_input("Date", min_value=datetime.today())
                    appointment_time = st.time_input("Time")
                    reason = st.text_area("Reason")

                    if st.form_submit_button("Schedule"):
                        datetime_str = f"{appointment_date.isoformat()} {appointment_time.strftime('%H:%M')}"
                        new_appt = pd.DataFrame([{
                            'doctor': doctor,
                            'datetime': datetime_str,
                            'reason': reason,
                            'confirmed': False
                        }])
                        appointments_df = pd.concat([appointments_df, new_appt])
                        save_appointments(st.session_state.username, appointments_df)
                        st.success("Appointment scheduled!")

            with st.expander("💊 Medications"):
                if not med_df.empty:
                    current_meds = []
                    for idx, row in med_df.iterrows():
                        current_meds.append(row['name'])
                        st.write(f"""
                        **Medication**: {row['name']}  
                        **Dosage**: {row['dosage']}  
                        **Frequency**: {row['frequency']}
                        """)
                    interactions = check_interactions(current_meds)
                    if interactions:
                        st.warning("⚠️ Drug Interactions Detected:")
                        for interaction in interactions:
                            st.write(interaction)
                else:
                    st.info("No medications recorded")

                with st.form("add_medication"):
                    name = st.text_input("Medication Name")
                    dosage = st.text_input("Dosage")
                    frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
                    if st.form_submit_button("Add Medication"):
                        new_med = pd.DataFrame([{
                            'name': name,
                            'dosage': dosage,
                            'frequency': frequency,
                            'start_date': datetime.now().strftime("%Y-%m-%d"),
                            'end_date': ""
                        }])
                        med_df = pd.concat([med_df, new_med])
                        save_medications(st.session_state.username, med_df)
                        st.success("Medication added!")

if st.sidebar.button("Logout"):
    st.session_state.username = ""
    st.rerun()