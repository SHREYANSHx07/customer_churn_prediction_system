"""
Streamlit Frontend for Customer Churn Prediction System
Optimized for Mac M1
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from io import StringIO
import time
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://127.0.0.1:8000/api')

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        color: #155724;
    }
    .error-message {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint, method="GET", data=None, files=None, timeout=60):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
        
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, timeout=timeout)
            else:
                response = requests.post(url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json(), None
    
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. Make sure the Django server is running."
    except requests.exceptions.HTTPError as e:
        try:
            error_data = e.response.json()
            return None, error_data.get('error', str(e))
        except:
            return None, f"HTTP Error: {e.response.status_code}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def load_customer_data():
    """Load customer data from API"""
    data, error = make_api_request("customers/")
    if error:
        return None, error
    return data, None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Data Management", "Model Training", "Predictions", "Analytics", "Export"]
    )
    
    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Management":
        show_data_management()
    elif page == "Model Training":
        show_model_training()
    elif page == "Predictions":
        show_predictions()
    elif page == "Analytics":
        show_analytics()
    elif page == "Export":
        show_export()

def show_dashboard():
    """Dashboard page"""
    st.header("📊 Dashboard")
    
    # Load customer data
    with st.spinner("Loading customer data..."):
        data, error = load_customer_data()
    
    if error:
        st.error(f"Error loading data: {error}")
        return
    
    if not data or not data.get('customers'):
        st.warning("No customer data available. Please upload data first.")
        return
    
    customers = data['customers']
    total_customers = data['pagination']['total_count']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", total_customers)
    
    with col2:
        high_risk = sum(1 for c in customers if c.get('risk_level') == 'High')
        st.metric("High Risk Customers", high_risk)
    
    with col3:
        avg_monthly = np.mean([c.get('monthly_charges', 0) for c in customers])
        st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    
    with col4:
        avg_tenure = np.mean([c.get('tenure', 0) for c in customers])
        st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level distribution
        risk_counts = {}
        for customer in customers:
            risk = customer.get('risk_level', 'Unknown')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        if risk_counts:
            fig = px.pie(
                values=list(risk_counts.values()),
                names=list(risk_counts.keys()),
                title="Risk Level Distribution",
                color_discrete_map={
                    'High': '#ff4444',
                    'Medium': '#ffaa00',
                    'Low': '#44ff44'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Contract type distribution
        contract_counts = {}
        for customer in customers:
            contract = customer.get('contract', 'Unknown')
            contract_counts[contract] = contract_counts.get(contract, 0) + 1
        
        if contract_counts:
            fig = px.bar(
                x=list(contract_counts.keys()),
                y=list(contract_counts.values()),
                title="Contract Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent customers table
    st.subheader("Recent Customers")
    df = pd.DataFrame(customers[:10])  # Show first 10
    if not df.empty:
        display_columns = ['customer_id', 'gender', 'tenure', 'monthly_charges', 'risk_level', 'churn_probability']
        available_columns = [col for col in display_columns if col in df.columns]
        st.dataframe(df[available_columns], use_container_width=True)

def show_data_management():
    """Data management page"""
    st.header("📁 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["Upload CSV", "Upload from URL", "Generate Sample Data"])
    
    with tab1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            if st.button("Upload CSV", type="primary"):
                with st.spinner("Uploading and processing CSV..."):
                    files = {"file": uploaded_file}
                    data, error = make_api_request("upload-csv/", method="POST", files=files, timeout=120)
                
                if error:
                    st.error(f"Upload failed: {error}")
                else:
                    st.success(f"✅ {data['message']}")
                    st.info(f"Created/Updated: {data['created_count']} customers")
                    if data.get('errors'):
                        st.warning("Some errors occurred:")
                        for err in data['errors'][:5]:
                            st.text(err)
    
    with tab2:
        st.subheader("Upload from URL")
        url = st.text_input("CSV File URL", placeholder="https://example.com/data.csv")
        
        if url and st.button("Upload from URL", type="primary"):
            with st.spinner("Downloading and processing CSV..."):
                data, error = make_api_request("upload-from-url/", method="POST", data={"url": url}, timeout=120)
            
            if error:
                st.error(f"Upload failed: {error}")
            else:
                st.success(f"✅ {data['message']}")
                st.info(f"Created: {data['created_count']}, Updated: {data['updated_count']}")
    
    with tab3:
        st.subheader("Generate Sample Data")
        n_customers = st.slider("Number of customers to generate", 10, 1000, 100)
        
        if st.button("Generate Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                data, error = make_api_request("generate-sample-data/", method="POST", data={"n_customers": n_customers})
            
            if error:
                st.error(f"Generation failed: {error}")
            else:
                st.success(f"✅ {data['message']}")
                st.info(f"Generated: {data['created_count']} customers")
    
    # Show current data status
    st.subheader("Current Data Status")
    with st.spinner("Loading data status..."):
        data, error = load_customer_data()
    
    if error:
        st.error(f"Error loading data: {error}")
    elif data:
        total_customers = data['pagination']['total_count']
        st.metric("Total Customers in Database", total_customers)
        
        if total_customers > 0:
            # Show sample of data
            customers = data['customers'][:5]
            df = pd.DataFrame(customers)
            st.subheader("Sample Data Preview")
            st.dataframe(df, use_container_width=True)

def show_model_training():
    """Model training page"""
    st.header("🤖 Model Training")
    
    # Check data availability
    with st.spinner("Checking data availability..."):
        data, error = load_customer_data()
    
    if error:
        st.error(f"Error checking data: {error}")
        return
    
    total_customers = data['pagination']['total_count'] if data else 0
    
    if total_customers < 10:
        st.warning("⚠️ Need at least 10 customers to train a model. Please upload data first.")
        return
    
    st.success(f"✅ {total_customers} customers available for training")
    
    # Training configuration
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Algorithm:** Random Forest")
        st.info("**Features:** Demographics, Services, Contract, Billing")
    
    with col2:
        st.info("**Validation:** 80/20 Train/Test Split")
        st.info("**Optimization:** Mac M1 Compatible")
    
    # Train model button
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model... This may take a few minutes."):
            progress_bar = st.progress(0)
            
            # Simulate progress
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
            
            data, error = make_api_request("train-model/", method="POST", timeout=180)
        
        progress_bar.empty()
        
        if error:
            st.error(f"❌ Training failed: {error}")
        else:
            st.success("✅ Model trained successfully!")
            
            # Display metrics
            metrics = data['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
            
            # Training details
            st.subheader("Training Details")
            st.info(f"Training Samples: {data['training_samples']}")
            st.info(f"Test Samples: {data['test_samples']}")
            st.info(f"Features Used: {data['feature_count']}")
            
            # Feature importance (if available)
            if data.get('feature_names'):
                st.subheader("Features Used")
                features_df = pd.DataFrame({
                    'Feature': data['feature_names']
                })
                st.dataframe(features_df, use_container_width=True)

def show_predictions():
    """Predictions page"""
    st.header("🔮 Churn Predictions")
    
    # Check if model exists and data is available
    with st.spinner("Checking system status..."):
        data, error = load_customer_data()
    
    if error:
        st.error(f"Error checking data: {error}")
        return
    
    total_customers = data['pagination']['total_count'] if data else 0
    
    if total_customers == 0:
        st.warning("No customer data available. Please upload data first.")
        return
    
    st.info(f"📊 {total_customers} customers available for prediction")
    
    # Generate predictions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎯 Predict All Customers", type="primary", use_container_width=True):
            with st.spinner("Generating predictions... This may take a moment."):
                pred_data, pred_error = make_api_request("predict/", method="POST", timeout=120)
            
            if pred_error:
                st.error(f"❌ Prediction failed: {pred_error}")
            else:
                st.success("✅ Predictions generated successfully!")
                
                # Show prediction summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", pred_data['total_predictions'])
                with col2:
                    st.metric("High Risk", pred_data['high_risk_count'])
                with col3:
                    risk_rate = (pred_data['high_risk_count'] / pred_data['total_predictions']) * 100
                    st.metric("Risk Rate", f"{risk_rate:.1f}%")
    
    with col2:
        st.info("**Prediction Process:**\n1. Load trained model\n2. Process customer features\n3. Calculate churn probability\n4. Assign risk levels")
    
    # Show predictions table
    st.subheader("Customer Predictions")
    
    # Reload data to get updated predictions
    with st.spinner("Loading predictions..."):
        data, error = load_customer_data()
    
    if data and data.get('customers'):
        customers = data['customers']
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
        
        with col2:
            churn_filter = st.selectbox("Filter by Prediction", ["All", "Yes", "No"])
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Risk Level", "Churn Probability", "Customer ID"])
        
        # Apply filters
        filtered_customers = customers
        
        if risk_filter != "All":
            filtered_customers = [c for c in filtered_customers if c.get('risk_level') == risk_filter]
        
        if churn_filter != "All":
            filtered_customers = [c for c in filtered_customers if c.get('churn') == churn_filter]
        
        # Create DataFrame
        if filtered_customers:
            df = pd.DataFrame(filtered_customers)
            
            # Select relevant columns
            display_columns = [
                'customer_id', 'gender', 'tenure', 'monthly_charges', 
                'churn_probability', 'risk_level', 'churn'
            ]
            available_columns = [col for col in display_columns if col in df.columns]
            
            # Sort data
            if sort_by == "Risk Level" and 'risk_level' in df.columns:
                risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
                df['risk_order'] = df['risk_level'].map(risk_order)
                df = df.sort_values('risk_order').drop('risk_order', axis=1)
            elif sort_by == "Churn Probability" and 'churn_probability' in df.columns:
                df = df.sort_values('churn_probability', ascending=False)
            
            # Display table
            st.dataframe(
                df[available_columns].head(50),  # Limit to 50 rows for performance
                use_container_width=True
            )
            
            if len(filtered_customers) > 50:
                st.info(f"Showing first 50 of {len(filtered_customers)} customers")
        
        else:
            st.info("No customers match the selected filters.")

def show_analytics():
    """Analytics page"""
    st.header("📈 Analytics & Insights")
    
    # Load analytics data
    with st.spinner("Loading analytics..."):
        analytics_data, error = make_api_request("analytics/")
    
    if error:
        st.error(f"Error loading analytics: {error}")
        return
    
    if not analytics_data:
        st.warning("No analytics data available.")
        return
    
    # Key metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", analytics_data['total_customers'])
    
    with col2:
        st.metric("Churn Rate", f"{analytics_data['churn_rate']:.1f}%")
    
    with col3:
        avg_monthly = analytics_data['financial']['avg_monthly_charges']
        st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    
    with col4:
        avg_tenure = analytics_data['financial']['avg_tenure']
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        st.subheader("Risk Level Distribution")
        risk_data = analytics_data['risk_distribution']
        
        fig = px.pie(
            values=list(risk_data.values()),
            names=list(risk_data.keys()),
            title="Customer Risk Levels",
            color_discrete_map={
                'High': '#ff4444',
                'Medium': '#ffaa00',
                'Low': '#44ff44'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution
        st.subheader("Demographics")
        gender_data = analytics_data['demographics']['gender']
        
        fig = px.bar(
            x=list(gender_data.keys()),
            y=list(gender_data.values()),
            title="Gender Distribution",
            color=list(gender_data.keys()),
            color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Service adoption
    st.subheader("Service Adoption")
    col1, col2 = st.columns(2)
    
    with col1:
        # Internet service
        internet_data = analytics_data['services']['internet_service']
        fig = px.pie(
            values=list(internet_data.values()),
            names=list(internet_data.keys()),
            title="Internet Service Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Contract types
        contract_data = analytics_data['contract_types']
        fig = px.bar(
            x=list(contract_data.keys()),
            y=list(contract_data.values()),
            title="Contract Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    st.subheader("Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Senior Citizens:** {analytics_data['demographics']['senior_citizens']} customers")
        st.info(f"**With Partners:** {analytics_data['demographics']['with_partners']} customers")
    
    with col2:
        st.info(f"**Phone Service:** {analytics_data['services']['phone_service']} customers")
        st.info(f"**Streaming TV:** {analytics_data['services']['streaming_tv']} customers")
    
    with col3:
        st.info(f"**Streaming Movies:** {analytics_data['services']['streaming_movies']} customers")
        st.info(f"**With Dependents:** {analytics_data['demographics']['with_dependents']} customers")

def show_export():
    """Export page"""
    st.header("💾 Export Data")
    
    st.subheader("Export Customer Predictions")
    st.info("Export all customer data including churn predictions and risk levels as CSV file.")
    
    if st.button("📥 Export to CSV", type="primary", use_container_width=True):
        with st.spinner("Preparing export..."):
            export_data, error = make_api_request("export/")
        
        if error:
            st.error(f"Export failed: {error}")
        else:
            st.success(f"✅ Export ready! {export_data['total_records']} records prepared.")
            
            # Provide download
            csv_data = export_data['csv_data']
            filename = export_data['filename']
            
            st.download_button(
                label="📁 Download CSV File",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
            
            # Show preview
            st.subheader("Data Preview")
            preview_df = pd.read_csv(StringIO(csv_data))
            st.dataframe(preview_df.head(10), use_container_width=True)
            
            st.info(f"Preview showing first 10 of {len(preview_df)} total records")

if __name__ == "__main__":
    main()
