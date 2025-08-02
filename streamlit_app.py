import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="HDB Housing Explorer for Families",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the HDB dataset with caching for better performance"""
    try:
        # Try to load the full dataset
        df = pd.read_csv('datasets/train.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Create a sample dataset for demonstration if file is too large
        st.warning("Loading sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if the main dataset cannot be loaded"""
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'resale_price': np.random.normal(400000, 150000, n_samples),
        'town': np.random.choice(['TAMPINES', 'BEDOK', 'JURONG WEST', 'WOODLANDS', 'PUNGGOL'], n_samples),
        'flat_type': np.random.choice(['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE'], n_samples),
        'floor_area_sqm': np.random.normal(85, 20, n_samples),
        'pri_sch_nearest_distance': np.random.exponential(500, n_samples),
        'mrt_nearest_distance': np.random.exponential(800, n_samples),
        'Mall_Nearest_Distance': np.random.exponential(1000, n_samples),
        'Hawker_Nearest_Distance': np.random.exponential(300, n_samples),
        'hdb_age': np.random.randint(1, 40, n_samples),
        'Tranc_Year': np.random.choice(range(2015, 2023), n_samples),
    }
    
    return pd.DataFrame(sample_data)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 HDB Housing Explorer for Families with Young Children</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading housing data..."):
        df = load_data()
    
    # Sidebar for filters
    st.sidebar.title("🔍 Filters")
    
    # Data overview
    st.sidebar.markdown("### Dataset Overview")
    st.sidebar.info(f"Total properties: {len(df):,}")
    
    # Price range filter
    if 'resale_price' in df.columns:
        price_min, price_max = st.sidebar.slider(
            "Price Range (SGD)",
            min_value=int(df['resale_price'].min()),
            max_value=int(df['resale_price'].max()),
            value=(int(df['resale_price'].min()), int(df['resale_price'].max())),
            format="$%d"
        )
        df = df[(df['resale_price'] >= price_min) & (df['resale_price'] <= price_max)]
    
    # Town filter
    if 'town' in df.columns:
        towns = st.sidebar.multiselect(
            "Select Towns",
            options=sorted(df['town'].unique()),
            default=sorted(df['town'].unique())
        )
        if towns:
            df = df[df['town'].isin(towns)]
    
    # Flat type filter
    if 'flat_type' in df.columns:
        flat_types = st.sidebar.multiselect(
            "Flat Types",
            options=sorted(df['flat_type'].unique()),
            default=sorted(df['flat_type'].unique())
        )
        if flat_types:
            df = df[df['flat_type'].isin(flat_types)]
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🏫 Schools & Education", 
        "🚇 Transportation", 
        "🛍️ Amenities", 
        "📈 Price Analysis"
    ])
    
    with tab1:
        show_overview(df)
    
    with tab2:
        show_schools_analysis(df)
    
    with tab3:
        show_transportation_analysis(df)
    
    with tab4:
        show_amenities_analysis(df)
    
    with tab5:
        show_price_analysis(df)

def show_overview(df):
    """Display overview of the housing data"""
    st.markdown('<h2 class="sub-header">📊 Housing Data Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = df['resale_price'].mean() if 'resale_price' in df.columns else 0
        st.metric("Average Price", f"${avg_price:,.0f}")
    
    with col2:
        total_properties = len(df)
        st.metric("Total Properties", f"{total_properties:,}")
    
    with col3:
        avg_size = df['floor_area_sqm'].mean() if 'floor_area_sqm' in df.columns else 0
        st.metric("Average Size", f"{avg_size:.0f} sqm")
    
    with col4:
        unique_towns = df['town'].nunique() if 'town' in df.columns else 0
        st.metric("Towns Available", f"{unique_towns}")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'flat_type' in df.columns:
            fig = px.pie(df, names='flat_type', title="Distribution by Flat Type")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'town' in df.columns:
            town_counts = df['town'].value_counts().head(10)
            fig = px.bar(x=town_counts.values, y=town_counts.index, 
                        orientation='h', title="Top 10 Towns by Number of Properties")
            st.plotly_chart(fig, use_container_width=True)
    
    # Family-friendly features info box
    st.markdown("""
    <div class="info-box">
    <h3>🏠 Family Housing Considerations</h3>
    <p>This explorer focuses on housing features important for families with young children:</p>
    <ul>
        <li><strong>School Proximity:</strong> Distance to quality primary and secondary schools</li>
        <li><strong>Transportation:</strong> Access to MRT stations and bus stops</li>
        <li><strong>Amenities:</strong> Nearby malls, hawker centres, and community facilities</li>
        <li><strong>Safety & Space:</strong> Flat types suitable for growing families</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_schools_analysis(df):
    """Analyze school proximity and its impact on housing"""
    st.markdown('<h2 class="sub-header">🏫 Schools & Education Analysis</h2>', unsafe_allow_html=True)
    
    if 'pri_sch_nearest_distance' not in df.columns:
        st.warning("School distance data not available in the current dataset.")
        return
    
    # School distance distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='pri_sch_nearest_distance', 
                          title="Distribution of Distance to Nearest Primary School",
                          labels={'pri_sch_nearest_distance': 'Distance (meters)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'sec_sch_nearest_dist' in df.columns:
            fig = px.histogram(df, x='sec_sch_nearest_dist', 
                              title="Distribution of Distance to Nearest Secondary School",
                              labels={'sec_sch_nearest_dist': 'Distance (meters)'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Price vs School Distance Analysis
    if 'resale_price' in df.columns:
        st.subheader("💰 How School Proximity Affects Housing Prices")
        
        # Create school distance categories
        df['school_distance_category'] = pd.cut(df['pri_sch_nearest_distance'], 
                                               bins=[0, 500, 1000, 2000, float('inf')],
                                               labels=['<500m', '500m-1km', '1-2km', '>2km'])
        
        avg_prices = df.groupby('school_distance_category')['resale_price'].mean().reset_index()
        
        fig = px.bar(avg_prices, x='school_distance_category', y='resale_price',
                    title="Average Housing Price by Distance to Primary School",
                    labels={'school_distance_category': 'Distance to School', 
                           'resale_price': 'Average Price (SGD)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        correlation = df[['pri_sch_nearest_distance', 'resale_price']].corr().iloc[0,1]
        st.metric("Correlation: School Distance vs Price", f"{correlation:.3f}")
        
        if correlation < -0.1:
            st.success("✅ Properties closer to schools tend to be more expensive")
        elif correlation > 0.1:
            st.info("ℹ️ Properties farther from schools tend to be more expensive")
        else:
            st.info("ℹ️ Weak correlation between school distance and price")

def show_transportation_analysis(df):
    """Analyze transportation accessibility"""
    st.markdown('<h2 class="sub-header">🚇 Transportation Accessibility</h2>', unsafe_allow_html=True)
    
    # MRT and Bus Stop Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        if 'mrt_nearest_distance' in df.columns:
            fig = px.histogram(df, x='mrt_nearest_distance', 
                              title="Distance to Nearest MRT Station",
                              labels={'mrt_nearest_distance': 'Distance (meters)'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'bus_stop_nearest_distance' in df.columns:
            fig = px.histogram(df, x='bus_stop_nearest_distance', 
                              title="Distance to Nearest Bus Stop",
                              labels={'bus_stop_nearest_distance': 'Distance (meters)'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Transportation convenience analysis
    if all(col in df.columns for col in ['mrt_nearest_distance', 'resale_price']):
        st.subheader("🚊 Transportation Impact on Housing Prices")
        
        # Create MRT distance categories
        df['mrt_distance_category'] = pd.cut(df['mrt_nearest_distance'], 
                                            bins=[0, 500, 1000, 1500, float('inf')],
                                            labels=['<500m', '500m-1km', '1-1.5km', '>1.5km'])
        
        avg_prices = df.groupby('mrt_distance_category')['resale_price'].mean().reset_index()
        
        fig = px.bar(avg_prices, x='mrt_distance_category', y='resale_price',
                    title="Average Housing Price by Distance to MRT",
                    labels={'mrt_distance_category': 'Distance to MRT', 
                           'resale_price': 'Average Price (SGD)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Bus interchange analysis
        if 'bus_interchange' in df.columns:
            interchange_prices = df.groupby('bus_interchange')['resale_price'].mean().reset_index()
            interchange_prices['bus_interchange'] = interchange_prices['bus_interchange'].map({True: 'Yes', False: 'No'})
            
            fig = px.bar(interchange_prices, x='bus_interchange', y='resale_price',
                        title="Average Price: Near Bus Interchange vs Not",
                        labels={'bus_interchange': 'Near Bus Interchange', 
                               'resale_price': 'Average Price (SGD)'})
            st.plotly_chart(fig, use_container_width=True)

def show_amenities_analysis(df):
    """Analyze proximity to family-friendly amenities"""
    st.markdown('<h2 class="sub-header">🛍️ Family Amenities Analysis</h2>', unsafe_allow_html=True)
    
    # Mall and Hawker Centre Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Mall_Nearest_Distance' in df.columns:
            fig = px.histogram(df, x='Mall_Nearest_Distance', 
                              title="Distance to Nearest Mall",
                              labels={'Mall_Nearest_Distance': 'Distance (meters)'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Hawker_Nearest_Distance' in df.columns:
            fig = px.histogram(df, x='Hawker_Nearest_Distance', 
                              title="Distance to Nearest Hawker Centre",
                              labels={'Hawker_Nearest_Distance': 'Distance (meters)'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Amenity density analysis
    st.subheader("🏢 Amenity Density Impact")
    
    amenity_cols = [col for col in df.columns if 'Within_500m' in col or 'Within_1km' in col]
    
    if amenity_cols:
        selected_amenity = st.selectbox("Select Amenity Type", amenity_cols)
        
        if 'resale_price' in df.columns and selected_amenity in df.columns:
            fig = px.scatter(df.sample(min(1000, len(df))), 
                            x=selected_amenity, y='resale_price',
                            title=f"Price vs {selected_amenity.replace('_', ' ').title()}",
                            labels={selected_amenity: 'Number of Amenities', 
                                   'resale_price': 'Resale Price (SGD)'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Community features analysis
    community_features = ['residential', 'commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion']
    available_features = [col for col in community_features if col in df.columns]
    
    if available_features and 'resale_price' in df.columns:
        st.subheader("🏘️ Community Features Impact on Price")
        
        feature_impact = {}
        for feature in available_features:
            if df[feature].dtype == 'bool' or df[feature].nunique() == 2:
                avg_with = df[df[feature] == True]['resale_price'].mean()
                avg_without = df[df[feature] == False]['resale_price'].mean()
                feature_impact[feature.replace('_', ' ').title()] = avg_with - avg_without
        
        if feature_impact:
            fig = px.bar(x=list(feature_impact.values()), 
                        y=list(feature_impact.keys()),
                        orientation='h',
                        title="Price Premium for Community Features",
                        labels={'x': 'Price Difference (SGD)', 'y': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)

def show_price_analysis(df):
    """Show comprehensive price analysis"""
    st.markdown('<h2 class="sub-header">📈 Price Analysis & Trends</h2>', unsafe_allow_html=True)
    
    if 'resale_price' not in df.columns:
        st.warning("Price data not available in the current dataset.")
        return
    
    # Price distribution by flat type
    col1, col2 = st.columns(2)
    
    with col1:
        if 'flat_type' in df.columns:
            fig = px.box(df, x='flat_type', y='resale_price',
                        title="Price Distribution by Flat Type")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'floor_area_sqm' in df.columns:
            fig = px.scatter(df.sample(min(1000, len(df))), 
                            x='floor_area_sqm', y='resale_price',
                            title="Price vs Floor Area",
                            labels={'floor_area_sqm': 'Floor Area (sqm)', 
                                   'resale_price': 'Price (SGD)'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    if 'Tranc_Year' in df.columns:
        st.subheader("📅 Price Trends Over Time")
        
        yearly_prices = df.groupby('Tranc_Year')['resale_price'].agg(['mean', 'median']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly_prices['Tranc_Year'], y=yearly_prices['mean'],
                                mode='lines+markers', name='Average Price'))
        fig.add_trace(go.Scatter(x=yearly_prices['Tranc_Year'], y=yearly_prices['median'],
                                mode='lines+markers', name='Median Price'))
        
        fig.update_layout(title="Housing Price Trends Over Time",
                         xaxis_title="Year",
                         yaxis_title="Price (SGD)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Price recommendations for families
    st.subheader("💡 Family Housing Recommendations")
    
    # Calculate family-friendly scores
    family_score_factors = []
    
    if 'pri_sch_nearest_distance' in df.columns:
        df['school_score'] = 1 / (1 + df['pri_sch_nearest_distance'] / 1000)
        family_score_factors.append('school_score')
    
    if 'mrt_nearest_distance' in df.columns:
        df['transport_score'] = 1 / (1 + df['mrt_nearest_distance'] / 1000)
        family_score_factors.append('transport_score')
    
    if 'Mall_Within_1km' in df.columns:
        df['amenity_score'] = df['Mall_Within_1km'] / df['Mall_Within_1km'].max()
        family_score_factors.append('amenity_score')
    
    if family_score_factors:
        df['family_friendly_score'] = df[family_score_factors].mean(axis=1)
        
        # Top family-friendly properties
        top_family_properties = df.nlargest(10, 'family_friendly_score')[
            ['town', 'flat_type', 'resale_price', 'family_friendly_score']
        ].round(2)
        
        st.subheader("🏆 Top 10 Family-Friendly Properties")
        st.dataframe(top_family_properties, use_container_width=True)
        
        # Value for money analysis
        df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm'] if 'floor_area_sqm' in df.columns else df['resale_price']
        df['value_score'] = df['family_friendly_score'] / (df['price_per_sqm'] / df['price_per_sqm'].median())
        
        best_value = df.nlargest(10, 'value_score')[
            ['town', 'flat_type', 'resale_price', 'value_score']
        ].round(2)
        
        st.subheader("💰 Best Value for Money (Family-Friendly)")
        st.dataframe(best_value, use_container_width=True)

if __name__ == "__main__":
    main()
