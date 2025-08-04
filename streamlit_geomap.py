import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# --- Title
st.title("Singapore HDB Resale: Regional Insights via AI-Enhanced Map")

st.markdown("""
This interactive map visualizes **popularity**, **average resale price**, and **value-for-money score** across Singapore regions.  
Rather than relying on generic charts, we use a geo-overlay to help you drill into areas that matter.
""")

# --- Simulated Data by Region (can replace with real aggregation)
data = pd.DataFrame({
    'region': ['Central', 'East', 'West', 'North'],
    'latitude': [1.2966, 1.3508, 1.3521, 1.4380],
    'longitude': [103.8496, 103.9332, 103.7074, 103.7865],
    'avg_resale_price': [750000, 620000, 580000, 600000],
    'popularity_score': [70, 95, 60, 65],
    'value_for_money': [0.00018, 0.00021, 0.00024, 0.0002]  # $/sqm
})

# --- Create Folium Map
m = folium.Map(location=[1.3521, 103.8198], zoom_start=11, tiles="CartoDB positron")
marker_cluster = MarkerCluster().add_to(m)

# --- Add Circle Markers
for _, row in data.iterrows():
    popup_html = f"""
    <b>Region:</b> {row['region']}<br>
    <b>Avg Resale Price:</b> ${row['avg_resale_price']:,}<br>
    <b>Popularity Score:</b> {row['popularity_score']}<br>
    <b>Value for Money:</b> {row['value_for_money']:.5f} sqm/$
    """
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=15,
        popup=folium.Popup(popup_html, max_width=300),
        color='blue' if row['region'] == 'Central' else 'green' if row['region'] == 'West' else 'orange',
        fill=True,
        fill_opacity=0.7
    ).add_to(marker_cluster)

# --- Show Map
st_folium(m, width=700, height=500)

# --- Takeaways
st.subheader("🔍 Key Takeaways")
st.markdown("""
- 🏙️ **Central** is the most expensive region, but not the most popular.
- 🌅 **East** wins on popularity—possibly due to lifestyle, food, and connectivity.
- 💸 **West** offers the best bang for your buck in terms of price-to-space ratio.
- 🧠 AI-powered geospatial analysis helps tell a more intuitive story than traditional charts.
""")

# --- Footer
st.caption("Demo based on simulated data. Replace with your resale flat dataset for full insight.")
