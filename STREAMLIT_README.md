# HDB Housing Explorer for Families

A Streamlit web application for exploring Singapore HDB housing data with a focus on family-friendly features and amenities.

## Features

### 🏠 Overview Dashboard

- Key housing metrics and statistics
- Distribution of properties by flat type and town
- Family housing considerations summary

### 🏫 Schools & Education Analysis

- Distance distribution to primary and secondary schools
- Impact of school proximity on housing prices
- Correlation analysis between school distance and property values

### 🚇 Transportation Analysis

- MRT and bus stop accessibility
- Transportation impact on housing prices
- Bus interchange proximity analysis

### 🛍️ Amenities Analysis

- Mall and hawker centre proximity
- Community features impact on pricing
- Amenity density analysis

### 📈 Price Analysis & Trends

- Price distribution by flat type and floor area
- Historical price trends
- Family-friendly property recommendations
- Best value for money analysis

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Or use the batch file (Windows):

```bash
run_app.bat
```

## Data Requirements

The app expects a CSV file named `train.csv` in the `datasets/` directory with the following key columns:

- `resale_price`: Property sale price in SGD
- `town`: HDB township
- `flat_type`: Type of flat (3 ROOM, 4 ROOM, etc.)
- `floor_area_sqm`: Floor area in square meters
- `pri_sch_nearest_distance`: Distance to nearest primary school
- `mrt_nearest_distance`: Distance to nearest MRT station
- `Mall_Nearest_Distance`: Distance to nearest mall
- `Hawker_Nearest_Distance`: Distance to nearest hawker centre

For a complete list of supported columns, see the main README.md file.

## Key Research Questions Addressed

1. How does proximity to good schools affect housing prices?
2. What is the relationship between distance from bus interchanges and housing prices?
3. How do rental prices vary based on these factors?
4. How does proximity to MRT stations, malls, and hawker centres influence housing prices?
5. What is the impact of flat characteristics on resale price?
6. How do neighborhood features affect family housing choices?

## Family-Focused Features

The app specifically caters to families with young children by analyzing:

- **Safety & Education**: School proximity and quality indicators
- **Transportation**: Public transport accessibility for daily commutes
- **Amenities**: Family-friendly facilities like malls, hawker centres, and parks
- **Space & Value**: Appropriate flat sizes and value-for-money analysis

## Usage Tips

1. Use the sidebar filters to narrow down properties by price range, town, and flat type
2. Navigate through different tabs to explore various aspects of the housing data
3. Look for correlation insights and recommendations in each section
4. Check the family-friendly scores and value recommendations for decision-making

## Note

If the main dataset is too large to load, the app will automatically generate sample data for demonstration purposes.
