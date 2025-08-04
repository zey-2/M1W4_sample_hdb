import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import numpy as np

# Load data with optimized settings
@st.cache_data
def load_data():
    """Load and cache the HDB data with optimized settings"""
    df = pd.read_csv('datasets/train.csv', low_memory=False)
    # Optimize data types for better performance
    df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
    df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    # Convert categorical columns to category dtype for memory efficiency
    categorical_cols = ['town', 'flat_type', 'flat_model', 'street_name']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

data = load_data()

# Create performance metrics in sidebar
with st.sidebar:
    st.markdown("### 📊 Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{len(data):,}")
        st.metric("Towns", f"{data['town'].nunique()}")
    with col2:
        st.metric("Flat Types", f"{data['flat_type'].nunique()}")
        price_range_text = f"${data['resale_price'].min()/1000:.0f}K - ${data['resale_price'].max()/1000:.0f}K"
        st.metric("Price Range", price_range_text)

st.set_page_config(page_title="HDB Resale Flat Explorer", layout="wide")
st.title("HDB Resale Flat Interactive Visualization Tool")
st.markdown("Empowering new home buyers with data-driven insights.")

# --- Sidebar Filters ---
with st.sidebar:
    st.markdown("### 🔍 Filter Properties")
    
    # Create tabs for better organization
    filter_tab1, filter_tab2 = st.tabs(["📍 Basic Filters", "🏢 Amenities"])
    
    with filter_tab1:
        # Optimize data type conversions with error handling
        try:
            price_min, price_max = int(data['resale_price'].min()), int(data['resale_price'].max())
        except (ValueError, TypeError):
            numeric_prices = pd.to_numeric(data['resale_price'], errors='coerce')
            price_min, price_max = int(numeric_prices.min()), int(numeric_prices.max())

        # Format price slider with better UX
        price_range = st.slider(
            "💰 Resale Price Range", 
            price_min, price_max, 
            (price_min, price_max), 
            step=10000,
            format="$%d"
        )
        st.caption(f"Selected: ${price_range[0]:,} - ${price_range[1]:,}")

        # Optimize unique value extraction
        towns_unique = sorted(data['town'].dropna().unique())
        towns = st.multiselect(
            "🏘️ Town", 
            towns_unique, 
            default=towns_unique,
            help="Leave empty to select all towns"
        )

        flat_types_unique = sorted(data['flat_type'].dropna().unique())
        flat_types = st.multiselect(
            "🏠 Flat Type", 
            flat_types_unique, 
            default=flat_types_unique,
            help="Select one or more flat types"
        )

        try:
            area_min, area_max = int(data['floor_area_sqm'].min()), int(data['floor_area_sqm'].max())
        except (ValueError, TypeError):
            numeric_area = pd.to_numeric(data['floor_area_sqm'], errors='coerce')
            area_min, area_max = int(numeric_area.min()), int(numeric_area.max())

        area_range = st.slider("📐 Floor Area (sqm)", area_min, area_max, (area_min, area_max), step=5)
        st.caption(f"Selected: {area_range[0]} - {area_range[1]} sqm")

        # Lease Commencement Date filter
        if 'lease_commence_date' in data.columns and data['lease_commence_date'].notna().any():
            lease_min, lease_max = int(data['lease_commence_date'].min()), int(data['lease_commence_date'].max())
            lease_range = st.slider("📅 Lease Commence Date", lease_min, lease_max, (lease_min, lease_max), step=1)
        else:
            lease_range = None

        if 'hdb_age' in data.columns and data['hdb_age'].notna().any():
            age_min, age_max = int(data['hdb_age'].min()), int(data['hdb_age'].max())
            age_range = st.slider("🏗️ HDB Age (years)", age_min, age_max, (age_min, age_max), step=1)
        else:
            age_range = None
    
    with filter_tab2:
        # Amenities filters with better icons and organization
        st.markdown("**🚇 Transportation**")
        show_mrt = st.checkbox("Near MRT Station", value=False, help="Properties within walking distance to MRT")
        
        st.markdown("**🎓 Education**")
        show_schools = st.checkbox("Near Primary Schools", value=False, help="Properties near primary schools")
        
        st.markdown("**🛒 Shopping**")
        show_malls = st.checkbox("Near Shopping Malls", value=False, help="Properties near shopping malls")
        
        # Additional amenity filters based on available data
        if 'hawker_food_stalls' in data.columns:
            show_hawkers = st.checkbox("Near Hawker Centers", value=False, help="Properties near food courts")
        else:
            show_hawkers = False

# --- Filtered Data ---
filtered = data[
    (data['resale_price'] >= price_range[0]) &
    (data['resale_price'] <= price_range[1]) &
    (data['town'].isin(towns)) &
    (data['flat_type'].isin(flat_types)) &
    (data['floor_area_sqm'] >= area_range[0]) &
    (data['floor_area_sqm'] <= area_range[1])
]

# Apply lease/age filters
if lease_range is not None:
    filtered = filtered[(filtered['lease_commence_date'] >= lease_range[0]) & (filtered['lease_commence_date'] <= lease_range[1])]
if age_range is not None:
    filtered = filtered[(filtered['hdb_age'] >= age_range[0]) & (filtered['hdb_age'] <= age_range[1])]

# Apply amenities filters
if show_mrt and 'mrt_name' in data.columns:
    filtered = filtered[filtered['mrt_name'].notna()]
if show_schools and 'pri_sch_name' in data.columns:
    filtered = filtered[filtered['pri_sch_name'].notna()]
if show_malls and 'Mall_Within_1km' in data.columns:
    filtered = filtered[filtered['Mall_Within_1km'] > 0]
elif show_malls and 'mall_name' in data.columns:
    filtered = filtered[filtered['mall_name'].notna()]
if show_hawkers and 'hawker_food_stalls' in data.columns:
    filtered = filtered[filtered['hawker_food_stalls'] > 0]

# Performance optimization: Early exit if no data
if filtered.empty:
    st.error("🚫 No properties match your current filters. Please adjust your selection.")
    st.stop()

# Enhanced debug info with better formatting
with st.sidebar:
    st.markdown("### 📈 Filter Results")
    st.success(f"**{len(filtered):,}** properties found")
    if len(filtered) > 0:
        avg_filtered_price = filtered['resale_price'].mean()
        st.info(f"Average price: **${avg_filtered_price:,.0f}**")

# --- Enhanced Map Visualization ---
st.subheader("🗺️ Interactive Map of Available Flats")

# Create map controls
map_col1, map_col2, map_col3 = st.columns([2, 1, 1])
with map_col1:
    st.caption(f"Displaying {min(1000, len(filtered)):,} properties")
with map_col2:
    color_by = st.selectbox("Color by", ["Price", "Floor Area", "HDB Age"], index=0)
with map_col3:
    map_style = st.selectbox("Map Style", ["OpenStreetMap", "Stamen Terrain", "CartoDB positron"], index=0)

if not filtered.empty:
    # Sample data for performance, but ensure geographical distribution
    if len(filtered) > 1000:
        # Stratified sampling by town to maintain geographical representation
        map_data = filtered.groupby('town').apply(
            lambda x: x.sample(min(len(x), max(10, 1000 // filtered['town'].nunique())))
        ).reset_index(drop=True)
    else:
        map_data = filtered
    
    # Create base map with better styling
    center_lat, center_lon = map_data['Latitude'].mean(), map_data['Longitude'].mean()
    
    if map_style == "Stamen Terrain":
        tiles = "Stamen Terrain"
    elif map_style == "CartoDB positron":
        tiles = "CartoDB positron"
    else:
        tiles = "OpenStreetMap"
    
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=11,
        tiles=tiles
    )
    
    # Determine color scheme based on selection
    if color_by == "Price":
        color_values = map_data['resale_price']
        color_palette = px.colors.sequential.Viridis
    elif color_by == "Floor Area":
        color_values = map_data['floor_area_sqm']
        color_palette = px.colors.sequential.Blues
    else:  # HDB Age
        if 'hdb_age' in map_data.columns:
            color_values = map_data['hdb_age']
        else:
            color_values = map_data['resale_price']  # fallback
        color_palette = px.colors.sequential.Reds
    
    # Normalize color values
    color_min, color_max = color_values.min(), color_values.max()
    
    for _, row in map_data.iterrows():
        # Calculate color index safely
        if color_max > color_min:
            color_norm = (row[color_values.name] - color_min) / (color_max - color_min)
        else:
            color_norm = 0
        color_index = max(0, min(9, int(color_norm * 9)))
        
        # Enhanced popup with more information
        popup_html = f"""
        <div style="width: 200px;">
            <h4>${row['resale_price']:,}</h4>
            <b>{row['town']} - {row['flat_type']}</b><br>
            📍 {row.get('street_name', 'N/A')}<br>
            📐 {row['floor_area_sqm']} sqm<br>
            📅 Lease: {row.get('lease_commence_date', 'N/A')}<br>
            🚇 MRT: {row.get('mrt_name', 'N/A')}<br>
            🎓 School: {row.get('pri_sch_name', 'N/A')}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=max(4, min(15, int(row['floor_area_sqm']/8))),
            color=color_palette[color_index],
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"${row['resale_price']:,} | {row['town']}"
        ).add_to(m)
    
    if len(filtered) > len(map_data):
        st.info(f"📍 Showing {len(map_data):,} properties (sampled from {len(filtered):,} matches) for optimal performance")
    
    # Display map
    map_data_result = st_folium(m, width=900, height=500, returned_objects=["last_object_clicked"])
    
    # Show clicked property details
    if map_data_result['last_object_clicked']:
        clicked_lat = map_data_result['last_object_clicked']['lat']
        clicked_lng = map_data_result['last_object_clicked']['lng']
        
        # Find the closest property to the clicked point
        clicked_property = map_data.loc[
            ((map_data['Latitude'] - clicked_lat).abs() + 
             (map_data['Longitude'] - clicked_lng).abs()).idxmin()
        ]
        
        with st.expander("🏠 Selected Property Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price", f"${clicked_property['resale_price']:,}")
                st.write(f"**Town:** {clicked_property['town']}")
            with col2:
                st.metric("Floor Area", f"{clicked_property['floor_area_sqm']} sqm")
                st.write(f"**Type:** {clicked_property['flat_type']}")
            with col3:
                if 'hdb_age' in clicked_property:
                    st.metric("Age", f"{clicked_property['hdb_age']} years")
                st.write(f"**Address:** {clicked_property.get('street_name', 'N/A')}")
else:
    st.info("🗺️ No properties match the selected criteria.")

# --- Enhanced Search Results Table ---
st.subheader("🔍 Search Results")

# Enhanced search functionality
search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    search_term = st.text_input("🔎 Search by address, town, or flat type", "", 
                               placeholder="e.g. Bishan, 4 ROOM, Ang Mo Kio")
with search_col2:
    sort_by = st.selectbox("Sort by", ["Price (Low to High)", "Price (High to Low)", 
                                      "Floor Area", "HDB Age", "Town"])

if search_term:
    mask = (
        filtered['street_name'].str.contains(search_term, case=False, na=False) |
        filtered['flat_type'].str.contains(search_term, case=False, na=False) |
        filtered['town'].str.contains(search_term, case=False, na=False)
    )
    display_data = filtered[mask]
    st.caption(f"Found {len(display_data):,} properties matching '{search_term}'")
else:
    display_data = filtered.copy()

# Apply sorting
if sort_by == "Price (Low to High)":
    display_data = display_data.sort_values('resale_price')
elif sort_by == "Price (High to Low)":
    display_data = display_data.sort_values('resale_price', ascending=False)
elif sort_by == "Floor Area":
    display_data = display_data.sort_values('floor_area_sqm', ascending=False)
elif sort_by == "HDB Age" and 'hdb_age' in display_data.columns:
    display_data = display_data.sort_values('hdb_age')
elif sort_by == "Town":
    display_data = display_data.sort_values('town')

# Enhanced results display
if not display_data.empty:
    # Key columns with better formatting
    key_cols = ['resale_price', 'town', 'flat_type', 'floor_area_sqm', 'lease_commence_date', 'street_name', 'block']
    available_cols = [col for col in key_cols if col in display_data.columns]
    
    # Format the display data
    display_df = display_data[available_cols].head(100).copy()  # Increased to 100 rows
    
    # Format price column for better readability
    if 'resale_price' in display_df.columns:
        display_df['resale_price'] = display_df['resale_price'].apply(lambda x: f"${x:,.0f}")
    
    # Format area column
    if 'floor_area_sqm' in display_df.columns:
        display_df['floor_area_sqm'] = display_df['floor_area_sqm'].apply(lambda x: f"{x:.0f} sqm")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "resale_price": st.column_config.TextColumn("💰 Price"),
            "town": st.column_config.TextColumn("🏘️ Town"),
            "flat_type": st.column_config.TextColumn("🏠 Type"),
            "floor_area_sqm": st.column_config.TextColumn("📐 Area"),
            "lease_commence_date": st.column_config.NumberColumn("📅 Lease Year"),
            "street_name": st.column_config.TextColumn("📍 Street"),
            "block": st.column_config.TextColumn("🏢 Block")
        }
    )
    
    if len(display_data) > 100:
        st.info(f"📊 Showing first 100 of {len(display_data):,} matching properties")
    
    # Quick stats for current results
    if len(display_data) > 1:
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        with result_col1:
            st.metric("Avg Price", f"${display_data['resale_price'].mean():,.0f}")
        with result_col2:
            st.metric("Median Price", f"${display_data['resale_price'].median():,.0f}")
        with result_col3:
            st.metric("Avg Area", f"{display_data['floor_area_sqm'].mean():.0f} sqm")
        with result_col4:
            st.metric("Properties", f"{len(display_data):,}")
            
else:
    st.warning("🔍 No properties match your search term. Try different keywords.")

# --- Enhanced Market Insights ---
st.subheader("📊 Market Intelligence Dashboard")

# Key metrics with enhanced styling
insights_col1, insights_col2, insights_col3, insights_col4 = st.columns(4)

with insights_col1:
    avg_price_by_town = data.groupby('town')['resale_price'].mean().sort_values(ascending=False)
    most_expensive = avg_price_by_town.index[0]
    st.metric(
        "🏆 Most Expensive Town", 
        most_expensive, 
        f"${avg_price_by_town.iloc[0]:,.0f}",
        help="Town with highest average resale price"
    )

with insights_col2:
    most_common_flat = data['flat_type'].value_counts().index[0]
    flat_count = data['flat_type'].value_counts().iloc[0]
    st.metric(
        "🏠 Most Popular Type", 
        most_common_flat, 
        f"{flat_count:,} units",
        help="Most frequently sold flat type"
    )

with insights_col3:
    overall_avg = data['resale_price'].mean()
    st.metric(
        "💰 Market Average", 
        f"${overall_avg:,.0f}",
        help="Overall average resale price"
    )

with insights_col4:
    if 'hdb_age' in data.columns:
        avg_age = data['hdb_age'].mean()
        st.metric(
            "🏗️ Average Age", 
            f"{avg_age:.1f} years",
            help="Average age of HDB flats"
        )
    else:
        total_transactions = len(data)
        st.metric(
            "📈 Total Transactions", 
            f"{total_transactions:,}",
            help="Total number of transactions in dataset"
        )

# Enhanced visualizations
st.markdown("### 📈 Market Analysis Charts")

chart_tab1, chart_tab2, chart_tab3 = st.tabs(["🏘️ By Location", "🏠 By Type", "📅 Trends"])

with chart_tab1:
    # Top 15 most expensive towns
    fig_avg_town = px.bar(
        x=avg_price_by_town.head(15).values, 
        y=avg_price_by_town.head(15).index,
        orientation='h',
        title="Top 15 Most Expensive Towns (Average Resale Price)",
        labels={'x': 'Average Resale Price ($)', 'y': 'Town'},
        color=avg_price_by_town.head(15).values,
        color_continuous_scale='Viridis'
    )
    fig_avg_town.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_avg_town, use_container_width=True)

with chart_tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Flat type distribution
        flat_counts = data['flat_type'].value_counts()
        fig_flat_dist = px.pie(
            values=flat_counts.values, 
            names=flat_counts.index,
            title="Distribution of Flat Types",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_flat_dist, use_container_width=True)
    
    with col2:
        # Average price by flat type
        avg_price_by_type = data.groupby('flat_type')['resale_price'].mean().sort_values()
        fig_type_price = px.bar(
            x=avg_price_by_type.values,
            y=avg_price_by_type.index,
            orientation='h',
            title="Average Price by Flat Type",
            labels={'x': 'Average Price ($)', 'y': 'Flat Type'},
            color=avg_price_by_type.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_type_price, use_container_width=True)

with chart_tab3:
    if 'Tranc_YearMonth' in data.columns:
        # Convert to datetime for better plotting
        data_with_date = data.copy()
        data_with_date['date'] = pd.to_datetime(data_with_date['Tranc_YearMonth'], format='%Y-%m')
        
        # Monthly trend
        monthly_trend = data_with_date.groupby('date')['resale_price'].agg(['mean', 'median', 'count']).reset_index()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_trend['date'], 
            y=monthly_trend['median'],
            mode='lines+markers',
            name='Median Price',
            line=dict(color='blue', width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_trend['date'], 
            y=monthly_trend['mean'],
            mode='lines',
            name='Average Price',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_trend.update_layout(
            title="HDB Resale Price Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Transaction volume
        fig_volume = px.bar(
            monthly_trend, 
            x='date', 
            y='count',
            title="Transaction Volume Over Time",
            labels={'count': 'Number of Transactions', 'date': 'Date'}
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    else:
        st.info("⏰ Time trend data not available in dataset")

# --- Price Trends ---
st.subheader("Price Trends Analysis")
if not filtered.empty and 'Tranc_YearMonth' in filtered.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall trend
        trend = filtered.groupby('Tranc_YearMonth')['resale_price'].median().reset_index()
        if not trend.empty:
            fig_trend = px.line(trend, x='Tranc_YearMonth', y='resale_price', 
                              title='Median Resale Price Over Time')
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Trend by selected town (if user filtered to specific towns)
        if len(towns) <= 5:  # Only show if few towns selected
            town_trend = filtered.groupby(['Tranc_YearMonth', 'town'])['resale_price'].median().reset_index()
            if not town_trend.empty:
                fig_town_trend = px.line(town_trend, x='Tranc_YearMonth', y='resale_price', 
                                       color='town', title='Price Trends by Selected Towns')
                st.plotly_chart(fig_town_trend, use_container_width=True)
        else:
            st.info("Select fewer towns to see town-specific trends")
            
elif 'Tranc_YearMonth' not in data.columns:
    st.info("Time trend data not available in dataset")

# --- Price by Location ---
st.subheader("Median Resale Price by Town")
if not filtered.empty:
    by_town = filtered.groupby('town')['resale_price'].median().reset_index()
    if not by_town.empty:
        fig_town = px.bar(by_town, x='town', y='resale_price', title='Median Resale Price by Town')
        fig_town.update_xaxes(tickangle=45)
        st.plotly_chart(fig_town, use_container_width=True)
    else:
        st.info("No location data available for current selection")

# --- Flexible Correlation Plots ---
st.subheader("Explore Correlations")
numeric_cols = filtered.select_dtypes(include='number').columns.tolist()
if len(numeric_cols) >= 2:
    x_axis = st.selectbox("X Axis", numeric_cols, index=numeric_cols.index('floor_area_sqm') if 'floor_area_sqm' in numeric_cols else 0)
    y_axis = st.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('resale_price') if 'resale_price' in numeric_cols else 1)
    fig_corr = px.scatter(filtered, x=x_axis, y=y_axis, color='town', title=f'{y_axis} vs. {x_axis}')
    st.plotly_chart(fig_corr, use_container_width=True)

# --- Property Comparison Tool ---
st.subheader("Compare Properties")
if not filtered.empty:
    compare_ids = st.multiselect(
        "Select up to 3 properties to compare", 
        filtered.index, 
        max_selections=3,
        format_func=lambda i: f"{filtered.loc[i, 'town']} | {filtered.loc[i, 'block']} | ${filtered.loc[i, 'resale_price']:,}"
    )
    if compare_ids:
        if len(compare_ids) >= 2:
            compare_cols = ['town','block','street_name','flat_type','floor_area_sqm','resale_price','lease_commence_date','hdb_age','mrt_name','pri_sch_name']
            available_compare_cols = [col for col in compare_cols if col in filtered.columns]

            comparison_df = filtered.loc[compare_ids][available_compare_cols].T
            comparison_df.columns = [f"Property {i+1}" for i in range(len(compare_ids))]
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("Select at least two properties to compare")
    else:
        st.info("Select properties from the map or search results above to compare")
else:
    st.info("No properties available for comparison. Adjust your filters.")

# --- Preference-Based Recommender ---
st.subheader("Find Your Ideal Flat")
with st.form("recommender"):
    col1, col2, col3 = st.columns(3)
    with col1:
        pref_town = st.selectbox("Preferred Town", ["Any"] + sorted(data['town'].unique()))
    with col2:
        pref_type = st.selectbox("Preferred Flat Type", ["Any"] + sorted(data['flat_type'].unique()))
    with col3:
        pref_budget = st.number_input("Maximum Budget", min_value=price_min, max_value=price_max, value=price_max)
    
    # Additional preferences
    pref_min_area = st.slider("Minimum Floor Area (sqm)", area_min, area_max, area_min)
    near_mrt = st.checkbox("Must be near MRT")
    near_school = st.checkbox("Must be near primary school")
    
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    # Build recommendation query
    recs = data.copy()
    
    if pref_town != "Any":
        recs = recs[recs['town'] == pref_town]
    if pref_type != "Any":
        recs = recs[recs['flat_type'] == pref_type]
    
    recs = recs[
        (recs['resale_price'] <= pref_budget) &
        (recs['floor_area_sqm'] >= pref_min_area)
    ]
    
    if near_mrt and 'mrt_name' in recs.columns:
        recs = recs[recs['mrt_name'].notna()]
    if near_school and 'pri_sch_name' in recs.columns:
        recs = recs[recs['pri_sch_name'].notna()]
    
    if not recs.empty:
        # Sort by price (best value first)
        recs = recs.sort_values('resale_price')
        
        rec_cols = ['town','block','street_name','flat_type','floor_area_sqm','resale_price','lease_commence_date','hdb_age','mrt_name','pri_sch_name']
        available_rec_cols = [col for col in rec_cols if col in recs.columns]
        
        st.write(f"🎯 Found {len(recs)} matching properties! Here are the top 10:")
        st.dataframe(recs.head(10)[available_rec_cols], use_container_width=True)
    else:
        st.warning("No properties match your preferences. Try adjusting your criteria.")

# --- User Feedback Section ---
st.markdown("---")
st.subheader("💬 Help Us Improve!")

feedback_col1, feedback_col2 = st.columns([2, 1])

with feedback_col1:
    st.markdown("Your feedback helps us make this tool better for future home buyers!")
    
    # Rating system
    rating = st.selectbox(
        "How would you rate this tool?", 
        ["⭐⭐⭐⭐⭐ Excellent", "⭐⭐⭐⭐ Good", "⭐⭐⭐ Average", "⭐⭐ Poor", "⭐ Very Poor"],
        index=0
    )
    
    # Feedback categories
    feedback_type = st.multiselect(
        "What aspects were most helpful?",
        ["🗺️ Interactive Map", "🔍 Search & Filters", "📊 Market Insights", 
         "📈 Price Trends", "🏠 Property Comparison", "🎯 Recommendations"],
        default=["🗺️ Interactive Map", "📊 Market Insights"]
    )
    
    feedback_text = st.text_area(
        "Additional comments or suggestions:", 
        placeholder="Share your experience or suggest improvements...",
        height=100
    )
    
    if st.button("📤 Submit Feedback", type="primary"):
        st.success("🙏 Thank you for your feedback! It helps us improve the tool.")
        st.balloons()

with feedback_col2:
    st.markdown("### 🎯 Usage Tips")
    st.markdown("""
    💡 **Quick Tips:**
    - Use filters to narrow down options
    - Click map markers for details
    - Compare up to 3 properties
    - Try the recommendation tool
    - Sort results by different criteria
    """)
    
    st.markdown("### 📞 Need Help?")
    st.markdown("""
    📧 Contact: support@hdbexplorer.sg  
    📱 Hotline: 1800-HDB-HELP  
    🌐 Website: www.hdbexplorer.sg
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <h4>🏠 HDB Resale Flat Explorer</h4>
    <p><i>Empowering informed decisions for new home buyers</i></p>
    <p>📊 Data source: HDB resale transactions | 🔄 Last updated: August 2025</p>
    <p>Built with ❤️ using Streamlit, Plotly, and Folium</p>
</div>
""", unsafe_allow_html=True)