import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient

# MongoDB connection setup
@st.cache_resource
def get_mongodb_client():
    """Create a MongoDB client connection"""
    connection_string = "mongodb+srv://talalmos:97VsovNUWwy7YuXe@shortsqueeze.rymraqj.mongodb.net/?etryWrites=true&w=majority&appName=ShortSqueeze"
    client = MongoClient(connection_string)
    return client

# Initialize MongoDB connection
try:
    client = get_mongodb_client()
    db = client["algo_trading_db"]
    daily_data_collection = db["daily_data"]
    trades_collection = db["trades"]
    pre_trade_stats_collection = db["pre_trade_stats"] 
    pre_trade_trends_collection = db["pre_trade_trends"]
    st.sidebar.success("✅ Connected to MongoDB")
except Exception as e:
    st.sidebar.error(f"Failed to connect to MongoDB: {str(e)}")
    st.stop()

# Set page title
st.title("Patterns Visualization")

# Function to load patterns report from an uploaded file
def load_patterns_report(uploaded_file):
    if uploaded_file is None:
        return None
        
    try:
        # Load the patterns report from the uploaded file
        report_df = pd.read_csv(uploaded_file)
        
        # Check if report has any rows
        if report_df.empty:
            st.warning("Patterns report file is empty")
            return None
            
        return report_df
        
    except Exception as e:
        st.error(f"Error loading patterns report: {str(e)}")
        return None

# Function to load and process data from MongoDB
def load_data(selected_date):
    try:
        # Format the date to match MongoDB format
        date_str = selected_date.strftime("%Y-%m-%d")
        
        # Query MongoDB for data on this date
        cursor = daily_data_collection.find({"trading_date": date_str})
        data_list = list(cursor)
        
        if not data_list:
            st.error(f"No data found for {date_str} in MongoDB")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Remove MongoDB _id field
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
            
        st.info(f"Loaded {len(df)} records for {date_str} from MongoDB")
        return df
    
    except Exception as e:
        st.error(f"Error loading data from MongoDB: {str(e)}")
        return None

# Get list of available dates from MongoDB
def get_available_dates():
    try:
        # Get distinct trading dates from daily data collection
        dates = daily_data_collection.distinct("trading_date")
        
        if not dates:
            st.warning("No trading dates found in MongoDB")
            return []
            
        # Convert string dates to datetime objects
        date_objects = []
        for date_str in dates:
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                date_objects.append(date_obj)
            except ValueError:
                continue
        
        return sorted(date_objects, reverse=True)
    
    except Exception as e:
        st.error(f"Error getting available dates from MongoDB: {str(e)}")
        return []

# Add a file uploader in the sidebar for patterns report
st.sidebar.header("Upload Data")
uploaded_patterns_file = st.sidebar.file_uploader(
    "Upload patterns report (CSV file):",
    type=["csv"],
    help="Upload the all_patterns_report.csv file"
)

# Load patterns report from uploaded file
patterns_data = load_patterns_report(uploaded_patterns_file)

# Display status of patterns data in sidebar
if patterns_data is not None:
    st.sidebar.success(f"✅ Patterns report loaded: {uploaded_patterns_file.name}")
    st.sidebar.text(f"Contains {len(patterns_data)} patterns")
else:
    st.sidebar.warning("⚠️ No patterns report uploaded")

# Get available dates from MongoDB
available_dates = get_available_dates()
if available_dates:
    st.sidebar.success(f"✅ Found {len(available_dates)} trading days in MongoDB")
else:
    st.sidebar.warning("⚠️ No trading days found in MongoDB")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Daily Data", "Patterns Report", "Connected Data", "All Data View", "Trades"])

with tab1:
    st.header("Daily Data")
    
    if not available_dates:
        st.warning("No data found in MongoDB. Please check your connection.")
    else:
        # Check for query parameters to maintain state between reruns
        default_date_index = 0

        # If date_index is in query params, use it as the default
        if "date_index" in st.query_params:
            try:
                default_date_index = int(st.query_params["date_index"])
                if default_date_index < 0 or default_date_index >= len(available_dates):
                    default_date_index = 0
            except (ValueError, IndexError):
                default_date_index = 0

        # Date selection
        selected_date = st.selectbox(
            "Select a date to view data:",
            available_dates,
            index=default_date_index,
            format_func=lambda x: x.strftime("%Y-%m-%d"),
            key="date_selector_tab1"
        )
        
        # Load data for the selected date
        data = load_data(selected_date)
        
        if data is not None:
            # Show data preview
            st.subheader("Data Preview")
            
            # Filter rows where SPX_Px_Last has no value
            if 'SPX_Px_Last' in data.columns:
                filtered_data = data.dropna(subset=['SPX_Px_Last'])
                if len(filtered_data) < len(data):
                    st.info(f"Filtered out {len(data) - len(filtered_data)} rows where SPX_Px_Last has no value.")
                data = filtered_data  # Replace the original data with the filtered one
            
            # Check the data type of ExpectedDividands
            if 'ExpectedDividands' in data.columns:
                dividends_dtype = data['ExpectedDividands'].dtype
                st.info(f"Data type of ExpectedDividands column: {dividends_dtype}")
            
            # Make a copy to avoid modifying the original dataframe
            preview_data = data.head().copy()
            
            # Remove Fut_StockPx_Last column if it exists
            if 'Fut_StockPx_Last' in preview_data.columns:
                preview_data = preview_data.drop(columns=['Fut_StockPx_Last'])
            
            # Format ID column if it exists
            if 'ID' in preview_data.columns:
                preview_data['ID'] = preview_data['ID'].astype(str).str.replace(',', '')
            
            # Display the preview without row numbers
            st.dataframe(preview_data.reset_index(drop=True), hide_index=True)
            
            # Find patterns for this date if patterns data is available
            filtered_patterns = None
            if patterns_data is not None and 'ID' in data.columns:
                # Format IDs for matching
                data_ids = data['ID'].astype(str).str.replace(',', '').tolist()
                
                # Find patterns that match this day's data
                patterns_data['start_idx'] = patterns_data['start_idx'].astype(str)
                matching_patterns = patterns_data[patterns_data['start_idx'].isin(data_ids)]
                
                if not matching_patterns.empty:
                    st.subheader(f"Patterns for {selected_date.strftime('%Y-%m-%d')}")
                    st.write(f"Found {len(matching_patterns)} patterns on this day")
                    
                    # Add filters in an expander
                    with st.expander("Filter Patterns", expanded=False):
                        st.write("Set filters to find specific patterns:")
                        
                        # Create columns for filters
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Filter by deviation strength
                            min_deviation = float(matching_patterns['avg_deviation'].min())
                            max_deviation = float(matching_patterns['avg_deviation'].max())
                            deviation_range = st.slider(
                                "Average Deviation Range:",
                                min_value=min_deviation,
                                max_value=max_deviation,
                                value=(min_deviation, max_deviation),
                                format="%.2f",
                                key="dev_range_tab1"
                            )
                            
                            # Filter by price impact
                            min_impact = float(matching_patterns['price_impact'].min())
                            max_impact = float(matching_patterns['price_impact'].max())
                            impact_range = st.slider(
                                "Price Impact Range:",
                                min_value=min_impact,
                                max_value=max_impact,
                                value=(min_impact, max_impact),
                                format="%.2f",
                                key="impact_range_tab1"
                            )
                        
                        with col2:
                            # Filter by pattern duration
                            min_duration = int(matching_patterns['duration'].min())
                            max_duration = int(matching_patterns['duration'].max())
                            duration_range = st.slider(
                                "Pattern Duration (length):",
                                min_value=min_duration,
                                max_value=max_duration,
                                value=(min_duration, max_duration),
                                key="duration_range_tab1"
                            )
                            
                            # Boolean filters using multiselect
                            bool_filters = []
                            
                            if 'direction_alignment' in matching_patterns.columns:
                                direction_options = st.multiselect(
                                    "Direction Alignment:",
                                    options=["Aligned with trend", "Against trend"],
                                    default=[],
                                    key="dir_align_tab1"
                                )
                                if "Aligned with trend" in direction_options:
                                    bool_filters.append(('direction_alignment', True))
                                if "Against trend" in direction_options:
                                    bool_filters.append(('direction_alignment', False))
                    
                    # Apply filters to patterns data
                    filtered_patterns = matching_patterns.copy()
                    
                    # Apply range filters
                    filtered_patterns = filtered_patterns[
                        (filtered_patterns['avg_deviation'] >= deviation_range[0]) &
                        (filtered_patterns['avg_deviation'] <= deviation_range[1]) &
                        (filtered_patterns['price_impact'] >= impact_range[0]) &
                        (filtered_patterns['price_impact'] <= impact_range[1]) &
                        (filtered_patterns['duration'] >= duration_range[0]) &
                        (filtered_patterns['duration'] <= duration_range[1])
                    ]
                    
                    # Apply boolean filters
                    for column, value in bool_filters:
                        if column in filtered_patterns.columns:
                            filtered_patterns = filtered_patterns[filtered_patterns[column] == value]
                    
                    # Display patterns data after filtering
                    st.write(f"Showing {len(filtered_patterns)} patterns after filtering")

                    # Create a more user-friendly pattern selection UI
                    st.subheader("Select Patterns to Add to Plot")

                    # Create selection options
                    selection_method = st.radio(
                        "Pattern selection method:",
                        ["Select all", "Select none", "Select by criteria", "Select individually"],
                        horizontal=True,
                        key="selection_method_tab1"
                    )

                    # Initialize selected_patterns as empty list
                    selected_patterns = []

                    if selection_method == "Select all":
                        selected_patterns = filtered_patterns.to_dict('records')
                        st.success(f"Selected all {len(filtered_patterns)} patterns")
                        
                    elif selection_method == "Select by criteria":
                        st.write("Choose criteria to select patterns:")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Select by minimum deviation
                            min_dev_threshold = st.slider(
                                "Minimum deviation threshold:",
                                min_value=float(filtered_patterns['avg_deviation'].min()),
                                max_value=float(filtered_patterns['avg_deviation'].max()),
                                value=float(filtered_patterns['avg_deviation'].median()),
                                format="%.2f",
                                key="min_dev_threshold_tab1"
                            )
                            
                            # Select by minimum impact
                            min_impact_threshold = st.slider(
                                "Minimum price impact:",
                                min_value=float(filtered_patterns['price_impact'].min()),
                                max_value=float(filtered_patterns['price_impact'].max()),
                                value=float(filtered_patterns['price_impact'].median()),
                                format="%.2f",
                                key="min_impact_threshold_tab1"
                            )
                        
                        with col2:
                            # Select by max duration
                            max_duration = st.slider(
                                "Maximum pattern duration:",
                                min_value=int(filtered_patterns['duration'].min()),
                                max_value=int(filtered_patterns['duration'].max()),
                                value=int(filtered_patterns['duration'].max()),
                                key="max_duration_tab1"
                            )
                            
                            # Select top N patterns
                            top_n = st.number_input(
                                "Select top N patterns by impact:",
                                min_value=1,
                                max_value=len(filtered_patterns),
                                value=min(5, len(filtered_patterns)),
                                key="top_n_tab1"
                            )
                        
                        # Apply criteria
                        criteria_patterns = filtered_patterns[
                            (filtered_patterns['avg_deviation'] >= min_dev_threshold) &
                            (filtered_patterns['price_impact'] >= min_impact_threshold) &
                            (filtered_patterns['duration'] <= max_duration)
                        ].copy()
                        
                        # Sort by price impact and select top N
                        criteria_patterns = criteria_patterns.sort_values(by='price_impact', ascending=False).head(top_n)
                        
                        if not criteria_patterns.empty:
                            selected_patterns = criteria_patterns.to_dict('records')
                            st.success(f"Selected {len(selected_patterns)} patterns based on criteria")
                        else:
                            st.warning("No patterns match the selected criteria")

                    elif selection_method == "Select individually":
                        # Create a DataFrame with selection checkboxes
                        selection_df = filtered_patterns.copy()
                        
                        # Add a formatted info column to make selection easier
                        selection_df['Pattern Info'] = selection_df.apply(
                            lambda row: f"Pattern {row['pattern_id']} | Impact: {row['price_impact']:.2f} | Duration: {row['duration']}",
                            axis=1
                        )
                        
                        # Display in a container with fixed height for scrolling
                        with st.container(height=300):
                            for idx, row in selection_df.iterrows():
                                if st.checkbox(row['Pattern Info'], key=f"pattern_select_{idx}"):
                                    selected_patterns.append(row.to_dict())
                        
                        st.info(f"Selected {len(selected_patterns)} patterns individually")

                    # Show a preview of selected patterns
                    if selected_patterns:
                        st.subheader("Selected Patterns Preview")
                        selected_df = pd.DataFrame(selected_patterns)
                        
                        # Format display columns
                        display_cols = ['pattern_id', 'start_idx', 'duration', 'price_impact', 'avg_deviation']
                        display_cols = [col for col in display_cols if col in selected_df.columns]
                        
                        st.dataframe(
                            selected_df[display_cols].style.format({
                                'price_impact': '{:.2f}',
                                'avg_deviation': '{:.2f}'
                            }),
                            height=150
                        )
                    else:
                        st.warning("No patterns selected. The plot will show only price data without patterns.")
                else:
                    st.warning("No matching patterns found for this date's data")
            
            # Add button to create plot with selected patterns
            plot_button = st.button("Create Plot with Selected Patterns")
            
            if plot_button:
                # Remove debug messages
                
                # Check if we have both TimeStamp and price data
                required_columns = ['TimeStamp', 'SPX_Px_Last', 'Cap_Contract']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if not missing_columns:
                    try:
                        # Ensure TimeStamp is in datetime format
                        if not pd.api.types.is_datetime64_any_dtype(data['TimeStamp']):
                            data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
                        
                        # Get trading date as string
                        selected_date_str = selected_date.strftime("%Y-%m-%d")
                        
                        # Set default time range (market hours)
                        default_start = pd.Timestamp(f"{selected_date_str} 09:30:00")
                        default_end = pd.Timestamp(f"{selected_date_str} 16:00:00")
                        
                        # Create the Plotly figure
                        fig = go.Figure()
                        
                        # Add Cap Contract trace (first so it's behind)
                        fig.add_trace(go.Scatter(
                            x=data['TimeStamp'],
                            y=data['Cap_Contract'],
                            name='Cap Contract',
                            line=dict(color='green', width=1.5),
                            hovertemplate='Time: %{x|%H:%M}<br>Cap Contract: %{y:.2f}<extra></extra>'
                        ))
                        
                        # Add SPX_Px_Last trace (second so it's in front)
                        fig.add_trace(go.Scatter(
                            x=data['TimeStamp'],
                            y=data['SPX_Px_Last'],
                            name='SPX Price Last',
                            line=dict(color='blue', width=2),
                            hovertemplate='Time: %{x|%H:%M}<br>SPX Price: %{y:.2f}<extra></extra>'
                        ))
                        
                        # Add selected patterns to the plot if we have any
                        if 'selected_patterns' in locals() and len(selected_patterns) > 0:
                            # Format IDs for matching
                            data_ids = data['ID'].astype(str).str.replace(',', '').tolist()
                            
                            # For each selected pattern, highlight its duration on the chart
                            for pattern in selected_patterns:
                                # Get pattern details
                                start_idx = str(pattern['start_idx'])
                                duration = int(pattern['duration'])
                                pattern_id = pattern['pattern_id']
                                
                                # Find where this ID is in the data
                                if start_idx in data_ids:
                                    start_position = data_ids.index(start_idx)
                                    
                                    # Determine end position (ensure it doesn't go beyond data length)
                                    end_position = min(start_position + duration, len(data) - 1)
                                    
                                    # Get timestamps for start and end
                                    if start_position < len(data) and end_position < len(data):
                                        start_time = data['TimeStamp'].iloc[start_position]
                                        end_time = data['TimeStamp'].iloc[end_position]
                                        
                                        # Get price values for annotation positioning
                                        price_at_start = data['SPX_Px_Last'].iloc[start_position]
                                        
                                        # Add a semi-transparent rectangle to highlight the pattern duration
                                        fig.add_shape(
                                            type="rect",
                                            x0=start_time,
                                            x1=end_time,
                                            y0=data['SPX_Px_Last'].min() * 0.99,
                                            y1=data['SPX_Px_Last'].max() * 1.01,
                                            fillcolor="rgba(255, 165, 0, 0.2)",
                                            line=dict(width=0),
                                            layer="below"
                                        )
                                        
                                        # Add an annotation to label the pattern
                                        fig.add_annotation(
                                            x=start_time,
                                            y=price_at_start,
                                            text=f"Pattern {pattern_id}",
                                            showarrow=True,
                                            arrowhead=2,
                                            arrowsize=1,
                                            arrowwidth=2,
                                            arrowcolor="#FF8C00",
                                            font=dict(size=10, color="#FF8C00"),
                                            align="center",
                                            ax=0,
                                            ay=-40
                                        )
                        
                        # Update layout with time range and formatting
                        fig.update_layout(
                            title=f'SPX Price Last and Cap Contract ({selected_date_str})',
                            xaxis=dict(
                                title='Time (HH:MM)',
                                range=[default_start, default_end],
                                tickformat='%H:%M',
                                gridcolor='lightgray',
                            ),
                            yaxis=dict(
                                title='Value',
                                gridcolor='lightgray',
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            hovermode='x unified',
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=500,
                        )
                        
                        # Show the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add note about zooming functionality
                        st.info("💡 Zoom Tips: You can zoom by selecting an area on the plot and double-click to reset the view.")

                        # Display detailed pattern information below the plot
                        if 'selected_patterns' in locals() and len(selected_patterns) > 0:
                            st.subheader("Selected Pattern Details")
                            
                            # Create tabs for different ways to view pattern information
                            pattern_tabs = st.tabs(["Table View", "Detailed View"])
                            
                            with pattern_tabs[0]:
                                # Create a clean dataframe for display
                                pattern_details_df = pd.DataFrame(selected_patterns)
                                
                                # Define important columns to show first
                                primary_cols = ['pattern_id', 'start_idx', 'duration', 'price_impact', 'avg_deviation']
                                
                                # Add any additional informative columns from the patterns report
                                informative_cols = [
                                    'end_idx', 'pattern_type', 'pattern_direction', 'magnitude', 
                                    'confidence_score', 'pattern_effectiveness'
                                ]
                                
                                # Filter to include only columns that exist in the DataFrame
                                primary_cols = [col for col in primary_cols if col in pattern_details_df.columns]
                                informative_cols = [col for col in informative_cols if col in pattern_details_df.columns]
                                
                                # Get remaining columns excluding what we've already included
                                all_cols = list(pattern_details_df.columns)
                                remaining_cols = [col for col in all_cols if col not in (primary_cols + informative_cols)]
                                
                                # Reorder columns: primary cols first, then informative, then others
                                ordered_cols = primary_cols + informative_cols + remaining_cols
                                
                                # Format the dataframe
                                formatted_df = pattern_details_df[ordered_cols].copy()
                                
                                # Apply formatting to numeric columns
                                numeric_cols = formatted_df.select_dtypes(include=['float']).columns
                                for col in numeric_cols:
                                    formatted_df[col] = formatted_df[col].round(2)
                                
                                # Display as a sortable table
                                st.dataframe(formatted_df, use_container_width=True)
                            
                            with pattern_tabs[1]:
                                # Create a list of all patterns with their IDs for a selectbox
                                pattern_options = [f"Pattern {p.get('pattern_id', i)}" for i, p in enumerate(selected_patterns)]
                                
                                # Add a selectbox to choose which pattern to view details for
                                selected_pattern_idx = st.selectbox(
                                    "Select a pattern to view details:",
                                    range(len(pattern_options)),
                                    format_func=lambda i: pattern_options[i]
                                )
                                
                                # Get the selected pattern
                                pattern = selected_patterns[selected_pattern_idx]
                                pattern_id = pattern.get('pattern_id', f"Pattern {selected_pattern_idx+1}")
                                
                                # Create two columns for the details
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Basic pattern metrics
                                    st.markdown("**Basic Metrics**")
                                    metrics = {
                                        "Start Index": pattern.get('start_idx', 'N/A'),
                                        "Duration": pattern.get('duration', 'N/A'),
                                        "Price Impact": f"{pattern.get('price_impact', 'N/A'):.2f}",
                                        "Average Deviation": f"{pattern.get('avg_deviation', 'N/A'):.2f}"
                                    }
                                    
                                    for key, value in metrics.items():
                                        st.markdown(f"**{key}:** {value}")
                                
                                with col2:
                                    # Additional pattern characteristics
                                    st.markdown("**Pattern Characteristics**")
                                    characteristics = {}
                                    
                                    # Add any available pattern characteristics from the report
                                    for char_key in ['pattern_type', 'pattern_direction', 'pattern_effectiveness', 
                                                  'confidence_score', 'magnitude']:
                                        if char_key in pattern:
                                            formatted_value = pattern[char_key]
                                            if isinstance(formatted_value, float):
                                                formatted_value = f"{formatted_value:.2f}"
                                            characteristics[char_key.replace('_', ' ').title()] = formatted_value
                                    
                                    for key, value in characteristics.items():
                                        st.markdown(f"**{key}:** {value}")
                                
                                # Display all pattern attributes in a table (instead of nested expander)
                                st.markdown("**All Pattern Data**")
                                pattern_df = pd.DataFrame({
                                    'Attribute': list(pattern.keys()),
                                    'Value': [str(val) for val in pattern.values()]
                                })
                                st.dataframe(pattern_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error creating plot: {str(e)}")
                else:
                    st.error(f"Missing required columns for plotting: {', '.join(missing_columns)}")
            
            # Add an option to download the data
            st.download_button(
                label="Download data as CSV",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name=f"trading_data_{selected_date.strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )

with tab2:
    st.header("Patterns Report")
    
    if patterns_data is not None:
        # Display data info
        st.write(f"Total patterns found: {len(patterns_data)}")
        
        # Add filters in an expander
        with st.expander("Filter Patterns", expanded=False):
            st.write("Set filters to find specific patterns:")
            
            # Create columns for filters
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter by deviation strength
                min_deviation = float(patterns_data['avg_deviation'].min())
                max_deviation = float(patterns_data['avg_deviation'].max())
                deviation_range = st.slider(
                    "Average Deviation Range:",
                    min_value=min_deviation,
                    max_value=max_deviation,
                    value=(min_deviation, max_deviation),
                    format="%.2f"
                )
                
                # Filter by price impact
                min_impact = float(patterns_data['price_impact'].min())
                max_impact = float(patterns_data['price_impact'].max())
                impact_range = st.slider(
                    "Price Impact Range:",
                    min_value=min_impact,
                    max_value=max_impact,
                    value=(min_impact, max_impact),
                    format="%.2f"
                )
                
                # Filter by pattern duration
                min_duration = int(patterns_data['duration'].min())
                max_duration = int(patterns_data['duration'].max())
                duration_range = st.slider(
                    "Pattern Duration (length):",
                    min_value=min_duration,
                    max_value=max_duration,
                    value=(min_duration, max_duration)
                )
                
            with col2:
                # Filter by effectiveness
                if 'pattern_effectiveness' in patterns_data.columns:
                    min_effectiveness = float(patterns_data['pattern_effectiveness'].min())
                    max_effectiveness = float(patterns_data['pattern_effectiveness'].max())
                    effectiveness_range = st.slider(
                        "Pattern Effectiveness:",
                        min_value=min_effectiveness,
                        max_value=max_effectiveness,
                        value=(min_effectiveness, max_effectiveness),
                        format="%.2f"
                    )
                
                # Boolean filters using multiselect
                bool_filters = []
                
                if 'direction_alignment' in patterns_data.columns:
                    direction_options = st.multiselect(
                        "Direction Alignment:",
                        options=["Aligned with trend", "Against trend"],
                        default=[]
                    )
                    if "Aligned with trend" in direction_options:
                        bool_filters.append(('direction_alignment', True))
                    if "Against trend" in direction_options:
                        bool_filters.append(('direction_alignment', False))
        
        # Apply filters to patterns data
        filtered_patterns = patterns_data.copy()
        
        # Apply range filters
        filtered_patterns = filtered_patterns[
            (filtered_patterns['avg_deviation'] >= deviation_range[0]) &
            (filtered_patterns['avg_deviation'] <= deviation_range[1]) &
            (filtered_patterns['price_impact'] >= impact_range[0]) &
            (filtered_patterns['price_impact'] <= impact_range[1]) &
            (filtered_patterns['duration'] >= duration_range[0]) &
            (filtered_patterns['duration'] <= duration_range[1])
        ]
        
        # Apply effectiveness filter if column exists
        if 'pattern_effectiveness' in patterns_data.columns:
            filtered_patterns = filtered_patterns[
                (filtered_patterns['pattern_effectiveness'] >= effectiveness_range[0]) &
                (filtered_patterns['pattern_effectiveness'] <= effectiveness_range[1])
            ]
        
        # Apply boolean filters
        for column, value in bool_filters:
            if column in filtered_patterns.columns:
                filtered_patterns = filtered_patterns[filtered_patterns[column] == value]
        
        # Format pattern_id to match the daily data ID format
        filtered_patterns['pattern_id'] = filtered_patterns['pattern_id'].astype(str).str.replace(',', '')
        
        # Display patterns data count after filtering
        st.write(f"Showing {len(filtered_patterns)} patterns after filtering")
        
        # Display filtered patterns data
        st.subheader("Patterns Data")
        st.dataframe(filtered_patterns.reset_index(drop=True), hide_index=True)
        
        # Add download button for filtered patterns data
        st.download_button(
            label="Download filtered patterns as CSV",
            data=filtered_patterns.to_csv(index=False).encode('utf-8'),
            file_name="filtered_patterns_report.csv",
            mime='text/csv',
        )
    else:
        st.info("Please upload a patterns report file using the uploader in the sidebar.")

with tab3:
    st.header("Connected Data")
    
    if not available_dates:
        st.warning("No daily data files found.")
    elif patterns_data is None:
        st.info("Please upload a patterns report file using the uploader in the sidebar.")
    else:
        # Allow selecting a date
        selected_date = st.selectbox(
            "Select a date to view data:",
            available_dates,
            format_func=lambda x: x.strftime("%Y-%m-%d"),
            key="date_selector_tab3"
        )
        
        # Load data for the selected date
        data = load_data(selected_date)
        
        if data is not None:
            # Ensure ID is properly formatted in both datasets
            if 'ID' in data.columns:
                data['ID'] = data['ID'].astype(str).str.replace(',', '')
                
                # Convert TimeStamp if it exists
                if 'TimeStamp' in data.columns:
                    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
                    data['TimeFormatted'] = data['TimeStamp'].dt.strftime('%H:%M')
                
                # Convert start_idx to string format in patterns data for matching
                patterns_data['start_idx'] = patterns_data['start_idx'].astype(str)
                
                # Show patterns that match the current data file
                st.subheader("Matching Patterns")
                
                # Get IDs from daily data
                daily_ids = data['ID'].unique()
                
                # Find matching patterns using start_idx
                matching_patterns = patterns_data[patterns_data['start_idx'].isin(daily_ids)]
                
                if not matching_patterns.empty:
                    st.write(f"Found {len(matching_patterns)} matching patterns")
                    
                    # Display matching patterns
                    st.dataframe(matching_patterns.reset_index(drop=True), hide_index=True)
                    
                    # Select a pattern to view details
                    pattern_options = matching_patterns['start_idx'].tolist()
                    
                    selected_pattern_idx = st.selectbox(
                        "Select a pattern to view details:",
                        pattern_options
                    )
                    
                    # Get the pattern details
                    pattern_row = matching_patterns[matching_patterns['start_idx'] == selected_pattern_idx].iloc[0]
                    
                    # Display pattern details
                    st.subheader(f"Pattern Details (start_idx: {selected_pattern_idx})")
                    
                    # Format pattern details for display
                    pattern_details = pd.DataFrame({
                        'Attribute': pattern_row.index,
                        'Value': [str(val) for val in pattern_row.values]  # Convert all values to strings
                    })
                    st.dataframe(pattern_details, hide_index=True)
                    
                    # Get corresponding daily data rows
                    pattern_data = data[data['ID'] == selected_pattern_idx].copy()
                    
                    if not pattern_data.empty:
                        st.subheader(f"Daily Data for Pattern (start_idx: {selected_pattern_idx})")
                        
                        # Get pattern information
                        pattern_duration = int(pattern_row['duration'])
                        pattern_id = str(pattern_row['pattern_id'])
                        
                        # Display the daily data for this pattern
                        if 'TimeFormatted' in pattern_data.columns:
                            display_cols = ['ID', 'TimeFormatted', 'SPX_Px_Last', 'Cap_Contract']
                            display_data = pattern_data[display_cols].copy()
                            display_data.columns = ['ID', 'Time (HH:MM)', 'SPX Price Last', 'Cap Contract']
                        else:
                            display_cols = ['ID', 'SPX_Px_Last', 'Cap_Contract']
                            display_data = pattern_data[display_cols].copy()
                            display_data.columns = ['ID', 'SPX Price Last', 'Cap Contract']
                        
                        st.dataframe(display_data.reset_index(drop=True), hide_index=True)
                    else:
                        st.warning(f"No daily data rows found with ID matching pattern start_idx {selected_pattern_idx}")
                else:
                    st.warning("No matching patterns found for this date's data")
            else:
                st.error("Daily data file does not contain an ID column, cannot connect with patterns") 

with tab4:
    st.header("All Data View")
    
    if not available_dates:
        st.warning("No data found in MongoDB. Please check your connection.")
    else:
        # Date range selection
        st.subheader("Select Date Range")
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.selectbox(
                "Start Date:",
                available_dates,
                index=min(5, len(available_dates)-1),
                format_func=lambda x: x.strftime("%Y-%m-%d"),
                key="start_date_all_data"
            )
        
        with col2:
            end_date = st.selectbox(
                "End Date:",
                available_dates,
                index=0,
                format_func=lambda x: x.strftime("%Y-%m-%d"),
                key="end_date_all_data"
            )
        
        # Ensure proper date order (start_date should be earlier than end_date)
        if start_date > end_date:
            start_date, end_date = end_date, start_date
            st.info("Start and end dates were swapped to maintain chronological order.")
        
        # Function to load data for multiple dates
        @st.cache_data
        def load_data_range(start_date, end_date):
            try:
                # Get all dates between start and end (inclusive)
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                
                # Query MongoDB for data in this date range
                cursor = daily_data_collection.find({
                    "trading_date": {"$gte": start_str, "$lte": end_str}
                })
                data_list = list(cursor)
                
                if not data_list:
                    st.error(f"No data found between {start_str} and {end_str} in MongoDB")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(data_list)
                
                # Remove MongoDB _id field
                if '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                
                # Ensure TimeStamp is in datetime format
                if 'TimeStamp' in df.columns:
                    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
                
                # Add trading_date as datetime if not already present
                if 'trading_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['trading_date']):
                    df['trading_date_dt'] = pd.to_datetime(df['trading_date'])
                
                # Create a combined date-time column for proper chronological plotting
                if 'TimeStamp' in df.columns and 'trading_date' in df.columns:
                    # Create a full datetime by combining trading_date and TimeStamp
                    df['full_datetime'] = df.apply(
                        lambda row: pd.Timestamp.combine(
                            pd.to_datetime(row['trading_date']).date(),
                            row['TimeStamp'].time()
                        ),
                        axis=1
                    )
                
                st.info(f"Loaded {len(df)} records between {start_str} and {end_str} from MongoDB")
                return df
            
            except Exception as e:
                st.error(f"Error loading data range from MongoDB: {str(e)}")
                return None
        
        # Load button to trigger data loading
        load_button = st.button("Load Data Range")
        
        if load_button:
            # Load data for the selected date range
            multi_date_data = load_data_range(start_date, end_date)
            
            if multi_date_data is not None:
                # Filter rows where SPX_Px_Last has no value
                if 'SPX_Px_Last' in multi_date_data.columns:
                    filtered_data = multi_date_data.dropna(subset=['SPX_Px_Last'])
                    if len(filtered_data) < len(multi_date_data):
                        st.info(f"Filtered out {len(multi_date_data) - len(filtered_data)} rows where SPX_Px_Last has no value.")
                    multi_date_data = filtered_data
                
                # Show data info
                st.subheader("Data Summary")
                
                # Count unique trading days
                if 'trading_date' in multi_date_data.columns:
                    unique_days = multi_date_data['trading_date'].nunique()
                    st.write(f"Number of trading days: {unique_days}")
                
                # Data preview
                st.subheader("Data Preview")
                preview_cols = ['trading_date', 'TimeStamp', 'SPX_Px_Last', 'Cap_Contract']
                preview_cols = [col for col in preview_cols if col in multi_date_data.columns]
                
                if preview_cols:
                    st.dataframe(multi_date_data[preview_cols].head(10), hide_index=True)
                
                # Create the multi-day plot
                st.subheader("Multi-Day Price Plot")
                
                # Check if we have the necessary columns
                required_columns = ['full_datetime', 'SPX_Px_Last', 'Cap_Contract']
                missing_columns = [col for col in required_columns if col not in multi_date_data.columns]
                
                if not missing_columns:
                    try:
                        # Create Plotly figure
                        fig = go.Figure()
                        
                        # Sort data chronologically
                        sorted_data = multi_date_data.sort_values(by='full_datetime')
                        
                        # Extract date and time components for filtering
                        sorted_data['date_only'] = sorted_data['full_datetime'].dt.date
                        sorted_data['time_only'] = sorted_data['full_datetime'].dt.time
                        
                        # Filter to include ONLY market hours (16:30 to 23:00) for each trading day
                        trading_hours_data = sorted_data[
                            (sorted_data['time_only'] >= pd.to_datetime('16:30:00').time()) &
                            (sorted_data['time_only'] <= pd.to_datetime('23:00:00').time())
                        ]
                        
                        # Inform user about filtering
                        if len(trading_hours_data) < len(sorted_data):
                            st.info(f"Filtered to show only trading hours (16:30 to 23:00). Showing {len(trading_hours_data)} of {len(sorted_data)} data points.")
                        
                        # Group by date for proper separation
                        grouped_by_date = trading_hours_data.groupby('date_only')
                        
                        # Create continuous trading timeline without overnight gaps
                        continuous_data = []
                        
                        # Keep track of trading day number for continuous timeline
                        trading_day_number = 0
                        
                        # Process each date group and create continuous timeline
                        for date, group in grouped_by_date:
                            # Sort by time within each day
                            day_data = group.sort_values('time_only')
                            
                            # For each row, create a continuous timeline value
                            for _, row in day_data.iterrows():
                                time_obj = row['time_only']
                                
                                # Calculate minutes since 16:30 (market open)
                                minutes_since_open = (time_obj.hour - 16) * 60 + time_obj.minute - 30
                                
                                # Total minutes in trading day
                                minutes_in_day = (23 - 16) * 60 + (0 - 30)  # 23:00 - 16:30 = 390 minutes
                                
                                # Create continuous timeline value:
                                # Each day starts exactly where the previous ended
                                continuous_time = trading_day_number * minutes_in_day + minutes_since_open
                                
                                # Keep original data plus continuous timeline
                                row_data = row.to_dict()
                                row_data['continuous_time'] = continuous_time
                                continuous_data.append(row_data)
                            
                            # Move to next trading day
                            trading_day_number += 1
                        
                        # Convert to DataFrame
                        continuous_df = pd.DataFrame(continuous_data)
                        
                        # Create Plotly figure
                        fig = go.Figure()
                        
                        # Add Cap Contract trace with continuous x-axis
                        fig.add_trace(go.Scatter(
                            x=continuous_df['continuous_time'],
                            y=continuous_df['Cap_Contract'],
                            name='Cap Contract',
                            line=dict(color='green', width=1.5),
                            hovertemplate='Date: %{text}<br>Time: %{customdata}<br>Cap Contract: %{y:.2f}<extra></extra>',
                            text=continuous_df['full_datetime'].dt.strftime('%Y-%m-%d'),
                            customdata=continuous_df['full_datetime'].dt.strftime('%H:%M'),
                        ))
                        
                        # Add SPX_Px_Last trace with continuous x-axis
                        fig.add_trace(go.Scatter(
                            x=continuous_df['continuous_time'],
                            y=continuous_df['SPX_Px_Last'],
                            name='SPX Price Last',
                            line=dict(color='blue', width=2),
                            hovertemplate='Date: %{text}<br>Time: %{customdata}<br>SPX Price: %{y:.2f}<extra></extra>',
                            text=continuous_df['full_datetime'].dt.strftime('%Y-%m-%d'),
                            customdata=continuous_df['full_datetime'].dt.strftime('%H:%M'),
                        ))
                        
                        # Create a list of trading days for x-axis labels
                        trading_days_list = sorted(continuous_df['date_only'].unique())
                        
                        # Create tickvals and ticktext for x-axis
                        # We want a tick for each trading day at market open (16:30)
                        tickvals = [day * ((23 - 16) * 60 + (0 - 30)) for day in range(len(trading_days_list))]
                        ticktext = [pd.Timestamp(day).strftime('%b %d') for day in trading_days_list]
                        
                        # Update layout with continuous timeline
                        fig.update_layout(
                            title=f'SPX Price and Cap Contract ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})',
                            xaxis=dict(
                                title='Trading Days (16:30 - 23:00)',
                                tickvals=tickvals,  # Position of ticks
                                ticktext=ticktext,  # Text for ticks
                                gridcolor='lightgray',
                            ),
                            yaxis=dict(
                                title='Value',
                                gridcolor='lightgray',
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            hovermode="x unified",
                        )
                        
                        # Option to add markers for market open/close
                        add_markers = st.checkbox("Add market open/close markers", value=True)
                        
                        if add_markers:
                            # For each trading day, add markers for market open (16:30) and close (23:00)
                            for day_num in range(len(trading_days_list)):
                                # Market open time (16:30)
                                market_open = day_num * ((23 - 16) * 60 + (0 - 30))
                                
                                # Market close time (23:00)
                                market_close = (day_num + 1) * ((23 - 16) * 60 + (0 - 30))
                                
                                # Add vertical lines for market open and close
                                fig.add_shape(
                                    type="line",
                                    x0=market_open,
                                    y0=continuous_df['SPX_Px_Last'].min() * 0.995,
                                    x1=market_open,
                                    y1=continuous_df['SPX_Px_Last'].max() * 1.005,
                                    line=dict(color="rgba(0,0,255,0.3)", width=1, dash="dash"),
                                )
                                
                                fig.add_shape(
                                    type="line",
                                    x0=market_close,
                                    y0=continuous_df['SPX_Px_Last'].min() * 0.995,
                                    x1=market_close,
                                    y1=continuous_df['SPX_Px_Last'].max() * 1.005,
                                    line=dict(color="rgba(255,0,0,0.3)", width=1, dash="dash"),
                                )
                        
                        # Show the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add note about zooming functionality
                        st.info("💡 Zoom Tips: You can zoom by selecting an area on the plot and double-click to reset the view.")
                        
                        # Add download button for the data
                        st.download_button(
                            label="Download multi-day data as CSV",
                            data=multi_date_data.to_csv(index=False).encode('utf-8'),
                            file_name=f"multi_day_data_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv",
                            mime='text/csv',
                        )
                    
                    except Exception as e:
                        st.error(f"Error creating multi-day plot: {str(e)}")
                else:
                    st.error(f"Missing required columns for plotting: {', '.join(missing_columns)}") 

with tab5:
    st.header("Trades Analysis")
    
    # Function to load pre-trade statistics from MongoDB
    @st.cache_data
    def load_pretrade_stats():
        try:
            # Query MongoDB for pre-trade stats
            cursor = pre_trade_stats_collection.find({})
            stats_list = list(cursor)
            
            if not stats_list:
                st.warning("No pre-trade stats found in MongoDB")
                return None
            
            # Convert to DataFrame
            stats_df = pd.DataFrame(stats_list)
            
            # Remove MongoDB _id field
            if '_id' in stats_df.columns:
                stats_df = stats_df.drop('_id', axis=1)
            
            # Filter out header row
            stats_df = stats_df[stats_df["Unnamed: 0_level_0.Unnamed: 0_level_1"] != "trade_type"]
            
            # Set trade type as index (for Pre-Long and Pre-Short)
            stats_df = stats_df.set_index("Unnamed: 0_level_0.Unnamed: 0_level_1")
            
            return stats_df
            
        except Exception as e:
            st.error(f"Error loading pre-trade stats from MongoDB: {str(e)}")
            return None

    # Function to load pre-trade trend data from MongoDB
    @st.cache_data
    def load_pretrade_trends():
        try:
            # Query MongoDB for pre-trade trends
            cursor = pre_trade_trends_collection.find({})
            trends_list = list(cursor)
            
            if not trends_list:
                st.warning("No pre-trade trends found in MongoDB")
                return None
            
            # Convert to DataFrame
            trends_df = pd.DataFrame(trends_list)
            
            # Remove MongoDB _id field
            if '_id' in trends_df.columns:
                trends_df = trends_df.drop('_id', axis=1)
            
            # Filter out header rows
            trends_df = trends_df[trends_df["trade_type.Unnamed: 1_level_1"].notna()]
            
            return trends_df
            
        except Exception as e:
            st.error(f"Error loading pre-trade trends from MongoDB: {str(e)}")
            return None

    # Function to load trades data from MongoDB
    @st.cache_data
    def load_trades_data():
        try:
            # Get all trades
            cursor = trades_collection.find({})
            trades_list = list(cursor)
            
            if not trades_list:
                st.error("No trades found in MongoDB")
                return None
            
            # Convert to DataFrame
            trades_df = pd.DataFrame(trades_list)
            
            # Remove MongoDB _id field
            if '_id' in trades_df.columns:
                trades_df = trades_df.drop('_id', axis=1)
                
            # Ensure dates are in datetime format
            if 'Start' in trades_df.columns and not pd.api.types.is_datetime64_any_dtype(trades_df['Start']):
                trades_df['Start'] = pd.to_datetime(trades_df['Start'])
                
            if 'End' in trades_df.columns and not pd.api.types.is_datetime64_any_dtype(trades_df['End']):
                trades_df['End'] = pd.to_datetime(trades_df['End'])
                
            # Format date columns for display
            trades_df['DisplayStart'] = trades_df['Start'].dt.strftime('%d/%m/%Y')
            trades_df['DisplayEnd'] = trades_df['End'].dt.strftime('%d/%m/%Y')
            
            # Calculate duration in trading days
            trades_df['Duration'] = (trades_df['End'] - trades_df['Start']).dt.days + 1
                
            # Add a display name for each trade
            trades_df['DisplayName'] = trades_df.apply(
                lambda row: f"{row['DisplayStart']} to {row['DisplayEnd']} ({row['Type']}, {row['Duration']} days)", 
                axis=1
            )
            
            return trades_df
            
        except Exception as e:
            st.error(f"Error loading trades data from MongoDB: {str(e)}")
            return None
    
    # Function to load daily data for a date range
    @st.cache_data
    def load_data_for_trade(start_date, end_date):
        try:
            # Format dates for MongoDB query
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Query MongoDB for data in this date range
            cursor = daily_data_collection.find({
                "trading_date": {"$gte": start_str, "$lte": end_str}
            })
            data_list = list(cursor)
            
            if not data_list:
                st.error(f"No data found between {start_str} and {end_str} in MongoDB")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            
            # Remove MongoDB _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Ensure TimeStamp is in datetime format
            if 'TimeStamp' in df.columns:
                df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
            
            # Convert trading_date to datetime
            if 'trading_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['trading_date']):
                df['trading_date_dt'] = pd.to_datetime(df['trading_date'])
            
            # Create a combined date-time column
            if 'TimeStamp' in df.columns and 'trading_date' in df.columns:
                df['full_datetime'] = df.apply(
                    lambda row: pd.Timestamp.combine(
                        pd.to_datetime(row['trading_date']).date(),
                        row['TimeStamp'].time()
                    ),
                    axis=1
                )
            
            return df
        
        except Exception as e:
            st.error(f"Error loading data for trade period: {str(e)}")
            return None
    
    # Load trades data
    trades_data = load_trades_data()
    
    # Load pre-trade statistics
    pretrade_stats = load_pretrade_stats()
    pretrade_trends = load_pretrade_trends()
    
    if trades_data is not None:
        st.success(f"Found {len(trades_data)} trades in MongoDB")
        
        # Create a dropdown to select a trade
        selected_trade_idx = st.selectbox(
            "Select a trade to analyze:",
            options=range(len(trades_data)),
            format_func=lambda i: trades_data.iloc[i]['DisplayName']
        )
        
        # Get the selected trade details
        selected_trade = trades_data.iloc[selected_trade_idx]
        
        # Display trade details
        st.subheader("Trade Details")
        
        # Create columns for displaying trade details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Start Date", selected_trade['DisplayStart'])
        
        with col2:
            st.metric("End Date", selected_trade['DisplayEnd'])
        
        with col3:
            st.metric("Type", selected_trade['Type'])
        
        # Show pre-trade statistics if available
        if pretrade_stats is not None:
            st.subheader("Pre-Trade Statistics")
            
            # Create tabs for different stat views
            stat_tab1, stat_tab2 = st.tabs(["Summary Statistics", "Trend Analysis"])
            
            with stat_tab1:
                # Determine which trade type stats to show
                trade_type = f"Pre-{selected_trade['Type']}"
                
                # Check if trade_type is in the index
                if trade_type in pretrade_stats.index:
                    trade_stats = pretrade_stats.loc[trade_type]
                    
                    # Display stats in a visually appealing way
                    st.write(f"### Statistics for {trade_type} Periods")
                    
                    # Create metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg Price Change", f"{trade_stats['price_diff.mean']:.2f}")
                        st.metric("Min Price Change", f"{trade_stats['price_diff.min']:.2f}")
                    
                    with col2:
                        st.metric("Price Change Std Dev", f"{trade_stats['price_diff.std']:.2f}")
                        st.metric("Max Price Change", f"{trade_stats['price_diff.max']:.2f}")
                    
                    with col3:
                        st.metric("Avg % Change", f"{trade_stats['price_diff_pct.mean']:.2%}")
                        st.metric("% Change Std Dev", f"{trade_stats['price_diff_pct.std']:.2%}")
                    
                    # Add interpretation
                    st.write("### What This Means")
                    
                    # Different explanations based on trade type
                    if trade_type == "Pre-Long":
                        if trade_stats['price_diff.mean'] < 0:
                            st.info("💡 **Price Dip Before Long Trades**: On average, there's a price decrease of "
                                  f"{abs(trade_stats['price_diff.mean']):.2f} points before long trades. This suggests "
                                  "the strategy enters long positions after a dip, potentially buying on the recovery.")
                        else:
                            st.info("💡 **Price Rise Before Long Trades**: On average, there's a price increase of "
                                  f"{trade_stats['price_diff.mean']:.2f} points before long trades. This suggests "
                                  "the strategy enters long positions during upward momentum.")
                    else:  # Pre-Short
                        if trade_stats['price_diff.mean'] < 0:
                            st.info("💡 **Price Dip Before Short Trades**: On average, there's a price decrease of "
                                  f"{abs(trade_stats['price_diff.mean']):.2f} points before short trades. This suggests "
                                  "the strategy enters short positions during downward momentum.")
                        else:
                            st.info("💡 **Price Rise Before Short Trades**: On average, there's a price increase of "
                                  f"{trade_stats['price_diff.mean']:.2f} points before short trades. This suggests "
                                  "the strategy enters short positions after a rise, potentially shorting the reversal.")
                else:
                    st.warning(f"No pre-trade statistics available for {trade_type}")
            
            with stat_tab2:
                # Show trend data visualization
                if pretrade_trends is not None:
                    # Filter for selected trade type
                    trade_type = f"Pre-{selected_trade['Type']}"
                    
                    # Use the correct column name from MongoDB
                    filtered_trends = pretrade_trends[pretrade_trends["trade_type.Unnamed: 1_level_1"] == trade_type]
                    
                    if not filtered_trends.empty:
                        st.write(f"### Trend Analysis for {trade_type}")
                        
                        # Create trend visualization
                        fig = go.Figure()
                        
                        # Add mean price difference line - use the correct column names
                        fig.add_trace(go.Scatter(
                            x=filtered_trends["days_bin.Unnamed: 2_level_1"],
                            y=filtered_trends["price_diff.mean"],
                            mode='lines+markers',
                            name='Mean Price Difference',
                            line=dict(color='blue', width=3),
                            error_y=dict(
                                type='data',
                                array=filtered_trends["price_diff.std"],
                                visible=True,
                                color='lightblue'
                            )
                        ))
                        
                        # Add percentage change line on secondary y-axis
                        fig.add_trace(go.Scatter(
                            x=filtered_trends["days_bin.Unnamed: 2_level_1"],
                            y=filtered_trends["price_diff_pct.mean"] * 100,  # Convert to percentage
                            mode='lines+markers',
                            name='Mean % Change',
                            line=dict(color='green', width=2, dash='dash'),
                            yaxis='y2'
                        ))
                        
                        # Update layout with dual y-axes
                        fig.update_layout(
                            title=f"Price Changes Before {selected_trade['Type']} Trades by Time Period",
                            xaxis=dict(
                                title="Days Before Trade",
                                tickmode='array',
                                tickvals=filtered_trends["days_bin.Unnamed: 2_level_1"],
                                ticktext=filtered_trends["days_bin.Unnamed: 2_level_1"]
                            ),
                            yaxis=dict(
                                title="Price Difference (points)",
                                titlefont=dict(color='blue'),
                                tickfont=dict(color='blue')
                            ),
                            yaxis2=dict(
                                title="Percentage Change (%)",
                                titlefont=dict(color='green'),
                                tickfont=dict(color='green'),
                                anchor="x",
                                overlaying="y",
                                side="right"
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            height=500,
                            hovermode="x unified"
                        )
                        
                        # Add a horizontal line at y=0 for reference
                        fig.add_shape(
                            type="line",
                            x0=filtered_trends["days_bin.Unnamed: 2_level_1"].iloc[0],
                            y0=0,
                            x1=filtered_trends["days_bin.Unnamed: 2_level_1"].iloc[-1],
                            y1=0,
                            line=dict(color="gray", width=1, dash="dash")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add interpretation
                        st.write("### Interpretation")
                        
                        # Get first and last trend values - Use correct column names
                        first_bin = filtered_trends.iloc[0]
                        last_bin = filtered_trends.iloc[-2]  # Skip the 5+ bin which is empty
                        
                        # Determine if there's a consistent trend
                        trend_direction = "mixed"
                        if all(filtered_trends['price_diff.mean'].iloc[:-1] < 0):  # Skip last row which might be NaN
                            trend_direction = "consistently negative"
                        elif all(filtered_trends['price_diff.mean'].iloc[:-1] > 0):  # Skip last row which might be NaN
                            trend_direction = "consistently positive"
                        
                        if trend_direction == "consistently negative":
                            st.info("📉 **Consistent Price Dips**: The data shows price declines across all time periods "
                                   "before this trade type. This suggests the strategy consistently enters "
                                   f"{selected_trade['Type']} positions after market weakness.")
                        elif trend_direction == "consistently positive":
                            st.info("📈 **Consistent Price Rises**: The data shows price increases across all time periods "
                                   "before this trade type. This suggests the strategy consistently enters "
                                   f"{selected_trade['Type']} positions after market strength.")
                        else:
                            # Check if there's a pattern from early periods to late periods
                            if first_bin['price_diff.mean'] > 0 and last_bin['price_diff.mean'] < 0:
                                st.info("🔄 **Reversal Pattern**: The data shows early price increases followed by "
                                      f"later decreases before {selected_trade['Type']} trades. This might indicate "
                                      "the strategy identifies reversal opportunities.")
                            elif first_bin['price_diff.mean'] < 0 and last_bin['price_diff.mean'] > 0:
                                st.info("🔄 **Recovery Pattern**: The data shows early price decreases followed by "
                                      f"later increases before {selected_trade['Type']} trades. This might indicate "
                                      "the strategy identifies recovery opportunities.")
                            else:
                                st.info("📊 **Mixed Pattern**: The data shows a mixed pattern of price changes "
                                      f"before {selected_trade['Type']} trades. This suggests the strategy "
                                      "may be using multiple factors beyond simple price action.")
                    else:
                        st.warning(f"No trend data available for {trade_type}")
                else:
                    st.error("Trend data could not be loaded")
        
        # Load data for the selected trade period
        start_date = selected_trade['Start']
        end_date = selected_trade['End']
        
        # Add a button to load data
        load_button = st.button("Load Trade Data")
        
        if load_button:
            trade_data = load_data_for_trade(start_date, end_date)
            
            if trade_data is not None:
                # Count the number of trading days with data
                trading_days = trade_data['trading_date'].nunique() if 'trading_date' in trade_data.columns else 0
                st.info(f"Loaded data for {trading_days} trading days between {start_date.strftime('%d/%m/%Y')} and {end_date.strftime('%d/%m/%Y')}")
                
                # Filter rows where SPX_Px_Last has no value
                if 'SPX_Px_Last' in trade_data.columns:
                    filtered_data = trade_data.dropna(subset=['SPX_Px_Last'])
                    if len(filtered_data) < len(trade_data):
                        st.info(f"Filtered out {len(trade_data) - len(filtered_data)} rows where SPX_Px_Last has no value.")
                    trade_data = filtered_data
                
                # Display data preview
                st.subheader("Data Preview")
                preview_cols = ['trading_date', 'TimeStamp', 'SPX_Px_Last', 'Cap_Contract']
                preview_cols = [col for col in preview_cols if col in trade_data.columns]
                
                if preview_cols:
                    st.dataframe(trade_data[preview_cols].head(10), hide_index=True)
                
                # Create plot for trade period
                if 'full_datetime' in trade_data.columns and 'SPX_Px_Last' in trade_data.columns:
                    st.subheader(f"Price Action for {selected_trade['Type']} Trade")
                    
                    try:
                        # Sort data chronologically
                        sorted_data = trade_data.sort_values(by='full_datetime')
                        
                        # Extract date and time components
                        sorted_data['date_only'] = sorted_data['full_datetime'].dt.date
                        sorted_data['time_only'] = sorted_data['full_datetime'].dt.time
                        
                        # Filter to include ONLY market hours (16:30 to 23:00) for each trading day
                        trading_hours_data = sorted_data[
                            (sorted_data['time_only'] >= pd.to_datetime('16:30:00').time()) &
                            (sorted_data['time_only'] <= pd.to_datetime('23:00:00').time())
                        ]
                        
                        # Inform user about filtering
                        if len(trading_hours_data) < len(sorted_data):
                            st.info(f"Filtered to show only trading hours (16:30 to 23:00). Showing {len(trading_hours_data)} of {len(sorted_data)} data points.")
                        
                        # Group by date for continuous timeline
                        grouped_by_date = trading_hours_data.groupby('date_only')
                        
                        # Create continuous trading timeline without overnight gaps
                        continuous_data = []
                        trading_day_number = 0
                        
                        # Process each date group
                        for date, group in grouped_by_date:
                            # Sort by time within each day
                            day_data = group.sort_values('time_only')
                            
                            # Create continuous timeline
                            for _, row in day_data.iterrows():
                                time_obj = row['time_only']
                                
                                # Calculate minutes since 16:30 (market open)
                                minutes_since_open = (time_obj.hour - 16) * 60 + time_obj.minute - 30
                                
                                # Total minutes in trading day
                                minutes_in_day = (23 - 16) * 60 + (0 - 30)  # 23:00 - 16:30 = 390 minutes
                                
                                # Create continuous timeline value
                                continuous_time = trading_day_number * minutes_in_day + minutes_since_open
                                
                                # Keep original data plus continuous timeline
                                row_data = row.to_dict()
                                row_data['continuous_time'] = continuous_time
                                continuous_data.append(row_data)
                            
                            # Move to next trading day
                            trading_day_number += 1
                        
                        # Convert to DataFrame
                        continuous_df = pd.DataFrame(continuous_data)
                        
                        # Create Plotly figure
                        fig = go.Figure()
                        
                        # Add background color based on trade type
                        if selected_trade['Type'] == 'Long':
                            bg_color = 'rgba(0, 128, 0, 0.1)'  # Light green for long
                        else:
                            bg_color = 'rgba(255, 0, 0, 0.1)'  # Light red for short
                        
                        fig.update_layout(
                            plot_bgcolor=bg_color,
                            paper_bgcolor='white'
                        )
                        
                        # Add Cap Contract trace
                        if 'Cap_Contract' in continuous_df.columns:
                            fig.add_trace(go.Scatter(
                                x=continuous_df['continuous_time'],
                                y=continuous_df['Cap_Contract'],
                                name='Cap Contract',
                                line=dict(color='green', width=1.5),
                                hovertemplate='Date: %{text}<br>Time: %{customdata}<br>Cap Contract: %{y:.2f}<extra></extra>',
                                text=continuous_df['full_datetime'].dt.strftime('%Y-%m-%d'),
                                customdata=continuous_df['full_datetime'].dt.strftime('%H:%M'),
                            ))
                        
                        # Add SPX_Px_Last trace
                        fig.add_trace(go.Scatter(
                            x=continuous_df['continuous_time'],
                            y=continuous_df['SPX_Px_Last'],
                            name='SPX Price Last',
                            line=dict(color='blue', width=2),
                            hovertemplate='Date: %{text}<br>Time: %{customdata}<br>SPX Price: %{y:.2f}<extra></extra>',
                            text=continuous_df['full_datetime'].dt.strftime('%Y-%m-%d'),
                            customdata=continuous_df['full_datetime'].dt.strftime('%H:%M'),
                        ))
                        
                        # Create a list of trading days for x-axis labels
                        trading_days_list = sorted(continuous_df['date_only'].unique())
                        
                        # Create tickvals and ticktext for x-axis
                        tickvals = [day * ((23 - 16) * 60 + (0 - 30)) for day in range(len(trading_days_list))]
                        ticktext = [pd.Timestamp(day).strftime('%b %d') for day in trading_days_list]
                        
                        # Update layout
                        fig.update_layout(
                            title=f'{selected_trade["Type"]} Trade: {selected_trade["DisplayStart"]} to {selected_trade["DisplayEnd"]}',
                            xaxis=dict(
                                title='Trading Days (16:30 - 23:00)',
                                tickvals=tickvals,
                                ticktext=ticktext,
                                gridcolor='lightgray',
                            ),
                            yaxis=dict(
                                title='Value',
                                gridcolor='lightgray',
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            hovermode='x unified',
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=600,
                        )
                        
                        # Add markers for market open/close
                        add_markers = st.checkbox("Add market open/close markers", value=True)
                        
                        if add_markers:
                            # For each trading day, add markers for market open (16:30) and close (23:00)
                            for day_num in range(len(trading_days_list)):
                                # Market open time (16:30)
                                market_open = day_num * ((23 - 16) * 60 + (0 - 30))
                                
                                # Market close time (23:00)
                                market_close = (day_num + 1) * ((23 - 16) * 60 + (0 - 30))
                                
                                # Add vertical lines for market open and close
                                fig.add_shape(
                                    type="line",
                                    x0=market_open,
                                    y0=continuous_df['SPX_Px_Last'].min() * 0.995,
                                    x1=market_open,
                                    y1=continuous_df['SPX_Px_Last'].max() * 1.005,
                                    line=dict(color="rgba(0,0,255,0.3)", width=1, dash="dash"),
                                )
                                
                                fig.add_shape(
                                    type="line",
                                    x0=market_close,
                                    y0=continuous_df['SPX_Px_Last'].min() * 0.995,
                                    x1=market_close,
                                    y1=continuous_df['SPX_Px_Last'].max() * 1.005,
                                    line=dict(color="rgba(255,0,0,0.3)", width=1, dash="dash"),
                                )
                        
                        # Show the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add note about zooming
                        st.info("💡 Zoom Tips: You can zoom by selecting an area on the plot and double-click to reset the view.")
                        
                        # Display trade entry and exit points
                        st.subheader("Trade Entry and Exit")
                        
                        # Get first and last data points
                        if not continuous_df.empty:
                            entry_date = continuous_df['full_datetime'].min().strftime('%Y-%m-%d %H:%M')
                            exit_date = continuous_df['full_datetime'].max().strftime('%Y-%m-%d %H:%M')
                            
                            # Get prices
                            entry_price = continuous_df.loc[continuous_df['full_datetime'] == continuous_df['full_datetime'].min(), 'SPX_Px_Last'].values[0]
                            exit_price = continuous_df.loc[continuous_df['full_datetime'] == continuous_df['full_datetime'].max(), 'SPX_Px_Last'].values[0]
                            
                            # Calculate profit/loss
                            if selected_trade['Type'] == 'Long':
                                pnl = exit_price - entry_price
                                pnl_pct = (pnl / entry_price) * 100
                            else:  # Short
                                pnl = entry_price - exit_price
                                pnl_pct = (pnl / entry_price) * 100
                            
                            # Display metrics in a more compact format using columns
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown("**Entry Date**")
                                st.markdown(f"<p style='font-size:14px'>{entry_date}</p>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("**Entry Price**")
                                st.markdown(f"<p style='font-size:14px'>{entry_price:.2f}</p>", unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown("**Exit Date**")
                                st.markdown(f"<p style='font-size:14px'>{exit_date}</p>", unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown("**Exit Price**")
                                st.markdown(f"<p style='font-size:14px'>{exit_price:.2f}</p>", unsafe_allow_html=True)
                            
                            # Show P&L with a simpler approach
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Profit/Loss**")
                                pnl_color = "green" if pnl > 0 else "red"
                                delta_symbol = "+" if pnl > 0 else ""
                                st.markdown(f"<p style='font-size:16px; color:{pnl_color}'>{pnl:.2f} ({delta_symbol}{pnl_pct:.2f}%)</p>", unsafe_allow_html=True)
                            
                            with col2:
                                # Determine if trade was profitable
                                if pnl > 0:
                                    st.markdown("<p style='font-size:16px; color:green'>✅ Profitable Trade</p>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<p style='font-size:16px; color:red'>❌ Unprofitable Trade</p>", unsafe_allow_html=True)
                        
                        # Add download button for the data
                        st.download_button(
                            label="Download trade data as CSV",
                            data=trade_data.to_csv(index=False).encode('utf-8'),
                            file_name=f"trade_data_{selected_trade['DisplayStart'].replace('/', '')}_to_{selected_trade['DisplayEnd'].replace('/', '')}.csv",
                            mime='text/csv',
                        )
                    
                    except Exception as e:
                        st.error(f"Error creating trade plot: {str(e)}")
                else:
                    st.error("Missing required columns for plotting: full_datetime or SPX_Px_Last")
            else:
                st.warning(f"No data found for the selected trade period")
    else:
        st.warning("No trades found. Please ensure your trades collection is populated.") 