import streamlit as st
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Set up Snowflake connection
@st.cache_resource
def create_snowflake_connection():
    return snowflake.connector.connect(
        user='Krish73',
        password='Kri$h2685',
        account='tuzcxib-ux02731',
        warehouse='COMPUTE_WH',
        database='URBANPLANNING',
        schema='URBANPLAN'
    )

conn = create_snowflake_connection()


# Set title for the app
st.title("Urban Planning Insights and Forecasts")

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = [
    "Population Density by State", 
    "Fully vs. Partially Covered Habitation by State", 
    "Quality-Affected Habitations by State", 
    "Land Use Distribution Across States",
    "Urban Population Growth Over Decades", 
    "Housing Demand vs. Allotment by State",
    "Summary",
    "Prediction"
]
choice = st.sidebar.radio("Select Analysis Section", sections)

# Query Snowflake

query1 = """
        SELECT STATES, AVG(POPULATION_DENSITY) AS avg_density
        FROM POPULATION
        GROUP BY STATES
    """
df_density = pd.read_sql(query1, conn)

query2 = """
        SELECT STATES, 
        (FULLY_COVERED / TOTAL) * 100 AS percent_fully_covered,
        (PARTIALLY_COVERED / TOTAL) * 100 AS percent_partially_covered
        FROM HABITATION
    """
df_habitation = pd.read_sql(query2, conn)

query3 = """
        SELECT STATES, QUALITY_AFFECTED_HABITATIONS
        FROM HABITATION
    """
df_quality = pd.read_sql(query3, conn)

query4 = """
        SELECT STATES, FORESTS, NOT_AVAILABLE_FOR_CULTIVATION, PERMANENT_PASTURES_AND_OTHER_GRAZING_LANDS,
        LAND_UNDER_MISCELLANEOUS_TREE_CROPS_GROVES, CULTURABLE_WASTELAND, 
        FALLOW_LANDS_OTHER_THAN_CURRENT_FALLOWS, CURRENT_FALLOWS, NET_AREA_SOWN
        FROM LAND_USE_PATTERN"""
df_land_use = pd.read_sql(query4, conn)

query5 = """
        SELECT YEAR, PERCENT_URBAN_TO_TOTAL, DECADAL_URBAN_POPULATION_GROWTH
        FROM GROWTH_OF_URBAN_POPULATION
    """
df_urban_growth = pd.read_sql(query5, conn)

query6 = """
        SELECT STATES, ASSESSED_DEMAND, HOUSES_SANCTIONED
        FROM HOUSING
    """
df_housing = pd.read_sql(query6, conn)

# Display the selected section
if choice == "Population Density by State":
    st.header("Population Density by State")
    st.write("This section analyzes the average population density by state.")
    
    # Display data in Streamlit
    st.dataframe(df_density)

    # Multiselect for selecting specific states or all states
    state_list = ["All States"] + df_density['STATES'].unique().tolist()
    selected_states = st.multiselect("Select State(s)", state_list, default="All States")

    # Filter data based on selected states
    if "All States" in selected_states:
        df_filtered = df_density
    else:
        df_filtered = df_density[df_density['STATES'].isin(selected_states)]
    st.dataframe(df_filtered)  # Show filtered data

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='STATES', y='AVG_DENSITY', data=df_filtered, palette='viridis',hue='STATES',dodge=False,legend=False)
    plt.xticks(rotation=90)
    plt.title("Average Population Density by State")
    plt.xlabel("State")
    plt.ylabel("Average Population Density")
    
    st.pyplot(plt)

    # Insight
    st.markdown("### Insight")
    st.markdown("States with high population density may face infrastructure challenges, such as urban congestion and limited resources, while low-density states might have opportunities for expansion and development.")


elif choice == "Fully vs. Partially Covered Habitation by State":
    st.header("Fully vs. Partially Covered Habitation by State")
    st.write("Examining the extent of habitation coverage in each state.")
    
    # Display data in Streamlit
    st.dataframe(df_habitation)

    # Multiselect for states with an "All States" option
    state_list = ["All States"] + df_habitation['STATES'].unique().tolist()
    selected_states = st.multiselect("Select State(s)", state_list, default="All States")

    # Filter data based on selected states
    if "All States" in selected_states:
        df_filtered = df_habitation  # Display all data if "All States" is selected
    else:
        df_filtered = df_habitation[df_habitation['STATES'].isin(selected_states)]
        st.dataframe(df_filtered)
            
    # Plotting
    plt.figure(figsize=(10, 6))
    if not df_filtered.empty:
        df_filtered.set_index('STATES')[['PERCENT_FULLY_COVERED', 'PERCENT_PARTIALLY_COVERED']].plot(
            kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'])
    else:
        plt.bar([],[])
    plt.xticks(rotation=90)
    plt.title("Fully vs. Partially Covered Habitations by State")
    plt.xlabel("State")
    plt.ylabel("Percentage")
    
    st.pyplot(plt)

    # Insight
    st.markdown("### Insight")
    st.markdown("Comparing fully and partially covered habitations helps identify areas needing resource allocation for complete habitation coverage.")

elif choice == "Quality-Affected Habitations by State":
    st.header("Quality-Affected Habitations by State")
    st.write("Identifying areas affected by quality issues in habitation.")

    # Display data in Streamlit
    st.dataframe(df_quality)

    # Slider for quality-affected threshold
    threshold = st.slider("Minimum Quality-Affected Habitations", min_value=0, max_value=int(df_quality['QUALITY_AFFECTED_HABITATIONS'].max()), value=100)

    # Filter data based on threshold
    df_filtered = df_quality[df_quality['QUALITY_AFFECTED_HABITATIONS'] >= threshold]
    st.dataframe(df_filtered)

    # Plotting
    plt.figure(figsize=(10, 6))
    if not df_filtered.empty:
        sns.barplot(x='STATES', y='QUALITY_AFFECTED_HABITATIONS', data=df_filtered, palette='magma')
    else:
        plt.bar([], [])  # Show an empty chart if no states meet the threshold
    plt.xticks(rotation=90)
    plt.title(f"Quality-Affected Habitations (Threshold: {threshold}+) by State")
    plt.xlabel("State")
    plt.ylabel("Quality-Affected Habitations")

    st.pyplot(plt)

    # Insight
    st.markdown("### Insight")
    st.markdown("States with higher quality-affected habitations may need targeted improvements in housing quality and resource distribution.")

elif choice == "Land Use Distribution Across States":
    st.header("Land Use Distribution Across States")
    st.write("Overview of land use types such as forests, agricultural land, and urban areas by state.")
    
    # Display data in Streamlit
    st.dataframe(df_land_use)

    # Dropdown for land use type selection
    land_use_types = [
        'FORESTS', 'NOT_AVAILABLE_FOR_CULTIVATION', 'PERMANENT_PASTURES_AND_OTHER_GRAZING_LANDS',
        'LAND_UNDER_MISCELLANEOUS_TREE_CROPS_GROVES', 'CULTURABLE_WASTELAND', 
        'FALLOW_LANDS_OTHER_THAN_CURRENT_FALLOWS', 'CURRENT_FALLOWS', 'NET_AREA_SOWN'
    ]
    selected_land_use = st.selectbox("Select Land Use Type", land_use_types)

    # Multiselect for selecting specific states or all states
    state_list = ["All States"] + df_land_use['STATES'].unique().tolist()
    selected_states = st.multiselect("Select State(s)", state_list, default="All States")

    # Filter data based on selected states and land use type
    if "All States" in selected_states:
        df_filtered = df_land_use[['STATES', selected_land_use]]  # Show all data if "All States" is selected
    else:
        df_filtered = df_land_use[df_land_use['STATES'].isin(selected_states)][['STATES', selected_land_use]]
    st.dataframe(df_filtered)

    # Plotting with Seaborn
    plt.figure(figsize=(12, 8))
    if not df_filtered.empty:
        sns.barplot(data=df_filtered, x='STATES', y=selected_land_use, palette='viridis')
    else:
        plt.bar([], [])  # Show empty chart if no data is selected

    plt.title(f"{selected_land_use.replace('_', ' ').title()} Across Selected States")
    plt.xlabel("State")
    plt.ylabel("Area")
    plt.xticks(rotation=90)
    st.pyplot(plt)

    # Insight
    st.markdown("### Insight")
    st.markdown("Analyzing land use by type helps understand each state’s resource utilization and potential areas for sustainable development.")

elif choice == "Urban Population Growth Over Decades":
    st.header("Urban Population Growth Over Decades")
    st.write("Analyzing urban population growth as a percentage of the total population across decades.")
   
    # Display data in Streamlit
    st.dataframe(df_urban_growth)

    # Range slider for year selection
    min_year = int(df_urban_growth['YEAR'].min())
    max_year = int(df_urban_growth['YEAR'].max())
    selected_year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

    # Filter data based on selected year range
    df_filtered = df_urban_growth[(df_urban_growth['YEAR'] >= selected_year_range[0]) & 
                                  (df_urban_growth['YEAR'] <= selected_year_range[1])]
    st.dataframe(df_filtered)

    # Plotting
    plt.figure(figsize=(10, 6))
    if not df_filtered.empty:
        sns.lineplot(x='YEAR', y='PERCENT_URBAN_TO_TOTAL', data=df_filtered, marker='o', color='b')
    else:
        plt.plot([], [])  # Show an empty chart if no data meets the range criteria
    plt.title(f"Urban Population Growth from {selected_year_range[0]} to {selected_year_range[1]}")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Urban Population to Total")
    
    st.pyplot(plt)

    # Insight
    st.markdown("### Insight")
    st.markdown("Rising urbanization rates reflect shifts toward urban areas, indicating demands on urban infrastructure and services.")

elif choice == "Housing Demand vs. Allotment by State":
    st.header("Housing Demand vs. Allotment by State")
    st.write("Comparing housing demand with the number of houses allotted to identify potential gaps.")
   
    # Display data in Streamlit
    st.dataframe(df_housing)

    # Multiselect for selecting specific states or view all
    state_list = ["All States"] + df_housing['STATES'].unique().tolist()
    selected_states = st.multiselect("Select State(s) with High Demand-Allotment Gap", state_list, default="All States")

    # Filter data based on selected states
    if "All States" in selected_states:
        df_filtered = df_housing  # Display all data if "All States" is selected
    else:
        df_filtered = df_housing[df_housing['STATES'].isin(selected_states)]
    st.dataframe(df_filtered)

    # Plotting
    plt.figure(figsize=(10, 6))
    if not df_filtered.empty:
        sns.barplot(data=df_filtered.melt(id_vars="STATES", var_name="Type", value_name="Count"),
                x="STATES", y="Count", hue="Type", palette=['#1f77b4', '#ff7f0e'])
    else:
        plt.bar([],[])
    plt.xticks(rotation=90)
    plt.title("Housing Demand vs. Allotment by State")
    plt.xlabel("State")
    plt.ylabel("Count")
    
    st.pyplot(plt)

    # Insight
    st.markdown("### Insight")
    st.markdown("A high demand-allotment gap reveals potential shortfalls in housing allocations, highlighting areas needing further support.")

elif choice == "Summary":
    st.header("Summary")
    st.markdown("This project provides an integrated analysis of India’s infrastructure and urban development landscape, revealing critical insights "
    "into the factors shaping the nation's growth trajectory. With rapid urbanization, states experience diverse challenges, such as "
    "population density strains, resource allocation disparities, and varying levels of habitation coverage. These findings underscore "
    "the need for balanced development and strategic interventions. Policies focused on equitable housing allocations, enhanced urban infrastructure, "
    "and sustainable land use can support resilient growth. To address emerging demands, a coordinated approach involving public-private partnerships, "
    "local governance enhancements, and sustainable investment strategies could be instrumental. This approach not only mitigates existing urban pressures "
    "but also aligns with long-term goals for a resilient, equitable, and sustainable urban future."
    )

elif choice == "Prediction":
    # Displaying the Data
    st.header("Urban Population Growth Prediction")
    st.write("Historical data on urban population growth as a percentage of the total population by decade.")
    st.write(df_urban_growth)

    # Prepare Data for Model
    X = df_urban_growth[['YEAR']]
    y = df_urban_growth['PERCENT_URBAN_TO_TOTAL']

    # Train Model
    model = LinearRegression()
    model.fit(X, y)

    # Predict Urban Growth for Future Decades (2021, 2031, 2041, ...)
    future_decades = pd.DataFrame({'YEAR': [2021, 2031, 2041, 2051, 2061]})
    future_growth_pred = model.predict(future_decades)

    # Use Polynomial Regression for non-linear prediction
    poly = PolynomialFeatures(degree=2)  # Degree 2 for quadratic curve
    X_poly = poly.fit_transform(df_urban_growth[['YEAR']])
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict for historical data
    actual_years_poly = poly.transform(df_urban_growth[['YEAR']])
    actual_predictions = model.predict(actual_years_poly)

    # Predict for future decades
    future_decades_poly = poly.transform(future_decades)
    future_predictions = model.predict(future_decades_poly)

    # Display Future Predictions
    future_decades['PREDICTED_PERCENT_URBAN_TO_TOTAL'] = future_growth_pred
    st.write("Predicted Urban Population Growth for Future Decades")
    st.write(future_decades)

    # Plot actual data
    plt.figure(figsize=(10, 6))
    plt.plot(df_urban_growth['YEAR'], df_urban_growth['PERCENT_URBAN_TO_TOTAL'], color='red',marker='o', label='Actual Data')
    plt.plot([2011,2021],[df_urban_growth['PERCENT_URBAN_TO_TOTAL'].iloc[-1],future_predictions[0]],color = 'blue')

    # Plot future predictions
    plt.plot(future_decades['YEAR'], future_predictions, label='Predicted Growth', color='blue', marker='o')

    # Final adjustments
    plt.xlabel('Year')
    plt.ylabel('Urban Population Growth (%)')
    plt.title('Urban Population Growth Prediction (by Decade)')
    plt.legend()
    st.pyplot(plt)