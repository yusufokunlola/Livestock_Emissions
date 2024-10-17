import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load dataset
df = pd.read_csv('dataset/Cattle_CH4_dataset_cleaned_2000_2021.csv')

st.set_page_config(page_title="Livestock Emission Dashboard", page_icon="🐮") 

# Title and description
st.title("Livestock Emission Dashboard")
st.write("Browse the livestock emission of cattle from the [FAOSTAT Emissions database](https://www.fao.org/faostat/en/#data/GLE).")

# Spacing
st.write("")
st.write("")

# Create a slider for year range selection
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
selected_years = st.slider("Which years are you interested in?", min_year, max_year, (min_year, max_year))

# Create a multi-select dropdown for Area Code (ISO3) selection
area_codes = df['Area Code (ISO3)'].unique()

# Default selection of 3 area codes (first three by default)
default_selection = area_codes[:3]  

selected_area_codes = st.multiselect("Which countries would you like to view?", area_codes, default=default_selection)

# Filter data based on the selected year range and Area Codes
filtered_data = df[(df['Year'].between(selected_years[0], selected_years[1])) & (df['Area Code (ISO3)'].isin(selected_area_codes))]

# Subheader for Cattle CH4 Emission chart
st.subheader(f"Cattle CH4 Emission from {selected_years[0]} to {selected_years[1]}")

# Create a line plot using Plotly
if not filtered_data.empty:
    fig = px.line(filtered_data, x='Year', y='Value', color='Area', markers=False)
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected criteria.")




# Group by continent and count unique areas (countries), sorted in descending order
continent_area_counts = df.groupby('Continent')['Area'].nunique().sort_values(ascending=True)

# Create a Plotly horizontal bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=continent_area_counts.values,
    y=continent_area_counts.index,
    orientation='h',
    marker_color='teal',
    text=continent_area_counts.values,  # Add the values as text on the bars
    textposition='auto'  # Automatically place the text in a good spot
))

# Add labels and customize the layout
fig.update_layout(
    title='Number of Unique Countries in Each Continent',
    xaxis_title='Number of Unique Countries',
    yaxis_title='Continent',
    showlegend=False,
    xaxis=dict(showticklabels=False),  # Remove x-axis ticks
    margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins for better layout
)

# Display the Plotly plot in Streamlit
st.plotly_chart(fig)


# -----------------------
# Methane Emissions Chart by Continent
# -----------------------

st.subheader(f"Continental Average Methane Emissions from Cattle (2000-2021)")
# Group the data by continent and year, and calculate the mean of the 'Value' column
emissions_by_continent_year = df.groupby(['Continent', 'Year'])['Value'].mean().reset_index()

# Create an empty figure
fig_emissions = go.Figure()

# Add a trace for each continent
continents = emissions_by_continent_year['Continent'].unique()
for continent in continents:
    continent_data = emissions_by_continent_year[emissions_by_continent_year['Continent'] == continent]
    fig_emissions.add_trace(go.Scatter(x=continent_data['Year'], y=continent_data['Value'], name=continent, visible=False))

# Make only the first continent visible
fig_emissions.data[0].visible = True

# Create dropdown menu options for continent selection
dropdown_buttons = []
for i, continent in enumerate(continents):
    dropdown_buttons.append(dict(method='update',
                                 label=continent,
                                 args=[{'visible': [j == i for j in range(len(continents))]},
                                       {'title': f'Average Methane Emissions from Cattle in {continent} (2000-2021)'}]))

# Add dropdown to the layout
fig_emissions.update_layout(
    updatemenus=[dict(active=0, buttons=dropdown_buttons, x=1.15, y=1.25)],
    xaxis_title='Year',
    yaxis_title='Total Methane Emissions (kt)',
)

# Display the emissions figure in the Streamlit app
st.plotly_chart(fig_emissions)


st.subheader("Methane Emissions from Cattle for Top 5/ Bottom 5 Countries Worldwide (2000-2021)")
# -----------------------
# Function to create the plot with a dropdown for selecting top/bottom countries
def plot_emissions_with_dropdown(df, n=5):
    # Calculate total emissions by country
    total_emissions_by_country = df.groupby('Area')['Value'].sum()

    # Create traces for top and bottom countries
    top_countries = total_emissions_by_country.nlargest(n).index
    bottom_countries = total_emissions_by_country.nsmallest(n).index

    # Filter the DataFrame for the selected countries
    df_top = df[df['Area'].isin(top_countries)]
    df_bottom = df[df['Area'].isin(bottom_countries)]

    # Initialize the figure
    fig = go.Figure()

    # Add traces for top countries
    for country in top_countries:
        country_data = df_top[df_top['Area'] == country]
        fig.add_trace(go.Scatter(x=country_data['Year'], y=country_data['Value'], name=country))

    # Add traces for bottom countries
    for country in bottom_countries:
        country_data = df_bottom[df_bottom['Area'] == country]
        fig.add_trace(go.Scatter(x=country_data['Year'], y=country_data['Value'], name=country, visible='legendonly'))

    # Create a dropdown menu for selecting top/bottom
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'label': 'Top 5 Countries',
                    'method': 'update',
                    'args': [
                        {'visible': [True] * n + [False] * n},  # Only show top countries
                        {'title': 'Methane Emissions from Cattle for Top 5 Countries Worldwide (2000-2021)'},
                    ],
                },
                {
                    'label': 'Bottom 5 Countries',
                    'method': 'update',
                    'args': [
                        {'visible': [False] * n + [True] * n},  # Only show bottom countries
                        {'title': 'Methane Emissions from Cattle for Bottom 5 Countries Worldwide (2000-2021)'},
                    ],
                },
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.98,
            'xanchor': 'left',
            'y': 1.28,
            'yanchor': 'top'
        }]
    )

    # Update layout for better readability
    fig.update_layout(xaxis_title='Year',
                      yaxis_title='Total Methane Emissions (kt)',
                      legend_title_text='Country',
                      template='plotly_white',
                      title='Methane Emissions from Cattle Worldwide (2000-2021)')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

# Call the function to plot for n=5 countries
plot_emissions_with_dropdown(df, n=5)




st.subheader(f"Top 10 Countries' Methane Emission per Continent")

# Group by Continent, Area, and Year, and average the 'Value'
grouped = df.groupby(['Continent', 'Area', 'Year'])['Value'].mean().reset_index()

# Find the top 10 countries per continent based on total Value
top_countries_per_continent = grouped.groupby(['Continent', 'Area'])['Value'].mean().reset_index()
top_10_countries = top_countries_per_continent.groupby('Continent').apply(lambda x: x.nlargest(10, 'Value')).reset_index(drop=True)

# Create a figure with dropdown
fig = go.Figure()

# Create dropdown options for each continent
for continent in top_10_countries['Continent'].unique():
    top_countries = top_10_countries[top_10_countries['Continent'] == continent]['Area'].values
    continent_data = grouped[(grouped['Continent'] == continent) & (grouped['Area'].isin(top_countries))]

    # Add trace for each country in the selected continent
    for country in top_countries:
        country_data = continent_data[continent_data['Area'] == country]
        fig.add_trace(go.Scatter(x=country_data['Year'], y=country_data['Value'],
                                 name=f'{country} - {continent}',
                                 visible=False))  # Initially set all traces to invisible

# Add visibility toggle through dropdown buttons
buttons = []
for i, continent in enumerate(top_10_countries['Continent'].unique()):
    visible = [False] * len(fig.data)
    start_idx = i * 10  # Each continent has 10 countries
    visible[start_idx:start_idx + 10] = [True] * 10  # Show only the countries for the selected continent

    buttons.append(dict(label=continent,
                        method="update",
                        args=[{"visible": visible},
                              {"title": f"Top 10 Countries' Methane Emission in {continent} (2000-2021)"}]))

# Update layout with dropdown menu
fig.update_layout(
    updatemenus=[dict(active=0,
                      buttons=buttons,
                      x=0.95, y=1.28,  # Position the dropdown
                      xanchor='left', yanchor='top')],
    title="Select a Continent to View Top 10 Countries' Methane Emissions",
    xaxis_title="Year",
    yaxis_title="Methane Emission Value (kt)",
    template="plotly_white"
)

# Make the first continent (default) visible
for i in range(10):
    fig.data[i].visible = True

# Display the emissions figure in the Streamlit app
st.plotly_chart(fig)

st.text('')
st.text('')
st.markdown('`Code:` [GitHub](https://github.com/yusufokunlola/Livestock_Emissions)')