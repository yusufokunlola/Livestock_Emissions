import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="African Continent Livestock Methane Emission Dashboard", page_icon="🐮") 

   
# Title and description
st.title("The African Continent Livestock Methane Emission Dashboard")

# Load dataset
africa_data = pd.read_csv('dataset/Cattle_CH4_dataset_cleaned_2000_2021.csv')

africa_data = africa_data[africa_data['Continent'] == 'Africa']

# Map countries into regions
east = ['Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia', 'Kenya', 'Madagascar', 'Malawi', 'Mauritius', 'Mozambique', 'Réunion',
        'Rwanda', 'Seychelles', 'Somalia', 'Uganda', 'United Republic of Tanzania', 'Zambia']
south = ['Botswana', 'Eswatini', 'Lesotho', 'Namibia', 'South Africa', 'Zimbabwe']
north = ['Algeria', 'Egypt', 'Libya', 'Morocco', 'Tunisia']
west = ['Benin', 'Burkina Faso', 'Cabo Verde', "Côte d'Ivoire", 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania',
        'Niger', 'Nigeria', 'Senegal', 'Sierra Leone', 'Togo']
central = ['Angola', 'Cameroon', 'Central African Republic', 'Chad', 'Congo', 'Democratic Republic of the Congo', 'Equatorial Guinea', 'Gabon', 'Sao Tome and Principe']

# Create a dictionary of regions and countries
regions = {
    'East Africa': east,
    'Southern Africa': south,
    'North Africa': north,
    'West Africa': west,
    'Central Africa': central
}

# Create a function to map countries to regions
def get_region(country):
    for region, countries in regions.items():
        if country in countries:
            return region
    return None

# Create a new column 'Region' using .loc to avoid the warning
africa_data.loc[:, 'Region'] = africa_data['Area'].apply(get_region)

# Spacing
st.write("")
st.write("")

# Create a slider for year range selection
min_year = int(africa_data['Year'].min())
max_year = int(africa_data['Year'].max())
selected_years = st.slider("Which years are you interested in?", min_year, max_year, (min_year, max_year))

# Create a multi-select dropdown for Area Code (ISO3) selection
area_codes = africa_data['Area Code (ISO3)'].unique()

# Default selection of 3 area codes (first three by default)
default_selection = area_codes[:3]  

selected_area_codes = st.multiselect("Which countries in Africa would you like to view?", area_codes, default=default_selection)

# Filter data based on the selected year range and Area Codes
filtered_data = africa_data[(africa_data['Year'].between(selected_years[0], selected_years[1])) & (africa_data['Area Code (ISO3)'].isin(selected_area_codes))]

# Spacing
st.write("")
st.write("")

# Subheader for Cattle CH4 Emission chart
st.subheader(f"Cattle CH4 Emission from {selected_years[0]} to {selected_years[1]}")
# Spacing
st.write("")
st.write("")

# Create a line plot using Plotly
if not filtered_data.empty:
    fig = px.line(filtered_data, x='Year', y='Value', color='Area', markers=False)
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected criteria.")
    

    
# Spacing
st.write("")
st.write("")
# @title Number of Unique Countries in Each Region

# Group by Region and count unique areas (countries), sorted in descending order
Region_area_counts = africa_data.groupby('Region')['Area'].nunique().sort_values(ascending=True)

# Create a Plotly horizontal bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=Region_area_counts.values,
    y=Region_area_counts.index,
    orientation='h',
    marker_color='teal',
    text=Region_area_counts.values,  # Add the values as text on the bars
    textposition='auto'  # Automatically place the text in a good spot
))

# Add titles and labels
fig.update_layout(
    title='Number of Unique Countries in Each Region',
    xaxis_title='Number of Unique Countries',
    yaxis_title='Region',
    showlegend=False,
    xaxis=dict(showticklabels=False)
)

# Display the Plotly plot in Streamlit
st.plotly_chart(fig)


# Spacing
st.write("")
st.write("")

# @title Livestock Methane Emmission Distribution for African Regions (2000-2021)

# Group by 'Region' and sum the 'Value'
region_value_sum = africa_data.groupby('Region')['Value'].sum().reset_index()

# Sort the data by 'Value' in ascending order
region_value_sum = region_value_sum.sort_values(by='Value', ascending=True)

# Create a horizontal bar chart
fig = px.bar(region_value_sum,
             x='Value',
             y='Region',
             orientation='h',  # Horizontal bars
             title='Total Methane Emissions by Region in Africa',
            #  labels={'Value': 'Total Methane Emissions (kt)', 'Region': 'Region'},
            #  text='Value'
            )  # Add text labels

# Customize layout
fig.update_layout(
    template='plotly_white',
    xaxis=dict(showticklabels=False),  # Remove x-axis ticks
    showlegend=False
)

# Add labels to the end of each bar
fig.update_traces(textposition='auto')

# Remove the top, right, and bottom border
fig.update_layout(
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis=dict(showline=False),
    xaxis=dict(showline=False)
)


# Display the Plotly plot in Streamlit
st.plotly_chart(fig)



# Spacing
st.write("")
st.write("")

#@title Average Methane Emissions from Cattle by African Region (2000-2021)

# Group the data by Region and year, and calculate the mean of the 'Value' column
emissions_by_Region_year = africa_data.groupby(['Region', 'Year'])['Value'].mean().reset_index()

# Create an empty figure
fig = go.Figure()

# Add a trace for each Region
Regions = emissions_by_Region_year['Region'].unique()
for Region in Regions:
    Region_data = emissions_by_Region_year[emissions_by_Region_year['Region'] == Region]
    fig.add_trace(go.Scatter(x=Region_data['Year'], y=Region_data['Value'], name=Region, visible=False))

# Make only the first Region visible
fig.data[0].visible = True

# Create dropdown menu options
dropdown_buttons = []
for i, Region in enumerate(Regions):
    dropdown_buttons.append(dict(method='update',
                                 label=Region,
                                 args=[{'visible': [j == i for j in range(len(Regions))]},
                                       {'title': f'Average Methane Emissions from Cattle in {Region} (2000-2021)'}]))

# Add dropdown to the layout
fig.update_layout(
    updatemenus=[dict(active=0, buttons=dropdown_buttons, x=1.15, y=1.15)],
    title=f'Average Methane Emissions from Cattle in {Regions[0]} (2000-2021)',
    xaxis_title='Year',
    yaxis_title='Total Methane Emissions (kt)',
)

# Show the figure
st.plotly_chart(fig)



# Spacing
st.write("")
st.write("")

# @title Top 5 Countries' Methane Emission in Africa Regions

# Group by Region, Area, and Year, and average the 'Value'
grouped = africa_data.groupby(['Region', 'Area', 'Year'])['Value'].mean().reset_index()

# Find the top 5 countries per Region based on total Value
top_countries_per_Region = grouped.groupby(['Region', 'Area'])['Value'].mean().reset_index()
top_5_countries = top_countries_per_Region.groupby('Region').apply(lambda x: x.nlargest(5, 'Value')).reset_index(drop=True)

# Create a figure with dropdown
fig = go.Figure()

# Create dropdown options for each Region
for Region in top_5_countries['Region'].unique():
    top_countries = top_5_countries[top_5_countries['Region'] == Region]['Area'].values
    Region_data = grouped[(grouped['Region'] == Region) & (grouped['Area'].isin(top_countries))]

    # Add trace for each country in the selected Region
    for country in top_countries:
        country_data = Region_data[Region_data['Area'] == country]
        fig.add_trace(go.Scatter(x=country_data['Year'], y=country_data['Value'],
                                 name=f'{country} - {Region}',
                                 visible=False))  # Initially set all traces to invisible

# Add visibility toggle through dropdown buttons
buttons = []
for i, Region in enumerate(top_5_countries['Region'].unique()):
    visible = [False] * len(fig.data)
    start_idx = i * 5  # Each Region has 5 countries
    visible[start_idx:start_idx + 5] = [True] * 5  # Show only the countries for the selected Region

    buttons.append(dict(label=Region,
                        method="update",
                        args=[{"visible": visible},
                              {"title": f"Top 5 Countries' Methane Emission in {Region} (2000-2021)"}]))

# Update layout with dropdown menu
fig.update_layout(
    updatemenus=[dict(active=0,
                      buttons=buttons,
                      x=0.17, y=1.15,  # Position the dropdown
                      xanchor='left', yanchor='top')],
    title="Select an African Region to View Top 5 Countries' Methane Emissions",
    xaxis_title="Year",
    yaxis_title="Methane Emission Value (kt)",
    template="plotly_white"
)

# Make the first Region (default) visible
for i in range(5):
    fig.data[i].visible = True

# Show the figure
st.plotly_chart(fig)


st.text('')
st.text('')
st.markdown('`Code:` [GitHub](https://github.com/yusufokunlola/Livestock_Emissions)')