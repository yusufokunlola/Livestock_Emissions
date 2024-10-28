import streamlit as st

st.set_page_config(page_title="Livestock Methane Emission Dashboard", page_icon="🐮")#, layout="wide")

st.title("Livestock Methane Emissions Project")

st.header("Introduction")
st.write("""
Ruminant animals, particularly cattle, play a crucial role in global food systems but also contribute significantly to climate change through their digestive processes. Methane, a potent greenhouse gas, is produced as part of normal digestion in these animals. The livestock sector accounts for approximately 14.5% of global anthropogenic greenhouse gas emissions, with ruminants being responsible for about half of this total.

Given the growing concern over climate change and the increasing demand for sustainable agricultural practices, there is a pressing need for accurate and reliable methods to assess and manage livestock emissions. This project addresses this critical need by developing innovative solutions for emission prediction and mitigation strategies.

The data was sourced from the [FAOSTAT Emissions database](https://www.fao.org/faostat/en/#data/GLE).
""")

st.header("Objectives")
st.markdown("""

1. To conduct a comprehensive analysis of methane (CH4) emissions from cattle across the globe.

2. To develop and train machine learning model capable of predicting cattle CH4 emissions.

3. To provide actionable recommendations for stakeholders in the agricultural sector, policymakers, and environmental organizations to reduce livestock-related greenhouse gas emissions and promote sustainable agriculture practices.

By achieving these goals, the project aims to contribute significantly to global efforts to combat climate change while supporting the continued development of the livestock industry through more efficient and environmentally conscious practices.
""")


st.header("Libraries Used")
st.markdown("""
- **Pandas**: For data manipulation and analysis.
- **Plotly**: For visualization.
- **Facebook Prophet**: For future predictions.
- **Streamlit**: For creating the web application interface.
""")

st.text('')
st.text('')
st.write("""Collaborators: [Yusuf Okunlola, GMNSE](mailto:yusufokunlola@gmail.com) and [Dr. Young Irivboje](mailto:youngiriv@yahoo.com)""")
st.markdown('`Code:` [GitHub](https://github.com/yusufokunlola/Livestock_Emissions)')