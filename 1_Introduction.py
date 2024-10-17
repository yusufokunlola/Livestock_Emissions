import streamlit as st

st.set_page_config(page_title="Livestock Emission Dashboard", page_icon="🐮")#, layout="wide")

st.header("Introduction")
st.write("""
Ruminant animals, particularly cattle, play a crucial role in global food systems but also contribute significantly to climate change through their digestive processes. Methane, a potent greenhouse gas, is produced as part of normal digestion in these animals. The livestock sector accounts for approximately 14.5% of global anthropogenic greenhouse gas emissions, with ruminants being responsible for about half of this total.

Given the growing concern over climate change and the increasing demand for sustainable agricultural practices, there is a pressing need for accurate and reliable methods to assess and manage livestock emissions. This project addresses this critical need by developing innovative solutions for emission prediction and mitigation strategies.
""")

st.header("Libraries Used")
st.markdown("""
- **Pandas**: For data manipulation and analysis.
- **Plotly**: For visualization.
- **Streamlit**: For creating the web application interface.
""")

st.text('')
st.text('')
st.markdown('`Code:` [GitHub](https://github.com/yusufokunlola/Livestock_Emissions)')