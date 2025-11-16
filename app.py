import streamlit as st
import requests

st.title("AI Ad Generator")

product_desc = st.text_input("Product Description")
niche = st.text_input("Niche / Industry")
demographic = st.text_input("Target Demographic")

if st.button("Generate Ad"):
    if not product_desc or not niche or not demographic:
        st.error("Please fill in all fields.")
    else:
        api_url = "http://127.0.0.1:8000/generate_ads"



        response = requests.post(
            api_url,
            data={
                "product_desc": product_desc,
                "niche": niche,
                "demographic": demographic
            },
        )

        if response.status_code == 200:
            st.success("Ad generated successfully!")
            st.json(response.json())
        else:
            st.error(f"Error: {response.status_code}")


