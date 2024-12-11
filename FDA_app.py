# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:46:11 2024

@author: user
"""

import streamlit as st
from FDA_V3 import pass_output

def main():
    st.title("Warning Letter - GenAI CGMP Violation Analyst")

    # Upload PDF button
    uploaded_file = st.sidebar.file_uploader("Upload a PDF Document", type=["pdf"])

    # Show the "Process" button only if a file is uploaded
    if uploaded_file:
        if st.sidebar.button("Process"):
            # Process the uploaded PDF
            warning_letter_info, inspection_summary,violations,formatted_RI_info,conclusion = pass_output(uploaded_file)

            # Display extracted information in two sections
            #st.subheader("Extracted Information")
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Warning Letter Info")
                #st.text_area("", value=warning_letter_info, height=200)
                st.markdown(warning_letter_info)

            with col2:
                st.write("### Inspection Summary")
                st.markdown(inspection_summary)
                #st.text_area("", value=inspection_summary, height=200)

            # Collapsible sections
            with st.expander("Violations"):
                #st.write("Content for Details Section 1 goes here.")
                st.markdown(violations)

            with st.expander("Response instructions"):
                st.markdown(formatted_RI_info)

            with st.expander("Conclusion"):
                st.markdown(conclusion)

if __name__ == "__main__":
    main()
