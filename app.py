import pandas as pd
import pdfplumber
import re
import numpy as np
import streamlit as st
from thefuzz import fuzz

# Place PDF path here
PATH = "genius_pdf.pdf"

with pdfplumber.open(PATH) as pdf:
    pages = pdf.pages
    page_no = []
    text_page = []
    chap = []
    for i, pg in enumerate(pages):
        # Get page no and text
        page = pdf.pages[i]
        text = page.extract_text()
        # Fix space
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        # Fix multiple spaces
        text = " ".join(text.split())
        # Get real page
        get_chap = re.findall(r"บทที่ (\d+)", text)
        # append lists
        chap.append(get_chap)
        text_page.append(text)
        page_no.append(i)

# Combine lists
df = pd.DataFrame(
    list(zip(page_no, chap, text_page,)), columns=["Page No", "chapter", "Text"]
)
# Get Clean chapter no.
df["chapter"] = df["chapter"].str[0]
df["chapter"] = df["chapter"].fillna(method="ffill")
df["chapter"] = np.where(df.index < 21, 0, df["chapter"])


def get_ratio(word, target):
    return fuzz.token_set_ratio(word, target)


target_word = st.text_input("ใส่ข้อความที่นี่:", "ใครสร้าง Microsoft")

df_chap_only = df.query("chapter != 0")
df_chap_only["Score"] = df_chap_only.apply(
    lambda row: get_ratio(row["Text"], target_word), axis=1
)
final = df_chap_only[["Page No", "chapter", "Text", "Score"]].sort_values(
    "Score", ascending=False
)

st.dataframe(data=final[["chapter", "Text", "Score"]].head(5), width=None, height=None)
st.table(final[["chapter", "Text", "Score"]].head(5))


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


csv = convert_df(final.head(10))
st.download_button(
    label="ดาวโหลดผลการค้นหา (CSV)", data=csv, file_name="large_df.csv", mime="text/csv",
)
