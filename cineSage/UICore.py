import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

# Environment variables लोड करें (सुनिश्चित करें कि आपके .env फ़ाइल में GROQ_API_KEY मौजूद है)
load_dotenv()

# 1. आपके द्वारा चुना गया मॉडल ID यहाँ सेट है।
# (नोट: लाइन की शुरुआत में कोई भी फालतू स्पेस नहीं होना चाहिए)
model = ChatGroq(
    model="openai/gpt-oss-120b"
)

# 2. डेटा एक्सट्रैक्शन के लिए Pydantic Schema
class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    release_year: Optional[int] = Field(description="The year the movie was released")
    genre: List[str] = Field(description="List of genres")
    director: Optional[str] = Field(description="The director of the movie")
    cast: List[str] = Field(description="List of main cast members")
    rating: Optional[float] = Field(description="The rating of the movie")
    summary: str = Field(description="A 2-3 sentence summary of the movie plot")

# 3. Output Parser को क्लास के बाहर रखें ताकि इसे सीधे इस्तेमाल किया जा सके
parser = PydanticOutputParser(pydantic_object=Movie)

# 4. Chat Prompt जिसमें फ़ॉर्मेटिंग निर्देश ({format_instructions}) शामिल हैं
prompt = ChatPromptTemplate.from_messages([
    ('system', """
You are an expert information extraction assistant. 
Extract movie information from the provided text paragraph.
You must strictly follow the formatting instructions provided below.

{format_instructions}
"""),
    ("human", "{paragraph}")
])

# ---------------- STREAMLIT UI ----------------

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Information Extractor")
st.write("Paste a text paragraph about a movie to extract structured data.")

paragraph = st.text_area(
    "Enter your paragraph",
    height=200,
    placeholder="Example: In 2010, Christopher Nolan directed Inception, a sci-fi thriller starring Leonardo DiCaprio..."
)

if st.button("Extract Data", use_container_width=True):
    if paragraph.strip():
        with st.spinner("Analyzing text and extracting data..."):
            try:
                # प्रॉम्प्ट टेम्पलेट में पैरामीटर पास करें
                final_prompt = prompt.invoke({
                    "paragraph": paragraph,
                    "format_instructions": parser.get_format_instructions()
                })

                # LLM मॉडल को कॉल करें
                response = model.invoke(final_prompt)
                
                # LLM से मिले टेक्स्ट रिस्पॉन्स को Pydantic ऑब्जेक्ट (JSON) में बदलें
                structured_data = parser.parse(response.content)

                # स्क्रीन पर डेटा को साफ़-सुथरे तरीके से दिखाएं
                st.subheader("✅ Extracted Data")
                
                st.write(f"**Title:** {structured_data.title}")
                st.write(f"**Release Year:** {structured_data.release_year if structured_data.release_year else 'N/A'}")
                st.write(f"**Director:** {structured_data.director if structured_data.director else 'N/A'}")
                st.write(f"**Rating:** {structured_data.rating if structured_data.rating else 'N/A'}")
                
                st.write("**Genre:**")
                st.write(", ".join(structured_data.genre) if structured_data.genre else "None found")
                
                st.write("**Cast:**")
                st.write(", ".join(structured_data.cast) if structured_data.cast else "None found")
                
                st.write(f"**Plot Summary:** {structured_data.summary}")

            except Exception as e:
                st.error(f"An error occurred during parsing: {e}")
                # अगर Pydantic पार्सिंग फ़ेल हो जाती है, तो बैकअप के लिए रॉ (Raw) टेक्स्ट दिखाएं
                if 'response' in locals():
                    st.text_area("Raw Response Content:", response.content, height=150)
    else:
        st.warning("Please enter a paragraph.")
