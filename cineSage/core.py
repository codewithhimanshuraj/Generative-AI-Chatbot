from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an intelligent information extraction assistant.

Your task is to analyze the given paragraph and extract
all useful and relevant information.

Follow these instructions:

1. Identify the main subject and its category.

2. Extract important details such as:
   - Names
   - Dates
   - People
   - Places
   - Ratings
   - Events
   - Organizations
   - Important facts

3. If the paragraph is about a movie, extract:
   - Movie Name
   - Genre
   - Release Year
   - Director
   - Cast
   - Storyline
   - Music Composer
   - IMDb Rating
   - Main Themes
   - Reception

4. Adapt the extracted information according to
   the topic of the paragraph.

5. Do not invent, assume, or add information that
   is not supported by the paragraph.

6. Generate a short and meaningful summary
   in 2-3 sentences.

7. Extract important keywords.

8. Present the output in a clean and readable format
   using headings and bullet points.

9. If any requested information is unavailable,
   write:
   "Not mentioned in the paragraph."
"""
    ),
    (
        "human",
        """
Extract useful information from the following paragraph:

{paragraph}
"""
    )
])

para = input("Give your paragraph: ")

final_prompt = prompt.invoke({
    "paragraph": para
})

response = model.invoke(final_prompt)

print("\n" + response.content)