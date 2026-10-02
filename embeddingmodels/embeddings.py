from langchain_openai import OPENAIEMBEDDINGS

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-large'
)

vector=embeddings.embed_query("You are going to learn ai")