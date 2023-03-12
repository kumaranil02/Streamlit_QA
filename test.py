import streamlit as st
import PyPDF2
from PIL import Image

img = Image.open("streamlit.jpg")
# display image using streamlit
# width is used to set the width of an image
st.image(img,width=600)

st.title('Welcome to QA Model')

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
st.cache(show_spinner=False)

reader = ''
uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    reader = PyPDF2.PdfReader(uploaded_file)
    context = reader.pages[0].extract_text()
    st.text("PDF Content \n")
    st.text(context)

QA_input = {}
question = st.text_input("Enter Your question here", "Type Here ...")
if(st.button('Submit')):
	QA_input['context'] = context
	QA_input['question'] = question.title()
	res = nlp(QA_input)
	st.success(res['answer'])