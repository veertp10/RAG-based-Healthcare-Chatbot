from flask import Flask, render_template, request,jsonify
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import random
import json
import nltk
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import json
import pandas as pd
import numpy as np
from flask_cors import CORS
import warnings

import pickle
app = Flask(__name__)
CORS(app)  # Add this line

warnings.filterwarnings("ignore", category=UserWarning)



with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Read disease names from disease.txt
with open('data/diseases.txt', 'r') as file:
    disease_names = [line.strip().lower() for line in file]
disease_tokens = [tokenize(name) for name in disease_names]

# Convert disease_tokens into the format expected by train_supervised
tagged_disease_tokens = [[(word, 'disease') for word in token_seq] for token_seq in disease_tokens]

# Train an HMM model using disease tokens
trainer = nltk.tag.hmm.HiddenMarkovModelTrainer()
hmm_model = trainer.train(tagged_disease_tokens)

# Tokenize disease names
disease_tokens = [tokenize(name) for name in disease_names]

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

with open('ExtraTrees', 'rb') as f:
    disease_model = pickle.load(f)

# Define the diseases
diseases = [
    '(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy',
    'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
    'Common Cold', 'Dengue', 'Diabetes', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
    'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C',
    'Hepatitis D', 'Hepatitis E', 'Hypertension', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism',
    'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)',
    'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid',
    'Urinary tract infection', 'Varicose veins', 'hepatitis A'
]

# Define the symptoms
symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
            'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
            'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
            'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
            'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine',
            'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain',
            'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
            'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
            'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
            'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region',
            'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
            'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
            'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
            'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
            'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
            'loss_of_smell', 'bladder_discomfort', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
            'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
            'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'watering_from_eyes', 'increased_appetite',
            'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
            'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
            'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum',
            'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
            'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
            'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

desc = pd.read_csv("data/symptom_Description.csv")
prec = pd.read_csv("data/symptom_precaution.csv")


def chat(sentence):
    bot_name = "aarogya"
    sentence1 = tokenize(sentence)
    X = bag_of_words(sentence1, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                if tag == "disease":
                    response = user_input(sentence)
                    return response["output_text"]
                else:
                    response = random.choice(intent['responses'])
                    return response
    else:
        response = "I do not understand..."
        return response

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    print(len(chunks))
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.0)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def hmm(sentence):
    sentence_tokens = tokenize(sentence.lower())
    

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 20})
    docs = retriever.get_relevant_documents(user_question)
    pretty_print_docs(docs)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    return response

@app.route("/")
def index():
    return render_template('chat.html')


@app.route('/disease', methods=["GET"])
def home():
    return app.send_static_file('index.html')

@app.route("/get", methods=["GET", "POST"])
def get_response():
    user_question = request.form["msg"]
    response = chat(user_question)
    return response

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    print(data)

    # Create a list of zeros
    features = [0] * 218

    # Set the corresponding indices to 1 for the symptoms present in the data
    for symptom in data:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            features[index] = 1

    # Make prediction using the model
    proba = disease_model.predict_proba([features])

    # Get the indices and probabilities of the top 5 classes
    top5_idx = np.argsort(proba[0])[-5:][::-1]
    top5_proba = np.sort(proba[0])[-5:][::-1]

    # Get the names of the top 5 diseases
    top5_diseases = [diseases[i] for i in top5_idx]

    # Prepare the response
    response = []
    for i in range(5):
        disease = top5_diseases[i]
        probability = top5_proba[i]

        # Get the disease description
        disp = desc[desc['Disease'] == disease].values[0][1] if disease in desc["Disease"].unique() else "No description available"

        # Get the precautions
        precautions = []
        if disease in prec["Disease"].unique():
            c = np.where(prec['Disease'] == disease)[0][0]
            for j in range(1, len(prec.iloc[c])):
                precaution = prec.iloc[c, j]
                if not pd.isna(precaution) and precaution != '':
                    precautions.append(precaution)

        # Add the disease prediction to the response
        response.append({
            'disease': disease,
            'probability': float(probability),
            'description': disp,
            'precautions': precautions
        })

    # Send back to the client
    print(response)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
