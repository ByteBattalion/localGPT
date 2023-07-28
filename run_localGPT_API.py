import logging
import os
import shutil
import subprocess

import torch
from auto_gptq import AutoGPTQForCausalLM
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from collections import defaultdict

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
from werkzeug.utils import secure_filename

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY

qa_dict = defaultdict(lambda: None)
retriever_dict = defaultdict(lambda: None)

DEVICE_TYPE = "cuda"
SHOW_SOURCES = True
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

# load the LLM for generating Natural Language responses
def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """

    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        # The code supports all huggingface models that ends with GPTQ
        # and have some variation of .no-act.order or .safetensors in their HF repo.
        print("Using AutoGPTQForCausalLM for quantized models")

        if ".safetensors" in model_basename:
            # Remove the ".safetensors" ending if present
            model_basename = model_basename.replace(".safetensors", "")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        logging.info("Tokenizer loaded")

        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=False,
            quantize_config=None,
        )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin file in their HF repo.
        print("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        model.tie_weights()
    else:
        print("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4096,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


# for HF models
# model_id = "TheBloke/vicuna-7B-1.1-HF"
# model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# model_id = "TheBloke/guanaco-7B-HF"
# model_id = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM.
# Using STransformers alongside will 100% create OOM on 24GB cards.
# LLM = load_model(device_type=DEVICE_TYPE, model_id=model_id)

# for GPTQ (quantized) models
# model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
# model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
# model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
# model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors"
# Requires ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
# model_id = "TheBloke/wizardLM-7B-GPTQ"
# model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
# ----------------
# model_id = "TheBloke/WizardLM-13B-V1.0-Uncensored-GPTQ"
# model_basename = "wizardlm-13b-v1.0-uncensored-GPTQ-4bit-128g.no-act.order.safetensors"
# ----------------
model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
LLM = load_model(device_type=DEVICE_TYPE, model_id=model_id, model_basename=model_basename)

app = Flask(__name__)
CORS(app)

@app.route("/api/delete_source", methods=["GET"])
def delete_source_route():
    folder_name = "SOURCE_DOCUMENTS"

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    return jsonify({"message": f"Folder '{folder_name}' successfully deleted and recreated."})


@app.route("/api/save_document", methods=["GET", "POST"])
def save_document_route():
    user_id = request.form.get("user_id")

    if "document" not in request.files:
        return "No document part", 400
    file = request.files["document"]
    if file.filename == "":
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        folder_path = os.path.join(get_user_directory(user_id), "documents")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        elif not True:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        file.save(file_path)
        return "File saved successfully", 200


@app.route("/api/run_ingest", methods=["GET"])
def run_ingest_route():
    user_id = request.args.get("user_id")
    print(f"--------------------------{user_id}--------------------------")

    if user_id:
        user_directory = get_user_directory(user_id)

        # Check if the user has a documents directory, else return error
        if not os.path.exists(os.path.join(user_directory, "documents")):
            return "No document found for this user", 400

        vector_db = os.path.join(user_directory, "vector_db")
        source_directory = os.path.join(user_directory, "documents")

        try:
            # If you want to clear the existing vector DB before ingesting new documents, uncomment the following lines
            if os.path.exists(vector_db):
                try:
                    shutil.rmtree(vector_db)
                except OSError as e:
                    print(f"Error: {e.filename} - {e.strerror}.")
            else:
                 print("The directory does not exist")

            run_langest_commands = ["python", "ingest.py", "--vector_db", vector_db, "--source_directory", source_directory]
            if DEVICE_TYPE == "cpu":
                run_langest_commands.append("--device_type")
                run_langest_commands.append(DEVICE_TYPE)

            result = subprocess.run(run_langest_commands, capture_output=True)
            if result.returncode != 0:
                return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500

            if retriever_dict[user_id]:
                del retriever_dict[user_id]
                print("deleted retriever with user id")
            
            if qa_dict[user_id]:
                del qa_dict[user_id]
                print("deleted qa with user id")

            return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
        except Exception as e:
            return f"Error occurred: {str(e)}", 500
    else:
        return "No user id received", 400
    

@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    user_prompt = request.form.get("user_prompt")
    user_id = request.form.get("user_id")
    
    if user_prompt and user_id:
        user_directory = os.path.join(get_user_directory(user_id), "vector_db")

        # Check if the user has a vector DB, else return error
        if not os.path.exists(user_directory):
            return "No document found for this user", 400

        # Check if a retriever instance for this user already exists, if not create one
        if not retriever_dict[user_id]:
            
            local_settings = CHROMA_SETTINGS.copy()
            local_settings.persist_directory = user_directory

            DB = Chroma(
                persist_directory=user_directory,
                embedding_function=EMBEDDINGS,
                client_settings=local_settings,
            )
            RETRIEVER = DB.as_retriever()
            retriever_dict[user_id] = RETRIEVER

        # Check if a QA instance for this user already exists, if not create one
        if not qa_dict[user_id]:
            qa_dict[user_id] = RetrievalQA.from_chain_type(
                llm=LLM, chain_type="stuff", retriever=retriever_dict[user_id], return_source_documents=SHOW_SOURCES
            )

        # Now use the QA instance from the dictionary
        QA = qa_dict[user_id]

        res = QA(user_prompt)
        answer, docs = res["result"], res["source_documents"]

        prompt_response_dict = {
            "Prompt": user_prompt,
            "Answer": answer,
        }

        prompt_response_dict["Sources"] = []
        for document in docs:
            prompt_response_dict["Sources"].append(
                (os.path.basename(str(document.metadata["source"])), str(document.page_content))
            )

        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400

def get_user_directory(user_id):
    base_directory = f"user_directory/{user_id}"
    return base_directory

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=False, port=5110)
