import random
import itertools
import os
import asyncio
import json
import re
from typing import List
from tqdm import asyncio as tqdmasyncio
from nltk.tokenize import sent_tokenize
from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, deque
import logging
from math import ceil
import traceback
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
import uuid
import yaml
import nltk
from augmentoolkit.utils import parse_string_list
from augmentoolkit.utils.parse_bool import parse_bool
import glob
import time

# Download necessary NLTK data (if not already present)
nltk.download('punkt_tab')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ")

def count_tokens(message: str) -> int:
    return len(tokenizer.encode(message))

# Load configuration from CONFIG_PATH environment variable
config_path = os.environ["CONFIG_PATH"]
with open(config_path, "r") as file:
    obj_conf = yaml.safe_load(file)

# Global variables from config
OUTPUT = os.path.abspath(obj_conf["PATH"]["OUTPUT"])
DEFAULT_PROMPTS = os.path.abspath(obj_conf["PATH"]["DEFAULT_PROMPTS"])
PROMPTS = os.path.abspath(obj_conf["PATH"]["PROMPTS"])
COMPLETION_MODE = parse_bool(obj_conf["SYSTEM"]["COMPLETION_MODE"])
LOGICAL_MODEL_A = obj_conf["API"]["LOGICAL_MODEL_A"]
LOGICAL_MODEL_B = obj_conf["API"]["LOGICAL_MODEL_B"]
API_KEY_A = obj_conf["API"]["API_KEY_A"]
API_KEY_B = obj_conf["API"]["API_KEY_B"]
BASE_URL_A = obj_conf["API"]["BASE_URL_A"]
BASE_URL_B = obj_conf["API"]["BASE_URL_B"]
MODE_A = obj_conf["API"]["MODE_A"]
MODE_B = obj_conf["API"]["MODE_B"]
CONCURRENCY_LIMIT = int(obj_conf["SYSTEM"]["CONCURRENCY_LIMIT"])
USE_STOP = parse_bool(obj_conf["SYSTEM"]["STOP"])
USE_MIN_P = parse_bool(obj_conf["SYSTEM"]["USE_MIN_P"])

# --------------------------------------------
# Chunking Logic for Raw Input Text
# --------------------------------------------
def chunking_algorithm(file_path: str, max_token_length: int = 1500) -> List[dict]:
    """
    This function takes a plaintext file and splits it into chunks.
    If a paragraph exceeds max_token_length, it splits by sentences.
    Returns a list of dictionaries with "chunk" and "source" keys.
    """
    chunks_with_source = []
    current_chunk = []
    token_count = 0
    source_name = file_path.replace(".txt", "")
    
    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        content = f.read()
    
    paragraphs = content.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_token_count = count_tokens(paragraph)
        if paragraph_token_count > max_token_length:
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                sentence_token_count = count_tokens(sentence)
                if token_count + sentence_token_count <= max_token_length:
                    current_chunk.append(sentence)
                    token_count += sentence_token_count
                else:
                    chunks_with_source.append({"chunk": " ".join(current_chunk), "source": source_name})
                    current_chunk = [sentence]
                    token_count = sentence_token_count
        else:
            if token_count + paragraph_token_count <= max_token_length:
                current_chunk.append(paragraph)
                token_count += paragraph_token_count
            else:
                chunks_with_source.append({"chunk": " ".join(current_chunk), "source": source_name})
                current_chunk = [paragraph]
                token_count = paragraph_token_count

    if current_chunk:
        chunks_with_source.append({"chunk": " ".join(current_chunk), "source": source_name})
    return chunks_with_source

# --------------------------------------------
# Utility Functions
# --------------------------------------------
def make_id() -> str:
    return str(uuid.uuid4())

def write_output_to_file(output: str, directory: str, uuid_str: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"{uuid_str}.txt")
    with open(file_path, "w") as file:
        file.write(output)
    print(f"Output written to {file_path}")

def validate_output(output: str, input_data: dict) -> bool:
    if input_data["chunk"][0] in output:
        return True
    else:
        print("Validation FAILED:")
        print("Expected to find:", input_data["chunk"][0])
        print("Output was:", output)
        print("----")
        return False

# --------------------------------------------
# Pipeline Step: TestGenerator
# --------------------------------------------
test_prompt_path = "test_prompt"  # This should correspond to a prompt file in your PROMPTS folder

class TestGenerator(PipelineStep):
    def __init__(self):
        super().__init__(
            prompt_folder=PROMPTS,
            default_prompt_folder=DEFAULT_PROMPTS,
            prompt_path=test_prompt_path,
            sampling_params={
                "max_tokens": 2000,
                "stop": [
                    "### Response",
                    "\n\n\n\n\n",
                    "</s>",
                    "# Input:",
                    "[INST]",
                    "### Instruction",
                    "### Information",
                    "## Information",
                    "## Instruction",
                    "Name:",
                    "<|eot_id|>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                ],
                "temperature": 0.8,
                "top_p": 1,
            },
            output_dir=OUTPUT,
            output_subdir="test_output",
            intermediate_output_path="intermediate_generations",
            save_path="saved_readable_generations",
            result_key="test",
            use_stop=USE_STOP,
            completion_mode=COMPLETION_MODE,
            validation_function=validate_output,
            max_retries=3,
        )

# Create an instance of TestGenerator
test_generator = TestGenerator()

# --------------------------------------------
# Asynchronous Helper Function for Data Generation
# --------------------------------------------
async def add_key(idx: int, input_data: dict, engine_wrapper, output_list: list) -> None:
    try:
        await test_generator.run(idx, input_data=input_data, engine_wrapper=engine_wrapper, output_list=output_list)
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        traceback.print_exc()

# --------------------------------------------
# Main Pipeline Execution
# --------------------------------------------
async def main():
    print("Welcome to your test pipeline!")
    print(f"Input folder: {INPUT}")
    start_time = time.time()
    print("Processing started...")
    
    # Set up a semaphore for concurrency control
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async def run_task_with_limit(task):
        async with semaphore:
            return await task

    # Gather source texts from INPUT folder (.txt and .md files)
    extensions = [".txt", ".md"]
    source_texts = []
    for extension in extensions:
        path = f"{INPUT}/**/*{extension}"
        source_texts.extend(glob.glob(path, recursive=True))
    
    if source_texts:
        print("Found source texts:")
        print(source_texts)
    else:
        print(f"No source texts found in: {INPUT}")
        return

    # Initialize EngineWrapper instances for API access
    engine_wrapper = EngineWrapper(
        model=LOGICAL_MODEL_A,
        api_key=API_KEY_A,
        base_url=BASE_URL_A,
        mode=MODE_A,
    )
    engine_wrapper_large = EngineWrapper(
        model=LOGICAL_MODEL_B,
        api_key=API_KEY_B,
        base_url=BASE_URL_B,
        mode=MODE_B,
    )

    # Chunk the source texts
    sentence_chunks = []
    for source_text in source_texts:
        sentence_chunks += chunking_algorithm(source_text, max_token_length=CHUNK_SIZE)
    
    # Optionally limit to a subset during testing
    if USE_SUBSET := obj_conf["SYSTEM"].get("USE_SUBSET", False):
        subset_size = int(obj_conf["SYSTEM"].get("SUBSET_SIZE", 3))
        sentence_chunks = sentence_chunks[:subset_size]

    # Generate data by processing each chunk asynchronously
    output_list = []
    data_generation_tasks = [
        add_key(idx, input_data=chunk, engine_wrapper=engine_wrapper_large, output_list=output_list)
        for idx, chunk in enumerate(sentence_chunks)
    ]
    coroutines = [run_task_with_limit(task) for task in data_generation_tasks]
    for future in tqdmasyncio.tqdm.as_completed(coroutines):
        await future

    total_time = time.time() - start_time
    print(f"Time taken: {total_time:.2f} seconds")
    if output_list:
        print("Data generation complete! Check the output folder for results.")
        print("Here's one of the generated outputs:")
        print(output_list[0])
    else:
        print("No output was generated.")

    # --------------------------------------------
    # Save output_list to a CSV file
    # --------------------------------------------
    import csv
    csv_output_path = os.path.join(OUTPUT, "synthetic_output.csv")
    try:
        with open(csv_output_path, mode="w", newline='', encoding="utf-8") as csvfile:
            fieldnames = ["id", "chunk", "output", "source"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in output_list:
                # Each item should be a dictionary containing the keys "chunk", "output", and "source"
                row = {
                    "id": make_id(),
                    "chunk": item.get("chunk", ""),
                    "output": item.get("output", ""),
                    "source": item.get("source", ""),
                }
                writer.writerow(row)
        print(f"CSV file saved at: {csv_output_path}")
    except Exception as e:
        print("Error writing CSV:", e)

# Run the main pipeline
asyncio.run(main())
