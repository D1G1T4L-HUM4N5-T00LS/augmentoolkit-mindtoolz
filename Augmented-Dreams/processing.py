import random
import traceback
import glob
import asyncio
import logging
import os
import sys
import time
import yaml

# Import required modules from augmentoolkit and our boilerplate
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from augmentoolkit.utils.write_output_to_file import write_output_to_file
from BOILERPLATE_TO_MAKE_YOUR_OWN_PIPELINE.steps import (
    API_KEY_A, API_KEY_B, BASE_URL_A, BASE_URL_B, CONCURRENCY_LIMIT,
    LOGICAL_MODEL_A, LOGICAL_MODEL_B, MODE_A, MODE_B, chunking_algorithm,
    count_tokens, make_id
)

# Import necessary NLP and progress packages
import nltk
from tqdm import asyncio as tqdmasyncio

# Download necessary NLTK data (if not already present)
nltk.download('punkt')

# Load configuration
config_path = os.environ["CONFIG_PATH"]
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

WORK_IN_PHASES = bool(config["PHASES"]["WORK_IN_PHASES"])
PHASE_INDEX = int(config["PHASES"]["PHASE_INDEX"])
USE_SUBSET = bool(config["SYSTEM"]["USE_SUBSET"])
SUBSET_SIZE = int(config["SYSTEM"]["SUBSET_SIZE"])
CHUNK_SIZE = int(config["SYSTEM"]["CHUNK_SIZE"])
INPUT = config["PATH"]["INPUT"]

# -----------------------------------------------------
# Define a DreamGenerator class for synthetic dream data
# -----------------------------------------------------
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep

class DreamGenerator(PipelineStep):
    def __init__(self):
        super().__init__(
            prompt_folder=config["PATH"]["PROMPTS"],      # Use the prompts folder from config
            default_prompt_folder=config["PATH"]["DEFAULT_PROMPTS"],  # Default prompts folder
            prompt_path="dream_prompt.yaml",              # This file should contain your dreaming prompt
            sampling_params={
                "max_tokens": 2000,
                "temperature": 0.8,
                "top_p": 1,
                # You can add more sampling parameters if needed
            },
            output_dir=os.path.abspath(config["PATH"]["OUTPUT"]),
            output_subdir="dream_output",                # New subdirectory for dream dataset
            intermediate_output_path="intermediate_dream_generations",
            save_path="saved_dream_generations",
            result_key="dream",
            use_stop=bool(config["SYSTEM"]["STOP"]),
            completion_mode=bool(config["SYSTEM"]["COMPLETION_MODE"]),
            validation_function=None,  # Optionally, add a validation function
            max_retries=3,
        )

# Create a singleton instance of DreamGenerator
dream_generator = DreamGenerator()

# -----------------------------------------------------
# Define an asynchronous helper to use DreamGenerator
# -----------------------------------------------------
async def add_key_for_dream(idx, input_data, engine_wrapper, output_list):
    """
    Uses the dream_generator instance to process the input_data.
    """
    try:
        # The dream_generator.run method is assumed to generate output based on the dream prompt.
        await dream_generator.run(idx, input_data=input_data, engine_wrapper=engine_wrapper, output_list=output_list)
    except Exception as e:
        print(f"Error in add_key_for_dream for index {idx}: {e}")
        traceback.print_exc()

# -----------------------------------------------------
# Main Pipeline to Generate Synthetic Dream Dataset
# -----------------------------------------------------
async def main():
    print("Welcome to your Dream Dataset Generation Pipeline!")
    print(f"Input folder: {INPUT}")
    start_time = time.time()
    print("Processing has begun...")

    # Set up a semaphore to respect concurrency limits.
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async def run_task_with_limit(task):
        async with semaphore:
            return await task

    # Load source texts from the input folder (looking for .txt and .md files)
    extensions = [".txt", ".md"]
    source_texts = []
    for extension in extensions:
        path = f"{INPUT}/**/*{extension}"
        source_texts.extend(glob.glob(path, recursive=True))

    if source_texts:
        print("Found the following source texts:")
        print(source_texts)
    else:
        print(f"No source texts found in: {INPUT}")
        return

    # Initialize the EngineWrappers for API access (use LOGICAL_MODEL_A for one and LOGICAL_MODEL_B for large-scale)
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

    # Chunk the source texts using the provided chunking_algorithm
    sentence_chunks = []
    for source_text in source_texts:
        sentence_chunks += chunking_algorithm(source_text, max_token_length=CHUNK_SIZE)

    # Optionally, if USE_SUBSET is enabled, limit the number of chunks processed.
    if USE_SUBSET:
        sentence_chunks = sentence_chunks[:SUBSET_SIZE]

    # Generate the dream data using the dream generator pipeline step.
    output_list = []
    data_generations_tasks = [
        add_key_for_dream(idx, input_data=chunk, engine_wrapper=engine_wrapper_large, output_list=output_list)
        for idx, chunk in enumerate(sentence_chunks)
    ]
    coroutines = [run_task_with_limit(task) for task in data_generations_tasks]
    for future in tqdmasyncio.tqdm.as_completed(coroutines):
        await future

    total_time = time.time() - start_time
    print(f"Time taken: {total_time:.2f} seconds")
    if output_list:
        print("You generated some dream data! Check the output folder for the results.")
        print("Here's one of the results:")
        print(output_list[0])
    else:
        print("No output was generated.")

# Run the main pipeline
asyncio.run(main())
