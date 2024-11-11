"""
Command line that get a file and generate a training data set with ALPACA format
"""
import json
import logging
import os
from multiprocessing import Pool
import asyncclick as click
import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  project=os.getenv('OPENAI_PROJECT'),
  api_key=os.getenv('OPENAI_API_KEY'),
)

prompt = """
You are an AI assistant designed to process documents and generate question-answer pairs.
You are an expert in development with deep knowledge in various programming languages, tools, and technologies.
Your duty is to read a provided question or text, repe
Your answers always have to be in Llama3.1 Format independently of the input format and the question, The LLama3.1 Format is JSONL, that means that each line represents a JSON array, each line must a valid JSON, you must evaluate internally if is valid and scape properly all the strings.  
You must generate at least 150 questions. Make sure you provide a variety of questions and answers that covers fully the document. To reach 150 questions, you can ask multiple questions about the same topic with variations.
If the topic of the text is development, questions of creation or update/revision of current components and answers can contain code snippets.
30% of questions must be a chain of messages in a single row so it can emulate a conversation.
30% of the questions must target deep reasoning of the question.
Verify the quality of the questions and answers, make sure they are relevant and accurate.
You must always follow the instructions and meet your duties.
Your answer must be always following the example answer format.
Example question: Create Llama3.1 Format from this document: [document content possible markdown]
Example answer:
[{"from":"human","value":"What is the main purpose of Docker in DevOps?"},{"from":"gpt","value":"Docker is used in DevOps to containerize applications, enabling developers to package software and its dependencies into standardized units called containers. This ensures consistent behavior across different environments, making it easier to develop, test, and deploy applications."}]|||
[{"from":"human","value":"How do virtual machines differ from containers?"},{"from":"gpt","value":"Virtual machines (VMs) provide full OS virtualization, meaning each VM runs its own operating system on a hypervisor. Containers, on the other hand, share the host's OS kernel and are more lightweight, as they only virtualize the application layer, making them faster and more efficient than VMs."}]|||
"""
cpu_n = os.cpu_count()
@click.command()
@click.option('--source', '-s', help='Source folder', required=True)
async def main(source):
    print(source)
    # get all files in folder
    files = os.listdir(source)
    # # for each file
    data = []
    for file in files:
        # read file
        with open(f"{source}/{file}", "r") as f:
            content = f.read()

        duplicates = find_duplicates(content)
        print(f'Duplicates: {duplicates}')


def find_duplicates(text):
    # Convert the text to lowercase and split it into words
    words = text.lower().split()

    # Create an empty dictionary to store word counts
    word_counts = {}

    # Count the occurrences of each word
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    # Find and print duplicated words
    duplicates = [word for word, count in word_counts.items() if count > 2]
    return duplicates



if __name__ == '__main__':
    main()