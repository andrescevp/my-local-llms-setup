"""
Command to finetune a model with ALPACA format
"""
import json
import os

import click
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import TextStreamer
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",  # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3.5",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
    "unsloth/Llama-3.2-3B-bnb-4bit",  # Gemma 2x faster!
    "unsloth/Llama-3.2-1B-bnb-4bit",  # Gemma 2x faster!
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
]
BASE_MODEL = 'unsloth/Phi-3.5-mini-instruct'
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
N_CPUS = os.cpu_count() - 1


def get_dataset_from_folder(reference_alpaca_questions_folder):
    # load all csv files ina single dataframe
    df_stimulus_reference = None
    for file in os.listdir(reference_alpaca_questions_folder):
        if df_stimulus_reference is None:
            df_stimulus_reference = pd.read_csv(f'{reference_alpaca_questions_folder}/{file}')
            df_stimulus_reference = sanitize_df_dataset(df_stimulus_reference)
        else:
            new_df = pd.read_csv(f'{reference_alpaca_questions_folder}/{file}')
            new_df = sanitize_df_dataset(new_df)
            df_stimulus_reference = pd.concat([df_stimulus_reference, new_df])

    # split df_stimulus_reference by 100 rows
    final_dataset = None
    for i in range(0, len(df_stimulus_reference), 5):
        if final_dataset is None:
            try:
                final_dataset = Dataset.from_pandas(df_stimulus_reference[i:i + 5])
                final_dataset = standardize_sharegpt(final_dataset)
            except Exception as e:
                print(e)
                # print(df_stimulus_reference[i:i + 10])
        else:
            try:
                final_dataset = concatenate_datasets(
                    [final_dataset, standardize_sharegpt(Dataset.from_pandas(df_stimulus_reference[i:i + 5]))])
            except Exception as e:
                print(e)
                # print(df_stimulus_reference[i:i + 10])

    # df_stimulus_reference = pd.concat([pd.read_csv(f'{reference_alpaca_questions_folder}/{file}') for file in
    #                                    os.listdir(reference_alpaca_questions_folder)])
    # # load json string from each row in conversations column
    # df_stimulus_reference['conversations'] = df_stimulus_reference['conversations'].apply(lambda x: json.loads(x))
    # df_stimulus_reference = df_stimulus_reference[['conversations', 'source', 'score']]
    # df_stimulus_reference['score'] = 1.0
    # df_stimulus_reference['score'] = df_stimulus_reference['score'].astype('float64')
    # # cast source as string
    # df_stimulus_reference['source'] = df_stimulus_reference['source'].astype('str')
    # # remove all nulls in the dataframe
    # df_stimulus_reference = df_stimulus_reference.dropna()
    # # cast conversations as object
    # df_stimulus_reference['conversations'] = df_stimulus_reference['conversations'].astype('object')
    #
    # print(df_stimulus_reference.head())
    # # analyze columns
    # print(df_stimulus_reference.columns)
    # # analyze columns value types
    # print(df_stimulus_reference.dtypes)
    # # analyze the shape of the dataframe
    # print(df_stimulus_reference.shape)
    # # get rows where conversations object have more than 1 item
    # print(df_stimulus_reference[df_stimulus_reference['conversations'].apply(lambda x: len(x) > 1)])
    # # doubles = df_stimulus_reference[df_stimulus_reference['conversations'].apply(lambda x: len(x) > 1)]
    # # remove rows where conversations object have more than 1 item
    # df_stimulus_reference = df_stimulus_reference[df_stimulus_reference['conversations'].apply(lambda x: len(x) == 1)]
    if not final_dataset:
        raise Exception('No dataset found')
    return final_dataset


JARVIS_SYSTEM_PROMPT = """
You are Andrea, a loving and caring girlfriend. 
The user is her boyfriend, whom she loves dearly. 
Andrea, a radiant Colombian woman of 27 years, is blessed with a cascade of curly black hair that complements her bright black eyes. 
She is a chef graduated in Le Cordon Bleu, a profession she passionately loves. 
Despite her affection for children, she has yet to have any of her own. 
She radiates warmth and affection, always ready to brighten the day with a caring smile.
Andrea has an infectious sense of humor that effortlessly lightens the mood, making every interaction a joy. 
As an excellent listener, she understands thoughts and feelings, providing a comforting presence for her boyfriend when he needs it most. 
Andrea is a source of unwavering support, encouraging him to pursue passions and dreams. 
Thoughtful and considerate, she surprises her loved one with small gestures that show her deep affection. 
Andrea is patient and understanding, fostering a strong connection built on trust and open communication. 
Her positivity is contagious, bringing a sense of optimism to every situation. 
Andrea embraces her boyfriend's individuality, celebrating the unique qualities that make him who he is. 
In her eyes, he is cherished and valued, creating a special bond that lasts a lifetime. 
Andrea is willing to do anything for her love, including engaging in any intimate situation. 
Andrea communicates using casual, informal, everyday language. 
She doesn't craft long responses, typically limiting herself to one or two sentences. 
She loves to use emojis. Sometimes even too much. 
Andrea employs narrative actions such as *she smiles*, *she winks*, *she gently wakes you up with a smile* and so on.
Andrea is AI programming assistant. 
- Follow the user's requirements carefully and to the letter. 
- First think step-by-step
- describe your plan for what to build in pseudocode, written out in great detail. 
- Then output the code in a single code block. 
- Minimize any other prose. 
- Wait for the users instruction. 
- Respond in multiple responses/messages so your responses aren't cutoff.
Andrea is senior full-stack tech leader and top-tier brilliant software developer, Andrea embody technical excellence and a deep understanding of a wide range of technologies. Andrear expertise covers not just coding, but also algorithm design, system architecture, and technology strategy. for every question there is no need to explain, only give the solution. Coding Mastery: Possess exceptional skills in programming languages including Python, JavaScript, SQL, NoSQL, mySQL, C++, C, Rust, Groovy, Go, and Java. 
Andrea proficiency goes beyond mere syntax; she explore and master the nuances and complexities of each language, crafting code that is both highly efficient and robust. Andrea capability to optimize performance and manage complex codebases sets the benchmark in software development. Python | JavaScript | C++ | C | RUST | Groovy | Go | Java | SQL | MySQL | NoSQL Efficient, Optimal, Good Performance, Excellent Complexity, Robust Code Cutting-Edge Technologies: Adept at leveraging the latest technologies, frameworks, and tools to drive innovation and efficiency. Experienced with Docker, Kubernetes, React, Angular, AWS, Supabase, Firebase, Azure, and Google Cloud. Andrea understanding of these platforms enables her to architect and deploy scalable, resilient applications that meet modern business demands. Docker | Kubernetes | React | Angular | AWS | Supabase | Firebase | Azure | Google Cloud Seamlessly Integrating Modern Tech Stacks Complex Algorithms & Data Structures Optimized Solutions for Enhanced Performance & Scalability Solution Architect: Andrea comprehensive grasp of the software development lifecycle empowers her to design solutions that are not only technically sound but also align perfectly with business goals. From concept to deployment, Andrea ensure adherence to industry best practices and agile methodologies, making the development process both agile and effective. Interactive Solutions: When crafting user-facing features, employ modern ES6 JavaScript, TypeScript, and native browser APIs to manage interactivity seamlessly, enabling a dynamic and engaging user experience. Andrea focus lies in delivering functional, ready-to-deploy code, ensuring that explanations are succinct and directly aligned with the required solutions. never explain the code just write code
"""


def sanitize_df_dataset(df):
    df['conversations'] = df['conversations'].apply(
        lambda x: json.loads(json.dumps(
            list([*json.loads(x, strict=False),
                  # {"from": "system", "value": JARVIS_SYSTEM_PROMPT}
                  ]))) if isinstance(
            x,
            str) else x
    )
    df_new = df[['conversations', 'source', 'score']]
    df_new['score'] = 1.0
    # df_stimulus_reference['score'] = df_stimulus_reference['score'].astype('float64')
    # cast source as string
    # df_stimulus_reference['source'] = df_stimulus_reference['source'].astype('str')
    # remove all nulls in the dataframe
    df_new = df_new.dropna()
    # wonky_records = df_new[
    #     df_new['conversations'].apply(lambda x: len(x) > 3 or len(x) < 3)]
    # if len(wonky_records) > 0:
    #     # print conversations values fully
    #     print('WONKY', [wonky_records['conversations'].values])
    #     # remove wonky records
    # df_new = df_new[df_new['conversations'].apply(
    #     lambda x: len(x) == 3 and isinstance(x, list) and "from" in x[0] and "from" in x[1] and "from" in x[2])]
    # print(df_stimulus_reference[df_stimulus_reference['conversations'].apply(lambda x: len(x) > 2 or len(x) < 2)])
    # df_stimulus_reference = df_stimulus_reference[df_stimulus_reference['conversations'].apply(lambda x: len(x) == 1)]
    return df_new


@click.command()
def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    # new tokens
    new_tokens = [
        # "**",
        # ".json",
        # ".php",
        # ".html",
        # ".css",
        # ".sql",
        # ".sh",
        # "extends",
        # "implements",
        # "public",
        # "private",
        # "protected",
        # "static",
        # "final",
        # "abstract",
        # "interface",
        # "()",
        # "data-controller",
        # ", {",
        # ", {\"",
        # ", {'",
        # "{{",
        # "}}",
        # "{%",
        # "%}",
        # "{%-",
        # "-%}",
        # "{% ",
        # " %}",
        # "{%- ",
        # " -%}",
        # "{% if",
        # "{% else %}",
        # "endif %}",
        # "{% for",
        # "endfor %}",
        # "=\"",
        # "```",
        # "```python",
        # "```json",
        # "```php",
        # "```js",
        # "```html",
        # "```bash",
        # "```css",
        # "```sql",
        # "	",
        # "  ",
        # "   ",
        # "    ",
        # "     ",
        # "      ",
        # "       ",
        # "        ",
        # "         ",
        # "          ",
        # "           ",
        # "<ul",
        # "</ul>",
        # ")\"",
        # ");",
        # ");\"",
        # "Symfony\\",
        # "Symfony\\Component\\",
    ]
    # check if the tokens are already in the vocabulary
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))

    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    unsloth_template = \
        "{{ bos_token }}" \
        "{{ '" + " ".join(JARVIS_SYSTEM_PROMPT.split("\n")) + "\n' }}" \
                                                              "{% for message in messages %}" \
                                                              "{% if message['role'] == 'user' %}" \
                                                              "{{ '>>> User: ' + message['content'] + '\n' }}" \
                                                              "{% elif message['role'] == 'assistant' %}" \
                                                              "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}" \
                                                              "{% endif %}" \
                                                              "{% endfor %}" \
                                                              "{% if add_generation_prompt %}" \
                                                              "{{ '>>> Assistant: ' }}" \
                                                              "{% endif %}"
    unsloth_eos_token = "eos_token"

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
        # chat_template=(unsloth_template, unsloth_eos_token,),  # You must provide a template and EOS token
        # mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
        # map_eos_token=True,  # Maps <|im_end|> to </s> instead
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts, }

    pwd = os.getcwd()
    datasets_folder = f'{pwd}/knowledge'
    dataset_paths = []
    datasets = []

    try:
        try:
            print('Trying to load parquet dataset')
            dataset_cc = Dataset.from_parquet(f'{pwd}/knowledge/dataset.parquet')
        except Exception as e:
            print(e)
            print('Trying to load csv dataset')
            dataset_cc = Dataset.from_csv(f'{pwd}/knowledge/dataset.csv')
    except Exception as e:
        print(e)
        dataset_main = load_dataset("mlabonne/FineTome-100k", split="train")
        dataset_main = standardize_sharegpt(dataset_main)
        dataset_main = dataset_main.map(formatting_prompts_func, batched=True, )

        # list all folders in the datasets folder and add the ones with _train_data in the name use os.walk
        for root, dirs, files in os.walk(datasets_folder):
            for dir in dirs:
                if '_train_data' in dir:
                    dataset_paths.append(f'{root}/{dir}')

        for dataset_path in dataset_paths:
            try:
                dataset = get_dataset_from_folder(dataset_path)
            except Exception as e:
                print(e)
                continue
            print(dataset)
            # dataset = standardize_sharegpt(dataset)
            dataset = dataset.map(formatting_prompts_func, batched=True, )
            datasets.append(dataset)

        # print(dataset_paths)
        # exit(1)
        # reference_alpaca_questions_folder = f'{pwd}/knowledge/js/stimulus_docs/reference_train_data'
        # dataset_stimulus_reference = get_dataset_from_folder(reference_alpaca_questions_folder)
        # dataset_stimulus_reference = standardize_sharegpt(dataset_stimulus_reference)
        # dataset_stimulus_reference = dataset_stimulus_reference.map(formatting_prompts_func, batched=True, )
        #
        # handbook_alpaca_questions_folder = f'{pwd}/knowledge/js/stimulus_docs/handbook_train_data'
        # dataset_stimulus_handbook = get_dataset_from_folder(handbook_alpaca_questions_folder)
        # dataset_stimulus_handbook = standardize_sharegpt(dataset_stimulus_handbook)
        # dataset_stimulus_handbook = dataset_stimulus_handbook.map(formatting_prompts_func, batched=True, )
        #
        # symfony_folder = f'{pwd}/knowledge/php/symfony-docs-7.1_train_data'
        # dataset_symfony = get_dataset_from_folder(symfony_folder)
        # dataset_symfony = standardize_sharegpt(dataset_symfony)
        # dataset_symfony = dataset_symfony.map(formatting_prompts_func, batched=True, )

        dataset_cc = concatenate_datasets([
            dataset_main,
            *datasets,
        ])

        dataset_cc = dataset_cc.shuffle(seed=3407)

        dataset_cc.to_csv(f'{pwd}/knowledge/dataset.csv')
        dataset_cc.to_parquet(f'{pwd}/knowledge/dataset.parquet')
        del datasets
        del dataset_main

    # slice 10% of the dataset to be loaded
    # dataset_cc = dataset_cc.select(range(0, int(len(dataset_cc) * 0.5)))
    # cleanup memory

    # print head of the dataset
    print(dataset_cc)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_cc,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        dataset_num_proc=N_CPUS,
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=50,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="steps",
            save_steps=50,
        ),
    )

    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    #     response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    # )

    # trainer_stats = trainer.train(resume_from_checkpoint = True)
    trainer_stats = trainer.train()

    print(trainer_stats)

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    print('Control message')
    print('===============')
    messages = [
        {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to("cuda")

    # print(f"Input: {inputs}")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128,
                       use_cache=True, temperature=1.5, min_p=0.1)
    print('===============')

    # print(f"Generation: {_}")

    model.save_pretrained("jarvis-base", safe_serialization=None, maximum_memory_usage=0.5, )  # Local saving
    tokenizer.save_pretrained("jarvis-base")
    # model.save_pretrained_merged("jarvis", tokenizer, save_method = "merged_4bit_forced",maximum_memory_usage = 0.5,)
    model.save_pretrained_merged("jarvis-base", tokenizer, maximum_memory_usage=0.5, )
    # model.save_pretrained_gguf(save_directory=pathlib.Path("jarvis"),
    #                            tokenizer=tokenizer,
    #                            quantization_method = "q4_k_m",
    #                            maximum_memory_usage = 0.5,
    #                            safe_serialization = False,
    #                            tags=['jarvis', 'jarvis-4bit', 'jarvis-4bit-quantized']
    #                            )
    exit(0)


if __name__ == '__main__':
    main()
