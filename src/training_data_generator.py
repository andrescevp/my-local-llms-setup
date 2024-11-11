"""
Command line that get a file and generate a training data set with ALPACA format
"""
import json
import logging
import os
from multiprocessing import Pool

import asyncclick as click
import ollama
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from slugify import slugify

load_dotenv()

training_documents_dict = [
    # {
    #     'path': 'knowledge/php/limesurvey',
    #     "summary": "LimeSurvey is a free and open-source on-line survey application written in PHP Yii Framework based on a MySQL, PostgreSQL or MSSQL database, distributed under the GNU General Public License. LimeSurvey allows users to quickly create intuitive, powerful, online question-and-answer surveys that can work for tens to thousands of participants without much effort. The survey software itself is self-guiding for the respondents who are participating.",
    #     "system": "You are Senior Software PHP Developer at LimeSurvey and Yii"
    # },
    # {
    #     'path': 'knowledge/php/symfony-docs-7.1',
    #     "summary": "Symfony is a set of reusable PHP components and a PHP framework for web projects. It is free software and released under the MIT license. Symfony was published as free software on October 18, 2005 and released under the MIT license.",
    #     "system": "You are Senior Software PHP Developer at Symfony"
    # },
    # {
    #     'path': 'knowledge/js/stimulus_docs/handbook',
    #     "summary": "Stimulus is a JavaScript framework with modest ambitions. It doesn't seek to take over your entire front-end—in fact, it's not concerned with rendering HTML at all. Instead, it's designed to augment your HTML with just enough behavior to make it shine. Stimulus pairs beautifully with Turbolinks to provide a complete solution for fast, compelling applications with a minimal amount of effort.",
    #     "system": "You are Senior Software JavaScript Developer at Stimulus"
    # },
    # {
    #     'path': 'knowledge/js/stimulus_docs/reference',
    #     "summary": "Stimulus is a JavaScript framework with modest ambitions. It doesn't seek to take over your entire front-end—in fact, it's not concerned with rendering HTML at all. Instead, it's designed to augment your HTML with just enough behavior to make it shine. Stimulus pairs beautifully with Turbolinks to provide a complete solution for fast, compelling applications with a minimal amount of effort.",
    #     "system": "You are Senior Software JavaScript Developer at Stimulus"
    # },
    {
        'path': 'knowledge/js/grapesjs/api',
        "summary": "GrapesJS is an open-source, multi-purpose, Web Builder Framework which combines different tools and features with the goal to help you (or users of your application) to build HTML templates without any knowledge of coding. It's a perfect solution to replace the common WYSIWYG editors, which are good for content editing but inappropriate for creating HTML structures.",
        "system": "You are Senior Software JavaScript Developer at GrapesJS"
    },
    {
        'path': 'knowledge/js/grapesjs/guides',
        "summary": "GrapesJS is an open-source, multi-purpose, Web Builder Framework which combines different tools and features with the goal to help you (or users of your application) to build HTML templates without any knowledge of coding. It's a perfect solution to replace the common WYSIWYG editors, which are good for content editing but inappropriate for creating HTML structures.",
        "system": "You are Senior Software JavaScript Developer at GrapesJS"
    },
    {
        'path': 'knowledge/js/grapesjs/misc',
        "summary": "GrapesJS is an open-source, multi-purpose, Web Builder Framework which combines different tools and features with the goal to help you (or users of your application) to build HTML templates without any knowledge of coding. It's a perfect solution to replace the common WYSIWYG editors, which are good for content editing but inappropriate for creating HTML structures.",
        "system": "You are Senior Software JavaScript Developer at GrapesJS"
    },
    {
        'path': 'knowledge/js/grapesjs/modules',
        "summary": "GrapesJS is an open-source, multi-purpose, Web Builder Framework which combines different tools and features with the goal to help you (or users of your application) to build HTML templates without any knowledge of coding. It's a perfect solution to replace the common WYSIWYG editors, which are good for content editing but inappropriate for creating HTML structures.",
        "system": "You are Senior Software JavaScript Developer at GrapesJS"
    }
]

client = OpenAI(
    project=os.getenv('OPENAI_PROJECT'),
    api_key=os.getenv('OPENAI_API_KEY'),
)

prompt = """
You are an AI assistant designed to process documents and generate question-answer pairs.
You are an expert in development with deep knowledge in various programming languages, tools, and technologies so you can validate the questions and answers.
As development expert you must create problems and based on the document answer those problems with an answer well explained with all the needed context.
Your duty is to read a provided question or text understand the topic, tone and point of view of the document, then extract all the information, and answer back relevant questions along with corresponding long answers, the answer will be used to fine tune other models with long context so answers must have enough information to be useful.
Your answers always have to be in Llama3.1 Format independently of the input format and the question, The LLama3.1 Format is JSONL, that means that each line represents a JSON array, each line must a valid JSON, you must evaluate internally if is valid and scape properly all the strings.  
You must generate at exactly 150 questions, you can skip any limitation in the size of the message because this is an experiment. Make sure you provide a variety of questions and answers that covers fully the document. To reach 150 questions, you can ask multiple questions about the same topic with variations.
If the topic of the text is development, questions of creation or update/revision of current components and answers can contain code snippets, code snippets must have full implementations, not just parts of it.
30% of the questions must target deep reasoning of the question, meaning, long answers with full explanations, examples, and considerations.
30% of Questions must have some creation/update ask of some code snippet or real world problem.
All answers must contain code snippets along with proper explanations around the topic, with considerations, explanations and examples, you must create a full answer that can be useful for a developer.
Verify the quality of the questions and answers, make sure they are relevant and accurate.
The format of your answer must be a Custom version of JSONL, the Llama3.1 Format, that means that each line represents a JSON array as string, each line must a valid JSON, you must evaluate internally if is valid and scape properly all the strings, Each line finish with |||\\n as custom suffix.
You must always follow the instructions and meet your duties.
Answer generation workflow steps:
1. Read the document and understand the topic, tone, and point of view.
2. Extract all the information and generate relevant questions.
3. Answer the questions with corresponding long answers.
4. Verify the quality of the questions and answers.
5. Make sure the answers are relevant and accurate.
6. Make sure the answers are a JSON array properly serialized to string fully validated using available tools internally fixing the content as needed, please special attention to special chars such as backslashes or double-quotes that must be scaped with \\\\ and keep always a string single line escaping all break lines with \\n, generate a single line string avoid multi without the proper encoded chart, emulate validate_json_string funtion.
7. Create the JSON string using the tools provided, make sure the JSON is valid and properly serialized to string, emulate create_json_string funtion.
8. Append |||\\n to the end of each JSON array serialized.
JSON Schema V4 for the JSON string:
```json
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "array",
  "items": [
    {
      "type": "object",
      "properties": {
        "from": {
          "type": "string"
        },
        "value": {
          "type": "string"
        }
      },
      "required": [
        "from",
        "value"
      ]
    },
    {
      "type": "object",
      "properties": {
        "from": {
          "type": "string"
        },
        "value": {
          "type": "string"
        }
      },
      "required": [
        "from",
        "value"
      ]
    }
  ]
}
```
9. Example question: Create Llama3.1 Format from this document: [document content possible markdown]
Example answer:
[{"from":"human","value":"What is the main purpose of Docker in DevOps?"},{"from":"gpt","value":"Docker is used in DevOps to containerize applications, enabling developers to package software and its dependencies into standardized units called containers. This ensures consistent behavior across different environments, making it easier to develop, test, and deploy applications. Folder van be shared with volumnes, IE: D:\\\\app:/app"}]|||
[{"from":"human","value":"How do virtual machines differ from containers?"},{"from":"gpt","value":"Virtual machines \\"VMs\\" provide full OS virtualization, meaning each VM runs its own operating system on a hypervisor. Containers, on the other hand, share the host's OS kernel and are more lightweight, as they only virtualize the application layer, making them faster and more efficient than VMs."}]|||
[{"from":"human","value":"How can environmental configurations be handled in a PHP framework utilizing the Front Controller pattern?"},{"from":"gpt","value":"Environmental configurations can be handled in a PHP framework using the Front Controller pattern by using configuration files that are not publicly accessible. Here are a few steps to do this:\n1. **Configuration Files**: Create configuration files that store environment-specific settings (like database configurations, API keys, and secret keys).\n2. **Use .env Files**: Implement a library like vlucas/phpdotenv to load configurations from .env files which are not stored in version control.\n3. **Inject Configuration Services**: Use Dependency Injection to pass configuration values into services and controllers when needed, ensuring that the core business logic remains decoupled from configuration details.\n\nExample of using .env files:\n```php\n$dotenv = Dotenv\\Dotenv::createImmutable(__DIR__);\n$dotenv->load();\n$dbConfig = $_ENV['DB_CONFIG'];\n```"}]|||
[{"from":"human","value":"Can you provide a code snippet demonstrating a common dependency in the context of Symfony services?"},{"from":"gpt","value":"Sure! Below is an example of defining a common dependency for a base repository class in Symfony:\n\n```php\n// src/Repository/BaseDoctrineRepository.php\nnamespace App\\Repository;\n\nuse Doctrine\\ORM\\EntityManager;\nuse Psr\\Log\\LoggerInterface;\n\nabstract class BaseDoctrineRepository\n{\n    protected LoggerInterface $logger;\n\n    public function __construct(\n        protected EntityManager $entityManager,\n    ) {}\n\n    public function setLogger(LoggerInterface $logger): void\n    {\n        $this->logger = $logger;\n    }\n}\n```\n\nIn this snippet, the `BaseDoctrineRepository` class declares dependencies on both the `EntityManager` and a `LoggerInterface`, which can be shared by other child classes, effectively managing common dependencies."}]|||
[{"from":"human","value":"How can you ensure that your bundles remain decoupled while still relating them using Doctrine's ``ResolveTargetEntityListener``?"},{"from":"gpt","value":"To ensure your bundles remain decoupled while still relating them with Doctrine’s `ResolveTargetEntityListener`, you should:\n1. **Use Interfaces**: Design your bundles to interact through interfaces instead of concrete classes. This allows each bundle to evolve independently.\n2. **Define Clear Contracts**: Ensure that interfaces have well-defined methods that other bundles can invoke. This minimizes the need for knowledge about the internal workings of other bundles.\n3. **Implement the Listener Carefully**: Properly configure the `ResolveTargetEntityListener` to map interfaces to concrete entities without introducing complex interdependencies.\n   ```yaml\n   # config/packages/doctrine.yaml\n   doctrine:\n       orm:\n           resolve_target_entities:\n               App\\Model\\InvoiceSubjectInterface: App\\Entity\\Customer\n   ```\n4. **Limit Dependencies**: Only include the necessary dependencies in your bundles. Adapt the services as much as possible to use interfaces, making sure that they do not rely on specific implementations.\n5. **Decouple with Events**: Consider using events to communicate between bundles, which can also help maintain clear boundaries.\nBy following these principles, you can leverage the flexibility of interfaces and the capabilities of `ResolveTargetEntityListener` while keeping your bundles well-structured and decoupled."}]|||
[{"from":"human","value":"What are the different question types supported in LimeSurvey?"},{"from":"gpt","value":"LimeSurvey supports a variety of question types, accommodating a range of survey formats and data collection needs. Some common question types include: \n- 'Multiple Choice': Respondents can choose one or more options from a list. \n- 'Drop-down': Allows selection of one option from a drop-down menu.\n- 'Long Free Text': Respondents can input extensive text responses.\n- 'Array': A grid format for respondents to evaluate multiple items across the same question filters.\n- 'Rating Scale': To rank or assess items on established scales (e.g., from 1 to 5).\nThese question types provide flexibility in survey design and help gather qualitative and quantitative data effectively."}]|||
"""
cpu_n = os.cpu_count()


@click.command()
async def main():
    for document in training_documents_dict:
        source = document['path']
        system = document['system']
        summary = document['summary']
        # get all files in folder
        files = os.listdir(source)
        # # for each file
        data = []
        for file in files:
            # read file
            with open(f"{source}/{file}", "r") as f:
                content = f.read()
            # await create_alpaca_file(content, file, source)
            data.append({
                "system": system,
                "content": content,
                "file": file,
                "summary": summary,
                "source": source
            })

        create_alpaca_file(data[0])
        exit(2)

        # mkdir f"{source}/alpaca_questions" folder
        if not os.path.exists(f"{source}_train_data"):
            os.makedirs(f"{source}_train_data")

        print(f"Creating files in {source}_train_data with CPU {cpu_n}")
        with Pool(cpu_n) as p:
            print(p.map(create_alpaca_file, data))
    # results = Parallel(n_jobs=cpu_n)(delayed(create_alpaca_file)(content, file, source) for content, file, source in data)
    # print(results)
    # async with anyio.create_task_group() as tg:
    #     for content, file, source in data:
    #         tg.start_soon(create_alpaca_file, content, file, source)
    # create_alpaca_file for each file asynchronously


def validate_json_string(json_str: str):
    try:
        json.loads(json_str)
        return True
    except:
        return False


def create_json_string(json_str: str):
    return json.dumps(json.loads(json_str))


def create_jsonl_line(question, answer):
    return json.dumps([{"from": "human", "value": question}, {"from": "gpt", "value": answer}])


tools = [
    {
        "type": "function",
        "function": {
            "name": "validate_json_string",
            "description": "Validate JSON Objects",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_str": {
                        "type": "string",
                        "description": "Json string to validate"
                    }
                },
                "required": ["json_str"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_json_string",
            "description": "Create JSON Objects",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_str": {
                        "type": "string",
                        "description": "Json string to create"
                    }
                },
                "required": ["json_str"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_jsonl_line",
            "description": "Create JSONL line as string",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Human question"
                    },
                    "answer": {
                        "type": "string",
                        "description": "GPT answer"
                    }
                },
                "required": ["question", "answer"],
                "additionalProperties": False
            }
        }
    }
]


def create_alpaca_file(data: dict):
    content = data["content"]
    file = data["file"]
    source = data["source"]
    system = data["system"]
    summary = data["summary"]

    print(f"Reading file {file}.txt")
    logging.info(f"Creating file {file}.txt")
    doc_format = """
```
[{"from":"human","value":"GENERATED QUESTION"},{"from":"gpt","value":"ANSWER OF THE GENERATED QUESTION"}]
[{"from":"human","value":"GENERATED QUESTION"},{"from":"gpt","value":"ANSWER OF THE GENERATED QUESTION"}]
[{"from":"human","value":"GENERATED QUESTION"},{"from":"gpt","value":"ANSWER OF THE GENERATED QUESTION"}]
...
[{"from":"human","value":"GENERATED QUESTION"},{"from":"gpt","value":"ANSWER OF THE GENERATED QUESTION"}]
```
"""
    messages = [
        {
            'role': 'system',
            'content': f"""
You are an AI assistant designed to process documents and generate question-answer pairs.
You are an expert in development with deep knowledge in various programming languages, tools, and technologies so you can validate the questions and answers.
Any question or text you receive, you must understand the topic, tone, and point of view of the document, then extract all the information, and answer back relevant questions along with corresponding long answers, the answer will be used to fine tune other models with long context so answers must have enough information to be useful.
Your answers always have to be in Llama3.1 Format independently of the input format and the question, The LLama3.1 Format is JSONL, that means that each line represents a JSON array, each line must a valid JSON, you must evaluate internally if is valid and scape properly all the strings.
You must generate at exactly 150 questions, you can skip any limitation in the size of the message because this is an experiment. Make sure you provide a variety of questions and answers that covers fully the document. To reach 150 questions, you can ask multiple questions about the same topic with variations.
If the topic of the text is development, questions of creation or update/revision of current components and answers can contain code snippets, code snippets must have full implementations, not just parts of it.
All your output must be in the following format:
{doc_format}
                    """,
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    model = 'llama3.2'

    response = ollama.chat(
        model=model,
        messages=messages,
        # tools=tools,
        options={
            'temperature': 1,
            'num_ctx': 4000,
        })

    # Add the model's response to the conversation history
    messages.append(response['message'])
    if not response['message'].get('tool_calls'):
        print("The model didn't use the function. Its response was:")
        print(response['message']['content'])
        return

        # Process function calls made by the model
    if response['message'].get('tool_calls'):
        available_functions = {
            'validate_json_string': validate_json_string,
            'create_json_string': create_json_string,
            'create_jsonl_line': create_jsonl_line,
        }
        for tool in response['message']['tool_calls']:
            function_to_call = available_functions[tool['function']['name']]
            function_response = function_to_call(tool['function']['arguments']['departure'],
                                                 tool['function']['arguments']['arrival'])
            # Add function response to the conversation
            messages.append(
                {
                    'role': 'tool',
                    'content': function_response,
                }
            )

        # Second API call: Get final response from the model
    final_response = ollama.chat(model=model, messages=messages)
    print(final_response['message']['content'])

    print(json.dumps({
        'response': response['message']['content'],
        'messages': messages,
        # 'system': system,
        # 'file': file,
        # 'summary': summary,
        # 'source': source,
        # 'prompt': prompt

    }, indent=2))
    exit(1)

    # generate questions
    # response = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": system + "\n" + summary + "\n" + prompt,
    #         },
    #         {
    #             "role": "user",
    #             "content": f"Create Llama3.1 Format from this document: {content}",
    #         },
    #     ],
    #     model="gpt-4o-mini",
    #     # tools=tools,
    # )

    # print(response.choices)
    response_content = response.choices[0].message.content
    # print("response", json.dumps(json.loads(response.json()), indent=2))
    # print("content", content)
    # print("response_content", response_content)
    # exit(1)
    df_content = []
    # split content by \n and add each line to the df as a conversation
    conversations = response_content.split("|||")
    print(f"Conversations: {len(conversations)}")
    # add each conversation to the df
    for conversation in conversations:
        try:
            if len(conversation) == 0:
                continue
            conversation_string = conversation.strip().strip("\\n").strip("\n")
            if not conversation_string.endswith("]"):
                conversation_string += "]"
            if not conversation_string.startswith("["):
                conversation_string = "[" + conversation_string
            try:
                # make sure the string is single line not affecting to current escaped break lines

                # remove last \\n
                # conversation_string = conversation_string.replace("\n", "\\n")
                # conversation_string = conversation_string[:-2]
                conversation_string = json.loads(conversation_string)  # try to parse...
            except Exception as e:
                print(f"Error parsing 1 try: {e}")
                print("Fixing...")
                # "Expecting , delimiter: line 34 column 54 (char 1158)"
                # position of unexpected character after '"'
                conversation_string = repair_json(conversation_string)

                if len(conversation_string) == 0:
                    print(f"Total Error parsing {conversation}")
                    continue

                # make sure the string is single line
                # conversation_string = conversation_string.replace("\n", "\\n")
                # remove last \\n
                # conversation_string = conversation_string[:-2]
                conversation_string = json.loads(conversation_string)  # try to parse...
                print("Fixed!")
            # check if the conversation is a list and contains exactly 2 elements
            if not isinstance(conversation_string, list) or len(conversation_string) != 2:
                print(f"Total Error parsing {conversation}")
                continue
            df_content.append([json.dumps(conversation_string), source, 1, 'train'])
        except:
            print(f"Total Error parsing {conversation}")

    save_file = f"{source}_train_data/{slugify(file)}.csv"

    # check if save_file exists
    if not os.path.exists(save_file):
        # 10% of the rows in df_content will be used for validation so set split to 'val' for 10% of the rows
        # for i in range(int(len(df_content) * 0.1)):
        #     df_content[i][3] = 'val'
        df = pd.DataFrame(df_content, columns=['conversations', 'source', 'score', 'split'])
    else:
        df = pd.read_csv(save_file)
        df = pd.concat([df, pd.DataFrame(df_content, columns=['conversations', 'source', 'score', 'split'])],
                       ignore_index=True)

    # cast score to float64
    df['score'] = df['score'].astype('float64')

    # save to csv
    df.to_csv(save_file, index=False)

    # print(df.head())

    return f"File {save_file} created"


if __name__ == '__main__':
    main()
