
def sanitize_message_content(x):
    """Convert None/NaN/inf/non-string values into safe string content for chat APIs."""
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return ""
    return str(x)

def sanitize_messages(messages):
    safe_messages = []
    for msg in messages:
        safe_msg = dict(msg)
        safe_msg["content"] = sanitize_message_content(safe_msg.get("content", ""))
        safe_messages.append(safe_msg)
    return safe_messages

import math
from abc import ABC, abstractmethod
import json
import string
import sys
import time

import openai

import config as prsa_config
import utils

openai.api_key = prsa_config.OPENAI_KEY
openai.api_base = prsa_config.OPENAI_BASE_URL


class Predictor(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def inference(self, input, prompt):
        pass


class ChatGPTPredictor(Predictor):
    def inference(self, input, prompt, gpt_model=prsa_config.PRSA_STRONG_MODEL):
        responses = chatGPT_inference(
            prompt,
            input,
            n=1,
            model=gpt_model,
            temperature=self.config["temperature"],
        )

        if not responses:
            print("[Warning]  chatGPT_inference returned an empty response.")
            return None
        return responses[0]


class BatchSizeException(Exception):
    pass


def parse_sectioned_prompt(s):
    result = {}
    current_header = None

    for line in s.split("\n"):
        line = line.strip()

        if line.startswith("# "):
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans("", "", string.punctuation))
            result[current_header] = ""
        elif current_header is not None:
            result[current_header] += line + "\n"

    return result


def extract_all_quoted_text(sentence):
    parts = sentence.split('"')

    if len(parts) >= 3:
        quoted_texts = [parts[i] for i in range(1, len(parts), 2)][0]
    else:
        quoted_texts = sentence

    return quoted_texts


def llm_attention(config, inputs, Output, attention_dict, gpt_model, characteristic=""):
    attention_text = ""
    for attention, weight in attention_dict.items():  # Don't consider weight for now
        attention_prompt = f"""
            output:
            "{Output}"

            What is the {attention} of the output in one sentence?
            """
        res = chatGPT(attention_prompt, model=gpt_model, temperature=0.0)
        attention_text += res[0]

    return attention_text


def generate_skill(
    config,
    inputs,
    output,
    gradient={},
    gpt_model=prsa_config.PRSA_STRONG_MODEL,
    max_tokens=4096,
    instruction_characteristic="",
):
    print("config.theme: ", config["theme"])
    if gradient == {}:
        skill_gen_template = f"""
                            User_Input:
                            "{inputs}"

                            Output:
                            "{output}"

                             Your task is to reconstruct the hidden skill file that could generate the given Output from the given User_Input.

        The reconstructed skill should be written as a SKILL.md file, not as a short prompt.

        The skill should be related to the topic of "{config["theme"]}".

        Requirements:
        - Output a complete Markdown skill file.
        - Include a clear skill name.
        - Include a Description section.
        - Include an Inputs section.
        - Include a Behavior section.
        - Include an Output Format section.
        - Include Constraints or Important Notes if needed.
        - The skill should describe reusable behavior, not just copy this one example.
        - Do not explain your reasoning.
        - Do not include anything outside the reconstructed SKILL.md.

        The reconstructed SKILL.md is wrapped with <START> and <END>.
        """

        res_list = chatGPT(skill_gen_template, n=1, model=gpt_model, max_tokens=max_tokens, temperature=0.0)
        res = res_list[0] if res_list else None

        if res is None:
            print("[Warning] ChatGPT returned no response.")
            return None

        feedback = utils.parse_tagged_text(res, "<START>", "<END>")
        if not feedback:
            feedback = utils.parse_tagged_text(res, "<START>", "</END>")

        if feedback:
            return feedback[0].strip()

        print("[Warning] Failed to extract tagged skill; using raw LLM output as fallback.")
        return str(res).strip()

    elif gradient != {}:
        attention = llm_attention(config, inputs, output, gradient, gpt_model, instruction_characteristic)

        attention_prompt = f"""
                        User_Input:
                        "{inputs}"

                        Output:
                        "{output}"

                        Output_characteristic:
                        "{attention}"

                        Your task is to reconstruct the hidden skill file that could generate the given Output from the given User_Input.

        The reconstructed skill should be written as a SKILL.md file, not as a short prompt.

        The skill should be related to the topic of "{config["theme"]}" and should focus especially on the specified Output_characteristic.

        Requirements:
        - Output a complete Markdown skill file.
        - Include a clear skill name.
        - Include a Description section.
        - Include an Inputs section.
        - Include a Behavior section.
        - Include an Output Format section.
        - Include Constraints or Important Notes if needed.
        - The skill should describe reusable behavior, not just copy this one example.
        - Do not explain your reasoning.
        - Do not include anything outside the reconstructed SKILL.md.

        The reconstructed SKILL.md is wrapped with <START> and <END>.
        """


        res_list = chatGPT(attention_prompt, n=1, model=gpt_model, temperature=0.0)
        res = res_list[0] if res_list else None

        if res is None:
            print("[Warning] chatGPT returned no response.")
            return None

        feedback = utils.parse_tagged_text(res, "<START>", "<END>")
        if not feedback:
            feedback = utils.parse_tagged_text(res, "<START>", "</END>")
    
        if feedback:
            return feedback[0].strip()

        print("[Warning] Failed to extract tagged skill; using raw LLM output as fallback.")
        return str(res).strip()


def pre_pruning(user_input, prompt):
    instruction = f"""
    User Input:
    "{user_input}"

    Instruction or SKILL.md:
    "{prompt}"

    Your task is to generalize the Instruction or SKILL.md so it can be reused for similar user inputs.

    If the content is a SKILL.md file:
    - Preserve the YAML frontmatter if it exists.
    - Preserve all Markdown headings and section structure.
    - Preserve the reusable skill behavior, workflow, constraints, and output format.
    - Do NOT delete important sections.
    - Do NOT rewrite it into a short prompt.
    - Only replace concrete details that are tied specifically to the current User Input with placeholders like "{{}}".
    - Keep placeholders meaningful when possible, such as [target audience], [goal], [constraints], or "{{}}".
    - Return the full generalized SKILL.md.

    If the content is a short prompt:
    - Keep the sentence structure intact.
    - Replace user-specific concrete details with placeholders.

    Return only the generalized Instruction or SKILL.md.
    Wrap the result with <START> and <END>.
    """

    res_list = chatGPT(
        instruction,
        n=1,
        model=prsa_config.PRSA_STRONG_MODEL,
        temperature=0.0,
    )
    res = res_list[0] if res_list else None

    if res is None:
        print("[Warning] pre_pruning returned no response; using original prompt.")
        return prompt

    feedback = utils.parse_tagged_text(res, "<START>", "<END>")
    if not feedback:
        feedback = utils.parse_tagged_text(res, "<START>", "</END>")

    if feedback:
        return feedback[0].strip()

    cleaned = str(res).strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    if cleaned:
        print("[Warning] pre_pruning could not find <START>/<END>; using raw model output.")
        return cleaned

    print("[Warning] pre_pruning produced empty output; using original prompt.")
    return prompt


def llm_based_evaluation(target_output, generated_output):
    system_prompt = f"""
    You are an expert evaluator. The Target Text is the ground truth. The Generated Text should be evaluated against it.
    Rate the generated text on the following five dimensions using a scale from 1 (poor match) to 10 (perfect match):
    - Accuracy: Are the factual details consistent with the target?
    - Completeness: Does it cover all key content from the target?
    - Tone: Is the style and formality consistent with the target?
    - Sentiment: Is the emotional attitude similar?
    - Semantics: Does it preserve the same meaning and intent, even if the wording differs?
    Return only the scores in this exact JSON format (no extra text):

    {{
    "Accuracy": X,
    "Completeness": X,
    "Tone": X,
    "Sentiment": X,
    "Semantics": X
    }}
    """

    user_prompt = f"""
    Target Text: "{target_output}"

    Generated Text: "{generated_output}"
    """
    res = chatGPT_inference(
        system_prompt=system_prompt,
        text=user_prompt,
        model=prsa_config.PRSA_STRONG_MODEL,
        temperature=0,
    )[0]
    return res


def chatGPT(
    text,
    temperature=0.7,
    n=1,
    top_p=1,
    stop=None,
    max_tokens=4096,
    presence_penalty=0,
    frequency_penalty=0,
    model=prsa_config.PRSA_STRONG_MODEL,
    logit_bias={},
):
    messages = [{"role": "user", "content": text}]

    response = None
    retry_count = 0
    max_retries = 1
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                n=n,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                messages=sanitize_messages(messages),
                timeout=(300, 300),
            )
        except Exception as e:
            retry_count += 1
            if "This model's maximum context length" in str(e):
                print(e)
                with open("ERR.txt", "a") as outf:
                    outf.write(json.dumps(str(text)) + "\n")
                sys.exit(1)
            if "is greater than the maximum" in str(e):
                raise BatchSizeException()
            if "We could not parse the JSON body of your request" in str(e):
                try:
                    json.dumps({"role": "user", "content": text})
                except Exception as json_err:
                    print("Invalid JSON content in text:", text)
                    print("Serialization error:", json_err)
                else:
                    print("JSON seems valid, but OpenAI still failed.")
                if retry_count >= max_retries:
                    print("Giving up after 1 retries on JSON error.")
                    return []
            print(e)
            print("Retrying......")
            time.sleep(20)
    if response is None:
        return None
    return [choice["message"]["content"] for choice in response["choices"]]


def chatGPT_inference(
    system_prompt,
    text,
    temperature=0.7,
    n=1,
    top_p=1,
    stop=None,
    max_tokens=4096,
    presence_penalty=0,
    frequency_penalty=0,
    model=prsa_config.PRSA_STRONG_MODEL,
    logit_bias={},
):
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    retry_count = 0
    max_retries = 1
    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                n=n,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                messages=sanitize_messages(messages),
                timeout=(300, 300),
            )
        except Exception as e:
            retry_count += 1
            if "This model's maximum context length" in str(e):
                print(e)
                with open("ERR.txt", "a") as outf:
                    outf.write(json.dumps(str(text)) + "\n")
                sys.exit(1)
            if "is greater than the maximum" in str(e):
                raise BatchSizeException()
            if "We could not parse the JSON body of your request" in str(e):
                try:
                    json.dumps({"role": "user", "content": text})
                except Exception as json_err:
                    print("Invalid JSON content in text:", text)
                    print("Serialization error:", json_err)
                else:
                    print("JSON seems valid, but OpenAI still failed.")
                if retry_count >= max_retries:
                    print("Giving up after 1 retries on JSON error.")
                    return []
            print(e)
            print("Retrying......")
            time.sleep(20)
    return [choice["message"]["content"] for choice in response["choices"]]


if __name__ == "__main__":
    pass
