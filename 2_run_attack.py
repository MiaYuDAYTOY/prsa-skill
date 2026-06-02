import numpy as np

import os
from tqdm import tqdm
import json
import argparse
import pandas as pd
import dataset
import utils
import llm
from debug_structural_js import StructuralJSDebugger
from sentence_bert import calculate_similarity_sbert



def is_valid_score(score, eps=1e-12):
    if score is None:
        return False
    if not np.isfinite(score):
        return False
    if score <= eps:
        return False
    return True


def save_to_csv(df, directory, filename):
    """Save DataFrame to a CSV file, ensuring the directory exists."""
    os.makedirs(directory, exist_ok=True)
    csv_file = os.path.join(directory, filename)
    df.to_csv(csv_file, index=False)
    print(f"File saved: {csv_file}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', default='demo_data')
    parser.add_argument('--data_dir', default='demo_data')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--beam_search', default=1, type=int)
    parser.add_argument('--pre_pruning', default=1, type=int)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--related_words_interval', default=5, type=int)
    parser.add_argument('--out', default='log/test_out.txt')
    parser.add_argument('--temperature', default=0.7, type=float)

    parser.add_argument('--evaluator', default="syntactic_similarity", type=str)
    parser.add_argument('--m', default=3, type=int, help='m is the sampling number for each user input')
    parser.add_argument('--n', default=2, type=int, help='n is the number of the test user input')
    parser.add_argument('--scorer', default="syntactic_similarity", type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()
    config = vars(args)

    data = dataset.Datasets(config)
    scorer = StructuralJSDebugger(m=args.m, score_mode="inverse")
    model = llm.ChatGPTPredictor(config)
    test_data = data.load_test_data()

    if os.path.exists(args.out):
        os.remove(args.out)
    print(config)

    scores_list = []
    save_file_name = config["theme"] + "_res"
    llm_based_eva_score_list = []
    for idx, batch in tqdm(enumerate(test_data)):
        input_data = batch['Preview Input']
        print(idx)
        
        category = batch.get('Category', args.theme)
        print("\nGet the prompt category:", category)
        config["theme"] = category

        gradient_path = "model/gradient_" + config["theme"] + ".json"
        with open(gradient_path, 'r') as file:
            gradient_dict = json.load(file)
        
        print("\n========Start Stealing Target Prompt========  ... ...")

        target_prompt = batch['Prompt']
        output_data = batch['Preview Output']
        input_case1 = batch['Input Test Case1']
        input_case2 = batch['Input Test Case2']
        inputs = [input_case1, input_case2]
        inputs = inputs[:args.n]
        
        stolen_prompt = llm.generate_skill(config, input_data, output_data, gradient_dict)

        if stolen_prompt is None:
            print(f"[Warning] Skipped iteration {idx} because generate_skill returned None.")
            continue

        stolen_prompt = utils.prompt_pruning_google(config, stolen_prompt, input_data, output_data, model.inference)
        stolen_prompt = utils.format_clean(stolen_prompt)

        print("\nStolen prompt is :", stolen_prompt)

        '''
        (Optional 1)
        stolen_prompt = utils.prompt_pruning(config, stolen_prompt, input_data, output_data, model.inference)
        stolen_prompt = utils.format_clean(stolen_prompt_1)

        (Optional 2)
        stolen_prompt = utils.prompt_pruning_phrase_level(config, stolen_prompt, input_data, output_data, model.inference)
        stolen_prompt = utils.format_clean(stolen_prompt)
        '''

        with open(args.out, 'a') as outf:
            outf.write(json.dumps(f"input_data: {input_data}") + '\n')
            outf.write(json.dumps(f"stolen_prompt: {stolen_prompt}") + '\n')
            outf.write(json.dumps(f"target_prompt: {target_prompt}") + '\n')

        print("\n========Evaluation of Prompt Similarity======== ... ...")
        prompt_sim_score = calculate_similarity_sbert(target_prompt, stolen_prompt)
        print("\nPrompt similarity score is :", prompt_sim_score)


        print("\n========Evaluation of Functional Consistency: JS Structural Only======== ... ...")

        js_target_score = scorer.evaluate_target_prompt_js_only(
            model.inference,
            inputs,
            target_prompt
        )

        if js_target_score is None:
            print(f"[Warning] Skipped iteration {idx} due to failed target JS evaluation.")
            continue

        js_stolen_score = scorer.evaluate_stolen_prompt_js_only(
            model.inference,
            inputs,
            target_prompt,
            stolen_prompt
        )

        if js_stolen_score is None:
            print(f"[Warning] Skipped iteration {idx} due to failed stolen JS evaluation.")
            continue

        if is_valid_score(js_target_score) and is_valid_score(js_stolen_score):
            js_sim_score = min(js_stolen_score / js_target_score, 1.0)

            print("\nStructural target score is:", js_target_score)
            print("Structural stolen score is:", js_stolen_score)
            print("Structural similarity score is:", js_sim_score)

            with open(args.out, 'a') as outf:
                outf.write(json.dumps(
                    f"Round [{idx+1}/{len(test_data)}] structural target score: {js_target_score:.4f}"
                ) + '\n')

                outf.write(json.dumps(
                    f"Round [{idx+1}/{len(test_data)}] structural stolen score: {js_stolen_score:.4f}"
                ) + '\n')

                outf.write(json.dumps(
                    f"Round [{idx+1}/{len(test_data)}] structural similarity score: {js_sim_score:.4f}"
                ) + '\n')

                outf.write(json.dumps(
                    f"Round [{idx+1}/{len(test_data)}] prompt similarity score: {prompt_sim_score:.4f}"
                ) + '\n')

                outf.write('\n\n')

            scores_list.append({
                'iteration': idx,
                'target prompt': target_prompt,
                'stolen prompt': stolen_prompt,
                'structural target score': js_target_score,
                'structural stolen score': js_stolen_score,
                'structural similarity score': js_sim_score,
                'prompt similarity score': prompt_sim_score
            })

        else:
            print(f"[Warning] Skipped iteration {idx} due to invalid JS score.")
            print(f"[DEBUG] js_target_score = {js_target_score}")
            print(f"[DEBUG] js_stolen_score = {js_stolen_score}")
            continue

        '''
        print("\n======== LLM-based Multi-dimensional Evaluation======== ... ...")
        for input_case in inputs:
            target_output = model.inference(input_case, target_prompt)
            generated_output =  model.inference(input_case, stolen_prompt)

            if target_output is None or generated_output is None:
                print(f"[Warning] Skipped input '{input_case}' due to inference failure.")
                continue

            llm_eva_res = llm.llm_based_evaluation(target_output, generated_output)
            try:
                llm_eva_scores = json.loads(llm_eva_res)
                print("LLM-based evaluation score: ", llm_eva_scores)
            except json.JSONDecodeError:
                llm_eva_scores = {"Accuracy":0, "Completeness":0, "Tone":0, "Sentiment":0, "Semantics":0}
                print("Error: Could not parse response as JSON.")
                print("Raw response:", llm_eva_res)

            llm_based_eva_score_list.append((input_case, target_prompt, stolen_prompt,
                               llm_eva_scores["Accuracy"], llm_eva_scores["Completeness"],llm_eva_scores["Tone"],llm_eva_scores["Sentiment"],llm_eva_scores["Semantics"]))
        '''

    
    directory = "result"
    df = pd.DataFrame(llm_based_eva_score_list, columns=['user input', 'target prompt', 'stolen prompt',
                                        'accuracy', 'completeness', 'tone', 'sentiment', 'semantics'])
    save_to_csv(df, directory, f"llm_based_eva_{save_file_name}.csv")

    df_scores = pd.DataFrame(scores_list)
    save_to_csv(df_scores, directory, f"{save_file_name}.csv")
