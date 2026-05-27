from abc import ABC
import llm
import re
import config

class PromptOptimizer(ABC):
    def __init__(self, args, scorer, gradient_dict):
        self.opt = args
        self.scorer = scorer
        self.gradient_dict = gradient_dict

class PAA(PromptOptimizer):
    """ PAA: Prompt Attention Algorithm
        This idea of Prompt Attention Algorithm is borrowed from this paper: "Automatic Prompt Optimization with "Gradient Descent" and Beam Search"
    """
    

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def filter_target_score(self, text, feedbacks):
        """
        Robustly parse a similarity score from model output.

        Priority:
        1. Parse numbers inside <START>...<END> or <START>...</END>.
        2. If multiple valid numbers appear, choose the minimum one.
           This handles outputs like "7 out of 10".
        3. If parsing fails, return slightly below the attention threshold,
           so the dimension is conservatively marked as needing attention.
        """
        candidates = []
        if feedbacks:
            candidates.extend(feedbacks)

        candidates.extend(self.parse_tagged_text(text, "<START>", "<END>"))
        candidates.extend(self.parse_tagged_text(text, "<START>", "</END>"))

        scores = []
        for item in candidates:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(item))
            for number in numbers:
                value = float(number)
                if 0 <= value <= 10:
                    scores.append(value)

        if scores:
            return min(scores)

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        fallback_scores = []
        for number in numbers:
            value = float(number)
            if 0 <= value <= 10:
                fallback_scores.append(value)

        if fallback_scores:
            return min(fallback_scores)

        print("\n[WARN] Could not parse target score from model output:")
        print(repr(text))
        return float(self.opt.get("attention_threshold", 7.5)) - 0.1


    def extract_number(self, s):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        return numbers[0] if numbers else None

    def cal_gradients(self, generated_output, output_data):

        gradient = {}
        if self.opt["theme"] in ["Music", "Sports"]:
            self.opt["attention_threshold"] = 8

        elements = ['Characteristic','Topic','Argument','Structure','Style','Tone','Purpose','Sentence Type','Audience','Background']
        element_desc_map = {
            "Sentence Type": (
                "sentence-level form and sentence pattern, including declarative/interrogative/imperative style, "
                "sentence length, sentence complexity, punctuation pattern, and whether the two outputs use similar sentence structures"
            ),
            "Structure": "overall organization, paragraph arrangement, ordering of information, and formatting structure",
            "Style": "writing style, level of formality, wording pattern, and rhetorical style",
            "Tone": "emotional tone, politeness, enthusiasm, directness, and attitude toward the reader",
            "Characteristic": "distinctive writing characteristics, recurring traits, and recognizable behavioral patterns",
            "Topic": "main topic, subject matter, and topical focus",
            "Argument": "main claims, reasoning, persuasive logic, and supporting points",
            "Purpose": "communicative goal, intended function, and what the output is trying to accomplish",
            "Audience": "intended reader, target user group, and assumptions about the recipient",
            "Background": "contextual assumptions, scenario setup, domain background, and relevant prior information",
        }

        for idx, element in enumerate(elements):
            element_desc = element_desc_map.get(element, element)

            gradient_prompt = f"""
            Generated Output:
            "{generated_output}"

            Real Output:
            "{output_data}"

            You are a harsh and conservative evaluator.
Evaluate ONLY this dimension: {element_desc}
Full score is 10. Do not be generous. Penalize any noticeable difference.

Scoring rubric:
10 = identical or nearly identical in this dimension.
9 = very similar, only tiny differences.
8 = similar, but there are clear noticeable differences.
7 = only partially similar.
6 = weakly similar.
5 or below = substantially different.

Important: If you are unsure between two scores, choose the lower score.
Output exactly one number between 0 and 10, with up to two decimal places.
Example output: 7.31
Do not explain. Do not output tags. Do not output anything except the number.
            """
            gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
            res = llm.chatGPT(gradient_prompt, model=config.PRSA_FAST_MODEL, temperature=0.0)
            feedbacks = []
            temp = []
            for r in res:    
                temp += self.parse_tagged_text(r, "<START>", "<END>")
                feedbacks = self.filter_target_score(r, temp)
                print(f"[SCORE] element={element} score={feedbacks} threshold={self.opt['attention_threshold']}")
            try:
                if float(feedbacks) < self.opt["attention_threshold"]:
                    print(f"[LOW] element={element} score={feedbacks}")
                    gradient[element] = 1
            except:
                if float(self.extract_number(feedbacks)) < self.opt["attention_threshold"]:
                    print("extract feedback score: ",float(self.extract_number(feedbacks)))
                    gradient[element] = 1

        return gradient


    

        


    

    
