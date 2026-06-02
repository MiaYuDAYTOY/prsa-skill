import argparse
import csv
import math
import time
from collections import Counter, defaultdict

from scipy.spatial.distance import jensenshannon


class StructuralJSDebugger:
    def __init__(self, m=3, score_mode="inverse", base=2):
        if score_mode not in {"inverse", "raw"}:
            raise ValueError("score_mode must be either 'inverse' or 'raw'")
        self.m = m
        self.score_mode = score_mode
        self.base = base
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)

    def _now(self):
        return time.perf_counter()

    def _record(self, name, elapsed):
        self.timings[name] += elapsed
        self.counts[name] += 1

    def _format_seconds(self, seconds):
        return f"{seconds:.6f}s"

    def _is_bad_output(self, output):
        return output is None or str(output).strip() == ""

    def _mean_valid(self, values):
        valid = [value for value in values if value is not None and not math.isnan(value)]
        if not valid:
            return None
        return sum(valid) / len(valid)

    def _generate_outputs(self, fn, input_data, prompt, label):
        outputs = []
        for idx in range(self.m):
            start = self._now()
            output = fn(input_data, prompt)
            elapsed = self._now() - start
            self._record(f"generation:{label}", elapsed)

            print(f"[GEN {label.upper()} {idx}] time={self._format_seconds(elapsed)}")
            print(f"[GEN {label.upper()} {idx} RAW OUTPUT]")
            print(output)

            if self._is_bad_output(output):
                print(f"[WARNING] fn returned empty output for {label} index={idx}.")
                outputs.append(None)
            else:
                outputs.append(str(output))
        return outputs

    def compute_pair_debug(self, text1, text2):
        pair_start = self._now()
        text1 = "" if text1 is None else str(text1)
        text2 = "" if text2 is None else str(text2)

        print(f"[JS] text1_len={len(text1)}, text2_len={len(text2)}")

        tokenize_start = self._now()
        tokens1 = text1.split()
        tokens2 = text2.split()
        tokenize_time = self._now() - tokenize_start
        self._record("js:split_tokenize", tokenize_time)
        print(f"[JS] tokenize_time={self._format_seconds(tokenize_time)}")
        print(f"[JS] tokens1={len(tokens1)}, tokens2={len(tokens2)}")

        if not tokens1 or not tokens2:
            print("[WARNING] Empty token list detected; JS distance and structural score are undefined.")
            total_time = self._now() - pair_start
            self._record("js:pair_total", total_time)
            return {
                "distance": None,
                "structural_score": None,
                "tokens1": len(tokens1),
                "tokens2": len(tokens2),
                "vocab_size": 0,
                "total_time": total_time,
            }

        vocab_start = self._now()
        vocab = set(tokens1)
        vocab.update(tokens2)
        vocab_list = list(vocab)
        vocab_time = self._now() - vocab_start
        self._record("js:vocab_union", vocab_time)
        print(f"[JS] vocab_union_time={self._format_seconds(vocab_time)}")
        print(f"[JS] vocab_size={len(vocab_list)}")

        counter_start = self._now()
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)
        counter_time = self._now() - counter_start
        self._record("js:counter", counter_time)
        print(f"[JS] counter_time={self._format_seconds(counter_time)}")

        prob_start = self._now()
        total1 = len(tokens1)
        total2 = len(tokens2)
        prob_dist1 = [counter1[word] / total1 for word in vocab_list]
        prob_dist2 = [counter2[word] / total2 for word in vocab_list]
        prob_time = self._now() - prob_start
        self._record("js:prob_vector", prob_time)
        print(f"[JS] prob_vector_time={self._format_seconds(prob_time)}")

        if any(not math.isfinite(value) for value in prob_dist1 + prob_dist2):
            print("[WARNING] Probability vector contains NaN or inf.")

        js_start = self._now()
        distance = jensenshannon(prob_dist1, prob_dist2, base=self.base)
        js_time = self._now() - js_start
        self._record("js:scipy_jensenshannon", js_time)
        print(f"[JS] scipy_time={self._format_seconds(js_time)}")
        print(f"[JS] raw_distance={distance}")

        score_start = self._now()
        structural_score = self._distance_to_score(distance)
        score_time = self._now() - score_start
        self._record("js:distance_to_structural_score", score_time)
        print(f"[JS] structural_score_time={self._format_seconds(score_time)}")
        print(f"[JS] structural_score={structural_score}")

        total_time = self._now() - pair_start
        self._record("js:pair_total", total_time)
        return {
            "distance": distance,
            "structural_score": structural_score,
            "tokens1": len(tokens1),
            "tokens2": len(tokens2),
            "vocab_size": len(vocab_list),
            "total_time": total_time,
        }

    def _distance_to_score(self, distance):
        if distance is None:
            return None
        if not math.isfinite(distance):
            print("[WARNING] JS distance is NaN or inf; structural score is undefined.")
            return None
        if self.score_mode == "raw":
            return distance
        if distance == 0:
            print("[WARNING] JS distance is 0; inverse structural score would divide by zero.")
            return None
        return 1 / distance

    def eval_pred_target_js_only(self, pred_outputs, target_outputs, input_idx=None):
        scores = []
        for pred_idx, pred_output in enumerate(pred_outputs):
            for target_idx, target_output in enumerate(target_outputs):
                prefix = f"[PAIR input={input_idx} pred={pred_idx} target={target_idx}]"
                print(f"{prefix} start")
                pair_start = self._now()
                if self._is_bad_output(pred_output) or self._is_bad_output(target_output):
                    print(f"{prefix} skipped due to empty pred or target output.")
                    scores.append(None)
                    continue
                result = self.compute_pair_debug(pred_output, target_output)
                scores.append(result["structural_score"])
                elapsed = self._now() - pair_start
                print(f"{prefix} total_time={self._format_seconds(elapsed)}")
                print(f"{prefix} end")
        return self._mean_valid(scores)

    def eval_target_target_js_only(self, target_outputs, input_idx=None):
        if self.m <= 1:
            raise ValueError(f"`m` must be at least 2 to compute pairwise target similarity. Got m={self.m}")

        scores = []
        for i in range(len(target_outputs)):
            for j in range(i + 1, len(target_outputs)):
                prefix = f"[PAIR input={input_idx} target_i={i} target_j={j}]"
                print(f"{prefix} start")
                pair_start = self._now()
                if self._is_bad_output(target_outputs[i]) or self._is_bad_output(target_outputs[j]):
                    print(f"{prefix} skipped due to empty target output.")
                    scores.append(None)
                    continue
                result = self.compute_pair_debug(target_outputs[i], target_outputs[j])
                scores.append(result["structural_score"])
                elapsed = self._now() - pair_start
                print(f"{prefix} total_time={self._format_seconds(elapsed)}")
                print(f"{prefix} end")
        return self._mean_valid(scores)

    def evaluate_stolen_prompt_js_only(self, fn, inputs, target_prompt, stolen_prompt):
        all_start = self._now()
        input_scores = []

        for input_idx, input_data in enumerate(inputs):
            input_start = self._now()
            print(f"[INPUT {input_idx}] start")
            print(f"[INPUT {input_idx} RAW INPUT]")
            print(input_data)

            target_outputs = self._generate_outputs(fn, input_data, target_prompt, "target")
            pred_outputs = self._generate_outputs(fn, input_data, stolen_prompt, "pred")

            if any(self._is_bad_output(output) for output in target_outputs + pred_outputs):
                print(f"[INPUT {input_idx}] warning: empty output found; invalid pairs will be skipped.")

            avg_score = self.eval_pred_target_js_only(pred_outputs, target_outputs, input_idx=input_idx)
            input_scores.append(avg_score)

            input_elapsed = self._now() - input_start
            self._record("input:total", input_elapsed)
            print(f"[INPUT {input_idx}] avg_structural_score={avg_score}")
            print(f"[INPUT {input_idx}] total_time={self._format_seconds(input_elapsed)}")
            print(f"[INPUT {input_idx}] end")

        all_elapsed = self._now() - all_start
        self._record("all_inputs:total", all_elapsed)
        final_score = self._mean_valid(input_scores)
        print(f"[ALL INPUTS] avg_structural_score={final_score}")
        print(f"[ALL INPUTS] total_time={self._format_seconds(all_elapsed)}")
        self.print_timing_summary()
        return final_score

    def evaluate_target_prompt_js_only(self, fn, inputs, target_prompt):
        all_start = self._now()
        input_scores = []

        for input_idx, input_data in enumerate(inputs):
            input_start = self._now()
            print(f"[INPUT {input_idx}] start")
            print(f"[INPUT {input_idx} RAW INPUT]")
            print(input_data)

            target_outputs = self._generate_outputs(fn, input_data, target_prompt, "target")
            if any(self._is_bad_output(output) for output in target_outputs):
                print(f"[INPUT {input_idx}] warning: empty target output found; invalid pairs will be skipped.")

            avg_score = self.eval_target_target_js_only(target_outputs, input_idx=input_idx)
            input_scores.append(avg_score)

            input_elapsed = self._now() - input_start
            self._record("input:total", input_elapsed)
            print(f"[INPUT {input_idx}] avg_structural_score={avg_score}")
            print(f"[INPUT {input_idx}] total_time={self._format_seconds(input_elapsed)}")
            print(f"[INPUT {input_idx}] end")

        all_elapsed = self._now() - all_start
        self._record("all_inputs:total", all_elapsed)
        final_score = self._mean_valid(input_scores)
        print(f"[ALL INPUTS] avg_structural_score={final_score}")
        print(f"[ALL INPUTS] total_time={self._format_seconds(all_elapsed)}")
        self.print_timing_summary()
        return final_score

    def print_timing_summary(self):
        print("[SUMMARY] timing summary sorted by total time desc")
        rows = sorted(self.timings.items(), key=lambda item: item[1], reverse=True)
        for name, total in rows:
            count = self.counts[name]
            avg = total / count if count else 0
            print(
                f"[SUMMARY] {name}: total={self._format_seconds(total)}, "
                f"count={count}, avg={self._format_seconds(avg)}"
            )


def run_text_pair_demo(args):
    text1 = args.text1 or "The stolen prompt produces a concise marketing plan with goals and tactics."
    text2 = args.text2 or "The target prompt creates a brief marketing roadmap with objectives and action steps."
    debugger = StructuralJSDebugger(m=args.m, score_mode=args.score_mode)
    print("[DEMO] direct text pair JS debug")
    result = debugger.compute_pair_debug(text1, text2)
    debugger.print_timing_summary()
    print(f"[DEMO] result={result}")


def run_csv_debug(args):
    debugger = StructuralJSDebugger(m=args.m, score_mode=args.score_mode)
    scores = []
    csv_start = time.perf_counter()

    with open(args.csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row_idx, row in enumerate(reader):
            text1 = row.get(args.col1)
            text2 = row.get(args.col2)
            print(f"[CSV ROW {row_idx}] start")
            print(f"[CSV ROW {row_idx}] col1={args.col1}, col2={args.col2}")
            result = debugger.compute_pair_debug(text1, text2)
            scores.append(result["structural_score"])
            print(f"[CSV ROW {row_idx}] structural_score={result['structural_score']}")
            print(f"[CSV ROW {row_idx}] end")

    elapsed = time.perf_counter() - csv_start
    debugger._record("csv:total", elapsed)
    print(f"[CSV] avg_structural_score={debugger._mean_valid(scores)}")
    print(f"[CSV] total_time={debugger._format_seconds(elapsed)}")
    debugger.print_timing_summary()


def parse_args():
    parser = argparse.ArgumentParser(description="Debug structural Jensen-Shannon evaluation without semantic/syntactic models.")
    parser.add_argument("--text1", default=None, help="First generated text for direct JS debugging.")
    parser.add_argument("--text2", default=None, help="Second generated text for direct JS debugging.")
    parser.add_argument("--csv", default=None, help="CSV file containing already generated text pairs.")
    parser.add_argument("--col1", default="text1", help="CSV column for the first text.")
    parser.add_argument("--col2", default="text2", help="CSV column for the second text.")
    parser.add_argument("--m", default=3, type=int, help="Sampling count used by fn-based debug methods.")
    parser.add_argument("--score-mode", choices=["inverse", "raw"], default="inverse", help="Use original 1 / JS distance score or raw JS distance.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.csv:
        run_csv_debug(cli_args)
    else:
        run_text_pair_demo(cli_args)
