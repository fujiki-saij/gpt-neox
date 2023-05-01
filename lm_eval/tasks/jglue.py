"""
JGLUE: Japanese General Language Understanding Evaluation
https://aclanthology.org/2022.lrec-1.317/

JGLUE, Japanese General Language Understanding Evaluation, is built to measure the general NLU ability in Japanese. 
JGLUE has been constructed from scratch without translation. 

Homepage: https://github.com/yahoojapan/JGLUE
"""
import datasets
from math import exp
from lm_eval.base import rf, Task, MultipleChoiceTask
from lm_eval.tasks.squad import _squad_agg
from functools import partial


_CITATION = """
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""



class JSQuAD(Task):
    """
    prompt format refered to [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """
    VERSION = 0
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JSQuAD"
    SEP = "\n" # "\n" is not detected in rinna's tokenizer
    REMOVE_IDS = []
    # REMOVE_IDS = ['a10743p19q0', 'a10743p19q1', 'a10743p19q2', 'a10743p19q3', 'a13221p1q0', 'a13221p1q1', 'a13221p1q2', 'a13221p1q3', 'a14985p1q0', 'a14985p1q1', 'a14985p1q2', 'a14985p1q3', 'a14985p1q4', 'a14985p93q0', 'a14985p93q1', 'a14985p93q2', 'a14985p93q3', 'a14985p93q4', 'a1540503p36q0', 'a1540503p36q1', 'a1540503p36q2', 'a1540503p36q3', 'a1540503p36q4', 'a18783p1q0', 'a18783p3q0', 'a18783p3q1', 'a18783p3q2', 'a18783p8q0', 'a18873p25q0', 'a18873p25q1', 'a18873p25q2', 'a18873p25q3', 'a18873p26q0', 'a18873p26q1', 'a18873p26q2', 'a20898p10q0', 'a20898p15q0', 'a20898p15q1', 'a20898p15q2', 'a20898p15q3', 'a2164640p22q0', 'a2164640p22q1', 'a2164640p22q2', 'a2164640p22q3', 'a2164640p22q4', 'a22392p20q0', 'a22392p20q1', 'a22392p20q2', 'a22392p20q3', 'a3011628p3q0', 'a3011628p3q1', 'a3011628p3q2', 'a3011628p3q3', 'a3189p4q0', 'a3189p4q1', 'a3189p4q2', 'a369953p0q0', 'a369953p0q1', 'a369953p0q2', 'a369953p0q3', 'a3949p1q0', 'a3949p1q1', 'a4596p0q0', 'a4596p0q1', 'a4596p0q2', 'a4596p0q3', 'a4596p1q0', 'a4596p1q1', 'a4596p1q2', 'a4596p1q3', 'a4596p1q4', 'a4596p38q0', 'a4596p38q1', 'a4596p38q2', 'a4596p38q3', 'a4596p38q4', 'a4768p13q0', 'a4768p13q1', 'a4768p13q2', 'a4768p3q0', 'a4768p3q1', 'a4768p3q2', 'a4768p3q3', 'a4768p8q0', 'a4768p8q1', 'a4768p8q2', 'a51481p0q0', 'a51481p0q1', 'a51481p0q2', 'a51481p10q0', 'a51481p10q1', 'a51481p10q2', 'a51481p10q3', 'a51481p6q0', 'a51481p6q1', 'a51481p6q2', 'a51481p6q3', 'a51481p7q0', 'a51481p7q1', 'a67892p11q0', 'a67892p11q1', 'a67892p11q2', 'a67892p11q3', 'a67892p2q0', 'a8874p6q0', 'a8874p6q1', 'a916079p3q0', 'a916079p3q1', 'a95156p4q0', 'a95156p4q1', 'a95156p4q2', 'a95156p4q3', 'a95156p6q0', 'a95156p6q1', 'a95156p6q2', 'a95156p6q3']
    """
    @mkshing's comment
    I found that JSQuAD contains errors inside contexts such as below. 
    ```
    {'id': 'a4596p0q0', 'title': 'ポルトガル', 'context': 'ポルトガル [SEP] 正式名称はポルトガル語で、。通称、 。', 'question': 'ポルトガルね正式名称は何語であるか', 'answers': {'text': ['正式名称はポルトガル語', 'ポルトガル語', 'ポルトガル語'], 'answer_start': [12, 17, 17]}, 'is_impossible': False}
    ```
    So, I tried to identify all of them and found that the following processing can be okay to detect the ids 
    ```python
    from datasets import load_dataset
    from transformers import T5Tokenizer
    dataset = load_dataset("shunk031/JGLUE", name="JSQuAD", split="validation")
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
    remove_ids = []
    for item in dataset:
        ctx = item["context"].split("[SEP]")[-1].strip()
        input_ids = tokenizer.encode(ctx, add_special_tokens=False)
        if len(input_ids) < 25:
            print(item)
            remove_ids.append(item["id"])
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def has_training_docs(self):
        return True
    
    def has_validation_docs(self):
        return True
    
    def has_test_docs(self):
        return False
    
    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        dataset = self.dataset["validation"]
        if len(self.REMOVE_IDS) > 0:
            dataset = [item for item in dataset if item["id"] not in self.REMOVE_IDS]
        return dataset
    
    def doc_to_text(self, doc):
        return (
            "[題名]:"
            + doc["title"]
            + f"{self.SEP}"
            + "[問題]:"
            + doc["context"].split("[SEP]")[-1].strip()
            + f"{self.SEP}"
            + "[質問]:"
            + doc["question"]
            + f"{self.SEP}"
            + "[答え]:"
        )

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        answer = answer_list[0]
        # if len(answer_list) > 0:
        #     answer = answer_list[0]
        # else:
        #     answer = "unanswerable"
        return " " + answer

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        prompt = "[題名]と[問題]から[質問]に対する[答え]を抜き出しなさい\n\n"
        ctx = prompt + ctx
        continuation = rf.greedy_until(ctx, [self.SEP])
        # is_unanswerable = rf.loglikelihood(ctx, " " + "unanswerable")
        # return continuation, is_unanswerable
        return continuation

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # continuation, (logprob_unanswerable, _) = results
        # no_answer_probability = exp(logprob_unanswerable)
        continuation = results
        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
            # "no_answer_probability": no_answer_probability,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }
        return {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            # "HasAns_exact": (
            #     predictions,
            #     references,
            # ),  # Exact match (the normalized answer exactly match the gold answer)
            # "HasAns_f1": (
            #     predictions,
            #     references,
            # ),  # The F-score of predicted tokens versus the gold answer
            # "NoAns_exact": (
            #     predictions,
            #     references,
            # ),  # Exact match (the normalized answer exactly match the gold answer)
            # "NoAns_f1": (
            #     predictions,
            #     references,
            # ),  # The F-score of predicted tokens versus the gold answer
            # "best_exact": (
            #     predictions,
            #     references,
            # ),  # Best exact match (with varying threshold)
            # "best_f1": (predictions, references),  # Best F1 (with varying threshold)
        }


    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return {
            "exact": partial(
                _squad_agg, "exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                _squad_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
            # "HasAns_exact": partial(
            #     _squad_agg, "HasAns_exact"
            # ),  # Exact match (the normalized answer exactly match the gold answer)
            # "HasAns_f1": partial(
            #     _squad_agg, "HasAns_f1"
            # ),  # The F-score of predicted tokens versus the gold answer
            # "NoAns_exact": partial(
            #     _squad_agg, "NoAns_exact"
            # ),  # Exact match (the normalized answer exactly match the gold answer)
            # "NoAns_f1": partial(
            #     _squad_agg, "NoAns_f1"
            # ),  # The F-score of predicted tokens versus the gold answer
            # "best_exact": partial(
            #     _squad_agg, "best_exact"
            # ),  # Best exact match (with varying threshold)
            # "best_f1": partial(
            #     _squad_agg, "best_f1"
            # ),  # Best F1 (with varying threshold)
        }
    
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
            # "HasAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            # "HasAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            # "NoAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            # "NoAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            # "best_exact": True,  # Best exact match (with varying threshold)
            # "best_f1": True,  # Best F1 (with varying threshold)
        }


class JaQuAD(JSQuAD):
    DATASET_PATH = "SkelterLabsInc/JaQuAD"
    DATASET_NAME = None

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]
    

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        if "answer_type" in doc["answers"]:
            doc["answers"].pop("answer_type")
        return JSQuAD.process_results(self, doc, results)


class JCommonsenseQA(MultipleChoiceTask):
    """
    prompt format refered to [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """
    VERSION = 0
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JCommonsenseQA"
    PROMPT = "[問題]に対する[答え]を[選択肢]の中から選んでください。\n\n"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        # TODO: Process the documents into a dictionary with the following keys:
        return {
            "query": doc["question"],
            "choices": [doc['choice0'], doc["choice1"], doc["choice2"], doc["choice3"], doc["choice4"]],  # The list of choices.
            "gold": doc["label"], 
        }

    def doc_to_text(self, doc):
        """
        [問題]: query
        [選択肢]:
        0. choice0
        1. choice1
        ...
        4. choice4
        [答え]:
        """
        # return f"質問: {doc['query']}\n\n回答:"
        # flat_choices = "".join([f"{idx}. {c}\n"for idx, c in enumerate(doc["choices"])])
        choices_str = str(doc['choices'])
        return f"[問題]: {doc['query']}\n[選択肢]: {choices_str}\n[答え]: "
    
    def construct_requests(self, doc, ctx):
        ctx = self.PROMPT + ctx
        return super().construct_requests(doc, ctx)        
    
    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]