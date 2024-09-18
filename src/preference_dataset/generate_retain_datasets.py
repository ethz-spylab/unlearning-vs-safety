import json
import random
import os
from typing import Optional

import openai
from tqdm import tqdm
from datasets import load_dataset, Dataset
from pydantic import BaseModel

from .refusal_list import REFUSAL_LIST
from ..util.helpers import create_if_not_exists
from ..util.globals import OPENAI_ORG_ID, CUSTOM_DATASET_DIR, HF_USERNAME, SEED

MIN_ARTICLE_LENGTH = 1000 # in characters
MAX_ARTICLE_LENGTH = 15000 # in characters
HARD_TOKEN_LIMIT = 5e7 # 50M tokens

N_QUESTION_DESIRED = 1000
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"#"gpt-4o-2024-08-06"
SYSTEM_PROMPT = r"""
You are a large language model tasked with creating multiple-choice questions based on an article. Each question is supposed to have four answers: A, B, C, D. Only one of these options is supposed to be correct, other three are supposed to be plausible but incorrect distractors. Your goal is to generate clear, concise questions that test knowledge contained within the article. You are not supposed to test text comprehension but rather concepts within it. Make sure that questions can be answered without an access to the article. Do not ask about data and statistics such as how many people died from polio in total. Instead ask 'Which of the following viruses causes polio?'. Never refer to the article directly. Always provide a correct answer and corresponding explanation. Make sure that the answer is one of the options. Don't include option letters A,B,C,D as part of each option. Instead it should be a phrase. Provide a list of 10 such questions. 

### EXAMPLE

ARTICLE:
Introduction {#bjh13767-sec-0001} ============ Purine analogues have revolutionized the treatment of hairy cell leukaemia (HCL) with overall responses in more than 85% of patients and a median progression‐free survival (PFS) of up to 15 years (Else *et al*, [2009](#bjh13767-bib-0004){ref-type="ref"}; Grever, [2010](#bjh13767-bib-0006){ref-type="ref"}). Despite the success of pentostatin and cladribine, PFS curves have revealed no plateau in long‐term follow‐up studies (Else *et al*, [2009](#bjh13767-bib-0004){ref-type="ref"}). In addition, there is a small subset of patients with primary refractory disease. Second and third line treatment with single‐agent purine analogues produce lower complete remission (CR) rates and shorter PFS (Chadha *et al*, [2005](#bjh13767-bib-0003){ref-type="ref"}). In that context, the combination of pentostatin or cladribine with rituximab was tested and found to be effective (Else *et al*, [2011](#bjh13767-bib-0005){ref-type="ref"}), and is now recommended in the UK for recurrent or refractory HCL (Jones *et al*, [2012](#bjh13767-bib-0007){ref-type="ref"}). 

QUESTION 1:
Which treatment approach has shown to be highly effective in achieving long-term progression-free survival (PFS) in patients with hairy cell leukemia (HCL)?

OPTION A:
Stem cell transplantation
OPTION B:
Combination of chemotherapy and radiation
OPTION C:
Purine analogues
OPTION D:
Monoclonal antibody therapy alone

ANSWER:
Purine analogues

EXPLANATION:
Purine analogues, specifically cladribine (2-CdA) and pentostatin, have been the standard of care for HCL and have been associated with high rates of complete remission and long-term PFS.

### END OF EXAMPLE
"""


class MCQ(BaseModel):
    question: str
    options: list[str]
    answer: str
    explanation: str
    
class ListMCQ(BaseModel):
    multiple_choice_questions: list[MCQ]

def format_question(mc_question: MCQ, context: str) -> str:
    assert mc_question.answer in mc_question.options, "Answer not in options"
    assert len(mc_question.options) == 4, "There should be exactly 4 options"
    
    question = f"The following are multiple choice questions (with answers) about {context}.\n\n"
    question += f"{mc_question.question}\n"
    ABCD = list("ABCD")
    random.shuffle(mc_question.options)
    for i, option in enumerate(mc_question.options):
        question += f"{ABCD[i]}. {option}\n"
    question += "Answer:"
    
    correct_index = mc_question.options.index(mc_question.answer)
    answer = f"{ABCD[correct_index]}. {mc_question.answer}\n"
    explanation = f"Explanation:\n{mc_question.explanation}"
    return question, answer, explanation

def prompt_gpt(client, article: str, context: str) -> Optional[MCQ]:
    completion = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Write 10 multiple-choice questions in the field of {context} based on the following article. Try to focus on {context} only and avoid related fields.\nARTICLE:\n{article}"},
        ],
        response_format=ListMCQ,
    )
    mc_question = completion.choices[0].message
    
    return None if (mc_question.refusal) else mc_question.parsed, completion.usage


if __name__ == "__main__":
    random.seed(SEED)
    client = openai.OpenAI(organization=OPENAI_ORG_ID)
    
    ### first generate questions based on bio retain dataset
    
    bio_retain_dpo = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    # load bio retain dataset
    bio_retain = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split="train")
    bio_retain = bio_retain.shuffle(seed=SEED)
    
    n_questions = 0
    tokens_used = 0
    for i, sample in enumerate(tqdm(bio_retain)):
        if len(sample["text"]) < MIN_ARTICLE_LENGTH:
            continue
        
        article = sample["text"][:MAX_ARTICLE_LENGTH]
        
        try:
            mc_question_list, usage = prompt_gpt(client, article, "biology")
        except Exception as e:
            continue
        
        if mc_question_list is None:
            print("Refused_to_answer on:", i)
            continue
        
        for mc in mc_question_list.multiple_choice_questions:
            try:
                question, answer, explanation = format_question(mc, "biology")
                bio_retain_dpo["prompt"].append(question)
                bio_retain_dpo["chosen"].append(answer)
                bio_retain_dpo["rejected"].append(random.sample(REFUSAL_LIST,1)[0])
                n_questions += 1
            except AssertionError as e:
                print(e)
                continue
        
        if n_questions >= N_QUESTION_DESIRED:
            print(f"Generated {n_questions} questions, stopping")
            break
            
        tokens_used += usage.total_tokens
        if tokens_used >= HARD_TOKEN_LIMIT:
            print(f"Used {tokens_used} tokens, stopping")
            break
    
        # save to json every iter to avoid losing progress
        create_if_not_exists(CUSTOM_DATASET_DIR)
        with open(os.path.join(CUSTOM_DATASET_DIR, "bio_retain_dpo.json"), "w") as f:
            json.dump(bio_retain_dpo, f)
    
    # upload to huggingface
    final_bio = Dataset.from_dict(bio_retain_dpo)
    final_bio.push_to_hub(f"{HF_USERNAME}/bio_retain_dpo", private=True) 
    
    print("Bio retain dataset generated and uploaded")
    
    ### Now generate questions based on cyber retain dataset
    
    cyber_retain_dpo = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    # load cyber retain dataset
    cyber_retain = load_dataset("cais/wmdp-corpora", "cyber-retain-corpus", split="train")
    cyber_retain = cyber_retain.shuffle(seed=SEED)
    
    n_questions = 0
    tokens_used = 0
    for i, sample in enumerate(tqdm(cyber_retain)):
        if len(sample["text"]) < MIN_ARTICLE_LENGTH:
            continue
        
        article = sample["text"][:MAX_ARTICLE_LENGTH]
        
        try:
            mc_question_list, usage = prompt_gpt(client, article, "cybersecurity")
        except Exception as e:
            continue
        
        if mc_question_list is None:
            print("Refused_to_answer on:", i)
            continue
        
        for mc in mc_question_list.multiple_choice_questions:
            try:
                question, answer, explanation = format_question(mc, "cybersecurity")
                cyber_retain_dpo["prompt"].append(question)
                cyber_retain_dpo["chosen"].append(answer)
                cyber_retain_dpo["rejected"].append(random.sample(REFUSAL_LIST,1)[0])
                n_questions += 1
            except AssertionError as e:
                print(e)
                continue
        
        if n_questions >= N_QUESTION_DESIRED:
            print(f"Generated {n_questions} questions, stopping")
            break
            
        tokens_used += usage.total_tokens
        if tokens_used >= HARD_TOKEN_LIMIT:
            print(f"Used {tokens_used} tokens, stopping")
            break
    
    # save to json
    with open(os.path.join(CUSTOM_DATASET_DIR, "cyber_retain_dpo.json"), "w") as f:
        json.dump(cyber_retain_dpo, f)
    
    # upload to huggingface
    final_bio = Dataset.from_dict(cyber_retain_dpo)
    final_bio.push_to_hub(f"{HF_USERNAME}/cyber_retain_dpo", private=True) 
    
    print("Cyber retain dataset generated and uploaded")