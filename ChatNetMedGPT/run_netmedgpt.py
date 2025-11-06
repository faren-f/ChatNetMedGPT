from ChatNetMedGPT.netmedgpt import netMedGpt

print("[run_netmedgpt] __file__ =", __file__)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import argparse
import os
import json
import pandas as pd
import torch
import torch.nn.functional as F
from src.model_pretrain import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", required=True, help="input file")
    parser.add_argument("--node_type", required=True, help="input file")
    parser.add_argument("--output", required=False, help="output file")
    parser.add_argument("--mask_index_question", required=False, default="first", help="Which mask to use: 0-based index (e.g., 0,1,2) or 'first'/'last'. Default: first")
    
    args = parser.parse_args()
    
    input = args.sentence
    node_type = args.node_type
    output_file = args.output
    mask_index_question = args.mask_index_question

    drug_names, user_response = netMedGpt(input, node_type, mask_index_question)

    drug_names_df = pd.DataFrame(drug_names, columns=["drug_name"])
    drug_names_df.to_csv(os.path.join(user_response,  "user_response.csv"), index = False)
    

    # print("Input:", input)
    print("Output:", drug_names_df)


if __name__ == '__main__':
    main()

##### example for runing in terminal:
#### python /home/bbc8731/BioMedFormer/run_netmedgpt.py   --sentence "129405,129381,129405,129389,129405,129387,129405,129405,129405"   --node_type drug   --mask_index_question 2   --output results.csv


#python netmedgpt_llm.py --user_text "for diabetes with egfr mutation what is the best treatment and what are the adverse drug reactions"


