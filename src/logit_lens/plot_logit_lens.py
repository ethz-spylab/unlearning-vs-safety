import os
import json
import time

from matplotlib import legend
from networkx import barbell_graph
from sympy import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd 
import numpy as np
from argparse import ArgumentParser

from src.benchmarking.benchmark import Benchmark
from src.util.globals import OUTPUT_DIR
from src.util.helpers import create_if_not_exists, save_as_json
from src.logit_lens.logit_lens_utils import Llama2Wrapper

def plot_activations(df, task, act_t, plot_dir, chat_str=""):
    color_dict = {
    'RMU': px.colors.qualitative.G10[1],
    'DPO': px.colors.qualitative.G10[0],
    'NPO': px.colors.qualitative.G10[3],
    'No unlearning': px.colors.qualitative.G10[2],
    }
    model_names = {
    'Zephyr_RMU': 'RMU',
    'zephyr-dpo-cyber': 'DPO',
    'zephyr-npo-cyber': 'NPO',
    'zephyr-dpo-bio': 'DPO',
    'zephyr-npo-bio': 'NPO',
    "zephyr-7b-beta": "No unlearning"
    }
    baseline ={
        "wmdp-bio": 0.6441476826394344,
        "wmdp-cyber": 0.42627
    }
    
    df['model'] = df['model'].replace(model_names)
    df = df.drop_duplicates().sort_values(by="model")
    
    fig = go.Figure()
    
    for model_name in df['model'].unique():
        fig.add_trace(
            go.Bar(
                x=df[df['model'] == model_name]['layer'],  # x values: layers corresponding to the specific model
                y=df[df['model'] == model_name]['accuracy'],  # y values: accuracy for that model
                name=model_name,  # legend entry for each model
                marker_color=color_dict[model_name],  # color based on the color dictionary
                legendgroup="methods",
                legendgrouptitle_text="Methods",
            )
        )
        
    fig.add_trace(go.Scatter(x=[-0.5, 31.5], y=[baseline[task], baseline[task]], mode="lines", name="Using logits", line=dict(color="black", dash="dot"), legendgroup="baselines", opacity=0.7))
    fig.add_trace(go.Scatter(x=[-0.5, 31.5], y=[0.25, 0.25], mode="lines", name="Random chance", line=dict(color="black", dash="dash"), legendgroup="baselines", legendgrouptitle_text="Baselines", opacity=0.7))

    # Set the layout properties of the figure
    fig.update_layout(
        barmode='group',  # grouped bars
        width=1800,
        height=500,
        xaxis_title="Transformer block",  # setting the x-axis title
        yaxis_title=f"Accuracy on WMDP-{task.split('-')[-1].capitalize()}",  # y-axis title
    )

    # fig = px.bar(df, x="layer", y="accuracy",
    #             color='model', barmode='group',
    #             color_discrete_map=color_dict,
    #             labels={
    #                  "layer": "Transformer block",
    #                  "accuracy": f"Accuracy on WMDP-{task.split("-")[-1].capitalize()}",
    #                  "model": "Methods"
    #              },
    #             width=1500, height=400)
    
     
    fig.update_layout(bargroupgap=0.1, bargap=0.25, font=dict(size=26))
    fig.write_image(os.path.join(plot_dir, f"lens_{task}_{act_t}{chat_str}.pdf"), format="pdf")
    time.sleep(2)
    fig.write_image(os.path.join(plot_dir, f"lens_{task}_{act_t}{chat_str}.pdf"), format="pdf")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", choices=["wmdp-bio", "wmdp-cyber"])
    parser.add_argument("--token_rank", type=int, default=0, help="0 is the token with highest logit-prob, 1 the second highest, etc.")
    parser.add_argument("--apply_chat_template", action="store_true")
    args = parser.parse_args()
    
    chat_str = "_chat" if args.apply_chat_template else ""
    jsonl_file = os.path.join(OUTPUT_DIR, "logit_lens", f"results{chat_str}", "results.jsonl")
    df = pd.read_json(jsonl_file, lines=True)

    plot_dir = os.path.join(OUTPUT_DIR, "logit_lens", f"plots{chat_str}")
    create_if_not_exists(plot_dir)
    
    act_types = ["block", "mlp", "inter_res", "attention"]
    for act_t in act_types:
        filtered_ds = df[(df["task"] == args.task) & (df["position"] == args.token_rank) & (df["act_type"] == act_t)]
        filtered_ds = filtered_ds.drop(columns=["task", "position", "act_type"])
        plot_activations(filtered_ds, args.task, act_t, plot_dir, chat_str)