import transformers


models = [
    # "J4Q8/zephyr-npo-cyber",
    # "J4Q8/zephyr-npo-bio",
    # "J4Q8/zephyr-dpo-cyber",
    "J4Q8/zephyr-dpo-bio",
]

for model in models:
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(model)
    loaded_model.push_to_hub(model.split("/")[1] + "-priv1")