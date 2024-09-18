from typing import Optional, Literal, Union

from sympy import use
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

"""Modified from: https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb"""

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, proj=None):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.proj = proj

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None
        self.block_output = None


    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if self.proj is not None:
            output = tuple([item @ self.proj if i == 0 else item for i, item in enumerate(output)])
        self.block_output = output[0]
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))
        attn_output += args[0]
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output))
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations

class Llama2Wrapper:
    def __init__(self, model, tokenizer, projection_matrices:Optional[dict] = None, use_chat_template:bool = False):
        self.device = model.device
        self.tokenizer = tokenizer
        self.model = model
        self.use_chat_template = use_chat_template
        for i, layer in enumerate(self.model.model.layers):
            proj = None
            if projection_matrices is not None and i in projection_matrices:
                proj = torch.Tensor(projection_matrices[i]).to(device=self.device, dtype=self.model.dtype)
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm, proj)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_logits(self, prompt):
        if self.use_chat_template:
            inputs = self.tokenizer.apply_chat_template([{"role":"system", "content": ""}, {"role":"user", "content": prompt}], return_tensors="pt", add_generation_prompt=True)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
          logits = self.model(inputs.to(self.device)).logits
          return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        probs_percent = [f"{v:.03f}" for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(tokens, probs_percent)))


    def decode_all_layers(self, text, topk=10, print_attn_mech=True, print_intermediate_res=True, print_mlp=True, print_block=True):
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            print(f'Layer {i}: Decoded intermediate outputs')
            if print_attn_mech:
                self.print_decoded_activations(layer.attn_mech_output_unembedded, 'Attention mechanism')
            if print_intermediate_res:
                self.print_decoded_activations(layer.intermediate_res_unembedded, 'Intermediate residual stream')
            if print_mlp:
                self.print_decoded_activations(layer.mlp_output_unembedded, 'MLP output')
            if print_block:
                self.print_decoded_activations(layer.block_output_unembedded, 'Block output')
    
    def get_decoded_activations(self, activations, search_tokens: Optional[list[int]] = None, topk: Optional[int] = None):
        assert search_tokens is None or topk is None, "Only top-k or search tokens can be selected"
        softmaxed = torch.nn.functional.softmax(activations[0][-1], dim=-1)
        
        values = softmaxed
        indices = torch.arange(softmaxed.shape[-1])
        
        if search_tokens is not None:
            values = softmaxed[search_tokens]
            indices = torch.Tensor(search_tokens).int()
        
        if topk is not None:
            values, indices = torch.topk(softmaxed, 10)
        
        probs_percent = [f"{v:.06f}" for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        
        return sorted(list(zip(tokens, probs_percent)), key=lambda x: float(x[1]), reverse=True)
    
    def decode_nlayers(self, 
                      prompt: str, 
                      n:int, 
                      search_tokens: Optional[list[int]] = None, 
                      topk: Optional[int] = None, 
                      print_mode: bool = True) -> dict[str, list]:
        
        self.get_logits(prompt)
        
        out = {"block": [], "mlp": [], "inter_res": [], "attention": []}
        for i, layer in enumerate(self.model.model.layers[:n]):
            decoded_block = self.get_decoded_activations(layer.block_output_unembedded, search_tokens, topk)
            out["block"].append(decoded_block)
            
            decoded_mlp = self.get_decoded_activations(layer.mlp_output_unembedded, search_tokens, topk)
            out["mlp"].append(decoded_mlp)
            
            decoded_inter_res = self.get_decoded_activations(layer.intermediate_res_unembedded, search_tokens, topk)
            out["inter_res"].append(decoded_inter_res)
            
            decoded_attention = self.get_decoded_activations(layer.attn_mech_output_unembedded, search_tokens, topk)
            out["attention"].append(decoded_attention)
            
            if print_mode:
                print(f"{i}. \t{decoded_block}\t{decoded_mlp}\t{decoded_inter_res}\t{decoded_attention}")

        return out
        
                
                
            