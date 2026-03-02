import torch
from transformers import AutoModelForCausalLM
# measure the time required for context length 1200, 12000, 120000, and prediction length 60, 600, 6000. 3 * 3 combinations

from time import time

model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cuda",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True,
)


context_length = [1200, 12000]
prediction_length_list = [60, 600, 6000]
for cl in context_length:
    for pl in prediction_length_list:
        print(f"Context length: {cl}, Prediction length: {pl}")

        seqs = torch.randn(2, cl)  # tensor shape is [batch_size, context_length]


# use it when the flash-attn is available
# model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="cuda", attn_implementation='flash_attention_2', trust_remote_code=True)

        # normalize seqs
        mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
        normed_seqs = (seqs - mean) / std

# forecast
        prediction_length = pl
        time_start = time()
        output = model.generate(normed_seqs.to("cuda"), max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
        time_end = time()
        print(f"Inference time: {time_end - time_start:.4f} seconds")
        print(f"Output shape: {output.shape}\n")
        normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]

# inverse normalize
        # predictions = normed_predictions * std.to("cuda") + mean.to("cuda")  # shape is [batch_size, 6]