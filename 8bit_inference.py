#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# name = "bigscience/bloom-7b1"
name = "bigscience/bloom"
text = "Hello my name is"
max_new_tokens = 20

# breakpoint()
# pipe = pipeline(model=name, model_kwargs= {"device_map": "auto", "load_in_8bit": True}, max_new_tokens=max_new_tokens)
# print(pipe(text))

def generate_from_model(model, tokenizer):
  encoded_input = tokenizer(text, return_tensors='pt')
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), max_new_tokens=64)
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)

print(generate_from_model(model, tokenizer))

# mem_int8 = model_8bit.get_memory_footprint()
# print(f"Memory footprint int8 model: {mem_int8}")

# model_native = AutoModelForCausalLM.from_pretrained(name, device_map="auto", torch_dtype="auto")
# mem_fp16 = model_native.get_memory_footprint()
# print(f"Memory footprint fp16 model: {mem_fp16}")