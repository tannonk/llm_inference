#!/usr/bin/env python3



inputs = 'a b c'.split()
outputs = 'd e f g h i'.split()

batch_size = len(inputs)
num_return_sequences = len(outputs)
return_seqs_per_input = int(num_return_sequences/batch_size)

breakpoint()

# a = [outputs[i:i+num_return_sequences] for i in range(0, len(outputs), num_return_sequences)]
# print(a)

for i in range(0, num_return_sequences, return_seqs_per_input):
    print(i)
    print(outputs[i:i+return_seqs_per_input])


outputs = [outputs[i:i+return_seqs_per_input]for i in range(0, num_return_sequences, return_seqs_per_input)]
print(outputs)

