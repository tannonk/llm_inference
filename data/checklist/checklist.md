# LLM Simplification Checklist
|      Model      |   Test   |Train (few-shot)|# samples|Prompt|# Ref|Seed|         Done?          |
|-----------------|----------|----------------|--------:|------|----:|---:|------------------------|
|bloom            |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloom            |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloom            |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloom            |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloom            |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloom            |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloom            |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloom            |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloom            |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloom-1b1        |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloom-1b1        |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloom-1b1        |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloom-1b1        |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloom-1b1        |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloom-1b1        |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloom-1b1        |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloom-1b1        |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloom-1b1        |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloom-3b         |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloom-3b         |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloom-3b         |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloom-3b         |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloom-3b         |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloom-3b         |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloom-3b         |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloom-3b         |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloom-3b         |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloom-560m       |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloom-560m       |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloom-560m       |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloom-560m       |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloom-560m       |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloom-560m       |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloom-560m       |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloom-560m       |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloom-560m       |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloom-560m.test  |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|bloom-560m.test  |asset-test|asset-valid     |        3|p0    |    1| 489|:heavy_multiplication_x:|
|bloom-560m.test  |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|bloom-560m.test  |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloom-560m.test  |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloom-560m.test  |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloom-560m.test  |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloom-560m.test  |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloom-560m.test  |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloom-7b1        |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloom-7b1        |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloom-7b1        |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloom-7b1        |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloom-7b1        |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloom-7b1        |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloom-7b1        |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloom-7b1        |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloom-7b1        |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloomz           |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloomz           |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloomz           |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloomz           |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloomz           |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloomz           |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloomz           |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloomz           |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloomz           |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloomz-1b1       |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloomz-1b1       |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloomz-1b1       |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloomz-1b1       |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloomz-1b1       |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloomz-1b1       |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloomz-1b1       |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloomz-1b1       |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloomz-1b1       |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloomz-3b        |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloomz-3b        |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloomz-3b        |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloomz-3b        |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloomz-3b        |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloomz-3b        |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloomz-3b        |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloomz-3b        |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloomz-3b        |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloomz-560m      |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloomz-560m      |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloomz-560m      |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloomz-560m      |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloomz-560m      |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloomz-560m      |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloomz-560m      |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloomz-560m      |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloomz-560m      |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|bloomz-7b1       |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|bloomz-7b1       |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|bloomz-7b1       |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|bloomz-7b1       |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|bloomz-7b1       |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|bloomz-7b1       |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|bloomz-7b1       |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|bloomz-7b1       |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|bloomz-7b1       |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|dummy            |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|dummy            |asset-test|asset-valid     |        3|p0    |    1| 489|:heavy_multiplication_x:|
|dummy            |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|dummy            |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|dummy            |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|dummy            |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|dummy            |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|dummy            |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|dummy            |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|flan-t5-base     |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|flan-t5-base     |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|flan-t5-base     |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|flan-t5-base     |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|flan-t5-base     |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|flan-t5-base     |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|flan-t5-base     |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|flan-t5-base     |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|flan-t5-base     |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|flan-t5-large    |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|flan-t5-large    |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|flan-t5-large    |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|flan-t5-large    |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|flan-t5-large    |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|flan-t5-large    |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|flan-t5-large    |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|flan-t5-large    |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|flan-t5-large    |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|flan-t5-small    |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|flan-t5-small    |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|flan-t5-small    |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|flan-t5-small    |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|flan-t5-small    |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|flan-t5-small    |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|flan-t5-small    |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|flan-t5-small    |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|flan-t5-small    |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|flan-t5-xl       |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|flan-t5-xl       |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|flan-t5-xl       |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|flan-t5-xl       |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|flan-t5-xl       |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|flan-t5-xl       |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|flan-t5-xl       |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|flan-t5-xl       |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|flan-t5-xl       |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|flan-t5-xxl      |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|flan-t5-xxl      |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|flan-t5-xxl      |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|flan-t5-xxl      |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|flan-t5-xxl      |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|flan-t5-xxl      |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|flan-t5-xxl      |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|flan-t5-xxl      |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|flan-t5-xxl      |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|flan-ul2         |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|flan-ul2         |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|flan-ul2         |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|flan-ul2         |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|flan-ul2         |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|flan-ul2         |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|flan-ul2         |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|flan-ul2         |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|flan-ul2         |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|gpt-j-6b         |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|gpt-j-6b         |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|gpt-j-6b         |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|gpt-j-6b         |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|gpt-j-6b         |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|gpt-j-6b         |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|gpt-j-6b         |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|gpt-j-6b         |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|gpt-j-6b         |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|gpt-neox-20b     |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|gpt-neox-20b     |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|gpt-neox-20b     |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|gpt-neox-20b     |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|gpt-neox-20b     |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|gpt-neox-20b     |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|gpt-neox-20b     |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|gpt-neox-20b     |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|gpt-neox-20b     |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|llama-13b        |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|llama-13b        |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|llama-13b        |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|llama-13b        |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|llama-13b        |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|llama-13b        |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|llama-13b        |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|llama-13b        |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|llama-13b        |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|llama-30b        |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|llama-30b        |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|llama-30b        |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|llama-30b        |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|llama-30b        |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|llama-30b        |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|llama-30b        |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|llama-30b        |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|llama-30b        |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|llama-65b        |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|llama-65b        |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|llama-65b        |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|llama-65b        |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|llama-65b        |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|llama-65b        |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|llama-65b        |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|llama-65b        |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|llama-65b        |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|llama-7b         |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|llama-7b         |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|llama-7b         |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|llama-7b         |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|llama-7b         |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|llama-7b         |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|llama-7b         |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|llama-7b         |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|llama-7b         |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|opt-1.3b         |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|opt-1.3b         |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|opt-1.3b         |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|opt-1.3b         |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|opt-1.3b         |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|opt-1.3b         |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|opt-1.3b         |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|opt-1.3b         |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|opt-1.3b         |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|opt-13b          |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|opt-13b          |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|opt-13b          |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|opt-13b          |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|opt-13b          |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|opt-13b          |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|opt-13b          |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|opt-13b          |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|opt-13b          |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|opt-30b          |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|opt-30b          |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|opt-30b          |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|opt-30b          |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|opt-30b          |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|opt-30b          |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|opt-30b          |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|opt-30b          |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|opt-30b          |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|opt-6.7b         |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|opt-6.7b         |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|opt-6.7b         |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|opt-6.7b         |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|opt-6.7b         |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|opt-6.7b         |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|opt-6.7b         |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|opt-6.7b         |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|opt-6.7b         |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|opt-66b          |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|opt-66b          |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|opt-66b          |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|opt-66b          |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|opt-66b          |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|opt-66b          |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|opt-66b          |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|opt-66b          |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|opt-66b          |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|opt-iml-max-1.3b |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|opt-iml-max-1.3b |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|opt-iml-max-1.3b |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|opt-iml-max-1.3b |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|opt-iml-max-1.3b |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|opt-iml-max-1.3b |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|opt-iml-max-1.3b |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|opt-iml-max-1.3b |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|opt-iml-max-1.3b |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|opt-iml-max-30b  |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|opt-iml-max-30b  |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|opt-iml-max-30b  |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|opt-iml-max-30b  |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|opt-iml-max-30b  |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|opt-iml-max-30b  |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|opt-iml-max-30b  |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|opt-iml-max-30b  |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|opt-iml-max-30b  |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t0               |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|t0               |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t0               |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|t0               |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t0               |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t0               |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t0               |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t0               |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t0               |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t0-3b            |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|t0-3b            |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t0-3b            |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|t0-3b            |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t0-3b            |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t0-3b            |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t0-3b            |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t0-3b            |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t0-3b            |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t0pp             |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|t0pp             |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t0pp             |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|t0pp             |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t0pp             |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t0pp             |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t0pp             |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t0pp             |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t0pp             |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-base          |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-base          |asset-test|asset-valid     |        3|p0    |    1| 489|:heavy_multiplication_x:|
|t5-base          |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-base          |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-base          |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-base          |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-base          |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-base          |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-base          |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-base-lm-adapt |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|t5-base-lm-adapt |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-base-lm-adapt |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|t5-base-lm-adapt |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-base-lm-adapt |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-base-lm-adapt |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-base-lm-adapt |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-base-lm-adapt |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-base-lm-adapt |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-large         |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-large         |asset-test|asset-valid     |        3|p0    |    1| 489|:heavy_multiplication_x:|
|t5-large         |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-large         |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-large         |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-large         |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-large         |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-large         |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-large         |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-large-lm-adapt|asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|t5-large-lm-adapt|asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-large-lm-adapt|asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|t5-large-lm-adapt|asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-large-lm-adapt|asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-large-lm-adapt|asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-large-lm-adapt|asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-large-lm-adapt|asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-large-lm-adapt|asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-small         |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-small         |asset-test|asset-valid     |        3|p0    |    1| 489|:heavy_multiplication_x:|
|t5-small         |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-small         |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-small         |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-small         |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-small         |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-small         |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-small         |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-small-lm-adapt|asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|t5-small-lm-adapt|asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-small-lm-adapt|asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|t5-small-lm-adapt|asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-small-lm-adapt|asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-small-lm-adapt|asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-small-lm-adapt|asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-small-lm-adapt|asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-small-lm-adapt|asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-base     |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-base     |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-v1-1-base     |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-base     |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-base     |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-base     |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-base     |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-base     |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-base     |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-large    |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-large    |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-v1-1-large    |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-large    |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-large    |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-large    |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-large    |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-large    |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-large    |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-small    |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-small    |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-v1-1-small    |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-small    |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-small    |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-small    |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-small    |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-small    |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-small    |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-xl       |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-xl       |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-v1-1-xl       |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-xl       |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-xl       |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-xl       |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-xl       |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-xl       |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-xl       |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-xxl      |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-xxl      |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-v1-1-xxl      |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-xxl      |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-xxl      |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-xxl      |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-v1-1-xxl      |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-v1-1-xxl      |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-v1-1-xxl      |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-xl            |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-xl            |asset-test|asset-valid     |        3|p0    |    1| 489|:heavy_multiplication_x:|
|t5-xl            |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-xl            |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-xl            |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-xl            |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-xl            |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-xl            |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-xl            |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-xl-lm-adapt   |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|t5-xl-lm-adapt   |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-xl-lm-adapt   |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|t5-xl-lm-adapt   |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-xl-lm-adapt   |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-xl-lm-adapt   |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-xl-lm-adapt   |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-xl-lm-adapt   |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-xl-lm-adapt   |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-xxl           |asset-test|asset-valid     |        3|p0    |    1| 287|:heavy_multiplication_x:|
|t5-xxl           |asset-test|asset-valid     |        3|p0    |    1| 489|:heavy_multiplication_x:|
|t5-xxl           |asset-test|asset-valid     |        3|p0    |    1| 723|:heavy_multiplication_x:|
|t5-xxl           |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-xxl           |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-xxl           |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-xxl           |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-xxl           |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-xxl           |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|t5-xxl-lm-adapt  |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|t5-xxl-lm-adapt  |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|t5-xxl-lm-adapt  |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|t5-xxl-lm-adapt  |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|t5-xxl-lm-adapt  |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|t5-xxl-lm-adapt  |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|t5-xxl-lm-adapt  |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|t5-xxl-lm-adapt  |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|t5-xxl-lm-adapt  |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
|ul2              |asset-test|asset-valid     |        3|p0    |    1| 287|:white_check_mark:      |
|ul2              |asset-test|asset-valid     |        3|p0    |    1| 489|:white_check_mark:      |
|ul2              |asset-test|asset-valid     |        3|p0    |    1| 723|:white_check_mark:      |
|ul2              |asset-test|asset-valid     |        3|p1    |    1| 287|:heavy_multiplication_x:|
|ul2              |asset-test|asset-valid     |        3|p1    |    1| 489|:heavy_multiplication_x:|
|ul2              |asset-test|asset-valid     |        3|p1    |    1| 723|:heavy_multiplication_x:|
|ul2              |asset-test|asset-valid     |        3|p2    |    1| 287|:heavy_multiplication_x:|
|ul2              |asset-test|asset-valid     |        3|p2    |    1| 489|:heavy_multiplication_x:|
|ul2              |asset-test|asset-valid     |        3|p2    |    1| 723|:heavy_multiplication_x:|
