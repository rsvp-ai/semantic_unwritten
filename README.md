[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repo contains codes and pre-trained models for our paper 

> [Semantics of the Unwritten: The Effect of End of Paragraph and Sequence Tokens on Text Generation with GPT2](https://arxiv.org/abs/2004.02251)
>
> He Bai, Peng Shi, Jimmy Lin, Luchen Tan, Kun Xiong, Wen Gao, Jie Liu, Ming Li
>
> ACL SRW 2021

## ChineseEassy

```bash
./examples/passage_generation/bash_run_essay.sh $1 $2 $3
```

This scripts support train, eval, test, and generation 4 modes by changing the value of **TRAIN** **EVAL** **TEST** and **GENERATE** in this bash script.

The position arguments **$1** is the **input type**, and could be **paragraph**, and **passage**. **paragraph** means add paragraph breaker, while **passage** means without paragraph breaker.

The position arguments **$2** the paragraph breaker type, True means paragraph seperator(SEP), False is the end of paragraph(EOP).

The position arguments **$3** is the visible cuda devices.

## EnglishStory(WritingPrompts) 

For writingprompts, we replace all "<newline> <newline>" into paragraph seperator, which can be find in preprocess.ipynb file.

```
./examples/passage_generation/bash_run_wp.sh $1 $2 $3
```

This scripts support train, eval, test, and generation 4 modes by changing the value of **TRAIN** **EVAL** **TEST** and **GENERATE** in this bash script.

The position arguments **$1** is the **paragraph breaker type**, and could be **none**, **newline**, and **eos**. **none** means no paragraph breaker, **newline** is \n, **eos** is the DIY paragraph breaker.

The position arguments **$2** the paragraph breaker type, True means paragraph seperator(SEP), False is the end of paragraph(EOP).

The position arguments **$3** is the visible cuda devices.
