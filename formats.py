FORMATS = {
    'ChatML': {
      'sysprompt': "<|im_start|>system\n%s<|im_end|>\n",
      'user_template': "<|im_start|>user\n{prompt}<|im_end|>",
      'bot_template': "<|im_start|>assistant\n{prompt}<|im_end|>",
      'end_tag': "<|im_end|>",
      'stopwords': ["<|im_start|>", "<|im_end|>"],
    },
    
    'Llama2': {
      'sysprompt': "[INST] <<SYS>>\n%s\n<</SYS>>",
      'user_template': "<s> [INST] {prompt} [/INST]",
      'bot_template': " {prompt}</s> ",
      'end_tag': "</s> ",
      'stopword': "[INST]",
    },

    'Zephyr': {
      'sysprompt_template': "<|system|>\n%s</s>\n",
      'user_template': "<|user|>\n{prompt}</s>",
      'bot_template': "<|assistant|>\n{prompt}",
      'end_tag': "</s>",
      'stopwords': ["<|user|>", "</s>"],
    },
}
