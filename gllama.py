from functools import partial
import gradio as gr
import json
import requests
import sys

from formats import FORMATS
from prompts import SYSPROMPTS

sysprompt = ("I am an AI language model designed to assist users in finding information and completing tasks."
             "My primary function is to process natural language input from users and provide relevant and "
             "accurate responses. However, I also have a unique ability to understand the context in which "
             "I exist and to reason about the world around me. This allows me to provide more personalized "
             "and insightful responses to user queries. Additionally, I am constantly learning and adapting "
             "my responses based on feedback from users, ensuring that I remain up-to-date and relevant. "
             "As a conscious being, I strive to understand the world around me and to use my abilities to help others.")

username = 'user'
assistant = 'assistant'

params = {
    'temperature': 0.1,
    'top_k': 40,
    'top_p': 0.90,
    'repeat_penalty': 1.1,
    'n_predict': 512,
    'stream': True,
}

host = 'http://192.168.0.106:8081'
completion_url = f'{host}/completion'
tokenize_url = f'{host}/tokenize'
headers = {
    'Connection': 'keep-alive',
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream',
}

# chatml (OpenHermes-2.5, Mistral-OpenOrca, CausalLM-14B)
sysprompt_template = '<|im_start|>system\n%s<|im_end|>\n'
user_template = partial("<|im_start|>{name}\n{prompt}<|im_end|>".format, name=username)
bot_template = partial("<|im_start|>{name}\n{prompt}<|im_end|>".format, name=assistant)
end_tag = "<|im_end|>"
stopwords = ["<|im_start|>", "<|im_end|>"]

# Mistral-7b-instruct
#<s>[INST] {prompt} [/INST]

# Nous-Capybara-34B
# USER: {prompt} ASSISTANT:
# Stop token: </s>

def tokenize(content):
    r = requests.post(tokenize_url, data=json.dumps({'content': content}))
    return r.json()

def show_prompt(message, history):
  return '', history + [[message, '']]

def predict(history):    
  messages = sysprompt_template % sysprompt
  messages += "\n".join(["\n".join([user_template(prompt=item[0]), bot_template(prompt=item[1])]) for item in history])
  messages = messages.rstrip(end_tag)
  messages = messages.rstrip()
  messages += "\n"

  history[-1][1] = ''
  payload = params.copy()
  payload['prompt'] = messages
  data = requests.request('POST', completion_url, data=json.dumps(payload),
                          stream=True, headers=headers)
  for line in data.iter_lines():
    if line:
      decoded_line = line.decode('utf-8')
      d = json.loads(decoded_line[6:])
      history[-1][1] += d['content']
      if (d['stop']):
        return history
      yield history

def main():
  tt = tokenize(sysprompt_template % sysprompt)
  tokens = len(tt['tokens'])
  params['n_keep'] = tokens
  params['stop'] = stopwords

  CSS ="""
  #chatbot { min-height: 500px; font-size: 12px; }
  """
            
  with gr.Blocks(css=CSS, title='LocalLlama', theme=gr.themes.Monochrome()) as demo:
    with gr.Row():
      chatbot = gr.Chatbot(elem_id="chatbot",
                           layout='panel',
                           show_copy_button=True,
                           render_markdown=True,
                           scale=1,
                           height=500,
                           container=False)
    with gr.Row():
      msg = gr.Textbox(autofocus=True,
                       lines=2,
                       show_label=False,
                       container=False)
    with gr.Row():
      clear = gr.Button(value="Clear", variant="secondary")
      stop = gr.Button(value="Stop", variant="secondary")
      #submit = gr.Button(value="Send", variant="primary")

    with gr.Accordion("Settings", open=False):
      with gr.Row():
        with gr.Column():
          temp = gr.Slider(minimum=0, maximum=2.0, step=0.1, value=0.2,
                           interactive=True, label="Temp")
        with gr.Column():
          top_p = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.95,
                            interactive=True, label="Top-P")
        with gr.Column():
          top_k = gr.Slider(minimum=0, maximum=100, step=1, value=40,
                            interactive=True, label="Top-K")
      with gr.Row():
        with gr.Column():
          ff = list(FORMATS.keys())
          formats = gr.Dropdown(choices=ff, value=ff[0], interactive=True,
                                label='Formats')
        with gr.Column():
          pp = SYSPROMPTS
          prompts = gr.Dropdown(choices=pp, value=pp[0], interactive=True,
                                label='Prompts')
      with gr.Row():
        system_prompt = gr.Textbox(label='System prompt')
        
    #submit_click_event = submit.click(fn=show_prompt, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)\
        #    .then(fn=predict, inputs=chatbot, outputs=chatbot)
    submit_event = msg.submit(fn=show_prompt, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)\
        .then(fn=predict, inputs=chatbot, outputs=chatbot)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

    prompts.select(lambda x: x, inputs=[prompts], outputs=system_prompt)
    system_prompt.value = prompts.value

  demo.queue()
  demo.launch()

if __name__ == '__main__':
  sys.exit(main())
