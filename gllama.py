from functools import partial
import gradio as gr
import json
import os
import requests
import sys
import yaml

params = {
    'temperature': 0.1,
    'top_k': 40,
    'top_p': 0.90,
    # 'min_p': 0.05,
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

with open(os.path.join(os.path.dirname(__file__), 'prompts.yml')) as f:
	sysprompts = yaml.load(f, Loader=yaml.Loader)

with open(os.path.join(os.path.dirname(__file__), 'formats.yml')) as f:
	formats = yaml.load(f, Loader=yaml.Loader)

# chatml (OpenHermes-2.5, Mistral-OpenOrca, CausalLM-14B)
username = 'user'
assistant = 'assistant'
sysprompt_template = '<|im_start|>system\n%s<|im_end|>\n'
user_template = partial("<|im_start|>{name}\n{prompt}<|im_end|>".format, name=username)
bot_template = partial("<|im_start|>{name}\n{prompt}<|im_end|>".format, name=assistant)
end_tag = "<|im_end|>"
stopwords = ["<|im_start|>", "<|im_end|>"]

def tokenize(content):
    r = requests.post(tokenize_url, data=json.dumps({'content': content}))
    return r.json()

def show_prompt(message, history):
  return '', history + [[message, '']]

def update_system_prompt(prompt):
  tt = tokenize(sysprompt_template % prompt)
  tokens = len(tt['tokens'])
  params['n_keep'] = tokens

def update_template(prompt_format):
  params['stop'] = prompt_format['stopwords']

def predict(history, format_name):    
  print('predict:', format_name)
  messages = sysprompt_template % system_prompt
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
          predictions = gr.Slider(minimum=-1, maximum=2048, step=1, value=-1,
                                  interactive=True, label="Predictions")
        with gr.Column():
          pass
        with gr.Column():
          pass
      with gr.Row():
        with gr.Column():
          ff = list(formats.keys())
          format_dropdown = gr.Dropdown(choices=ff, value=ff[0],
                                      interactive=True, label='Formats')
        with gr.Column():
          pp = sysprompts
          prompts = gr.Dropdown(choices=pp, value=pp[0], interactive=True,
                                label='Prompts')
      with gr.Row():
        system_prompt = gr.Textbox(label='System prompt')
        
    #submit_click_event = submit.click(fn=show_prompt, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)\
        #    .then(fn=predict, inputs=chatbot, outputs=chatbot)
    submit_event = msg.submit(fn=show_prompt, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)\
        .then(fn=predict, inputs=[chatbot, format_dropdown], outputs=chatbot)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

    prompts.select(lambda x: x, inputs=[prompts], outputs=system_prompt)
    system_prompt.value = prompts.value

  demo.queue()
  demo.launch()

if __name__ == '__main__':
  sys.exit(main())
