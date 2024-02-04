import gradio as gr
import json
import os
import requests
import sys
import yaml

params = {
    'temperature': 0.2,
    'top_k': 40,
    'top_p': 0.90,
    'repeat_penalty': 1.1,
    'n_predict': -1,
    'min_p': 0.0, # disabled
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

def tokenize(content):
    r = requests.post(tokenize_url, data=json.dumps({'content': content}))
    return r.json()

def show_prompt(message, history):
  return '', history + [[message, '']]

def predict(history, format_name, sysprompt):
  if format_name in formats:
    llm_format = formats[format_name]
    sysprompt_templ = llm_format['sysprompt']
    user_template = llm_format['user_template']
    bot_template = llm_format['bot_template']
    end_tag = llm_format['end_tag']
    stopwords = llm_format['stopwords']
  else:
    raise Error("Unknown format:", format_name)

  params['stop'] = stopwords

  tokens = tokenize(sysprompt_templ.format(sysprompt))
  if 'tokens' in tokens:
    params['n_keep'] = len(tokens['tokens'])
  else:
    params['n_keep'] = -1

  print('params:', params)

  messages = sysprompt_templ.format(sysprompt)
  messages += "\n".join(["\n".join([user_template.format(item[0]), bot_template.format(item[1])]) for item in history])
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

def update_params(temp, top_p, top_k, predictions, repeat_penalty, min_p):
  params['temperature'] = temp
  params['top_p'] = top_p
  params['top_k'] = top_k
  params['n_predict'] = predictions
  params['repeat_penalty'] = repeat_penalty
  params['min_p'] = min_p

def main():
  CSS ="""
  #chatbot { min-height: 500px; }
  """

  with gr.Blocks(css=CSS, title='LocalLlama',
                 theme=gr.themes.Monochrome(text_size="sm")) as demo:
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
      with gr.Column(scale=3):
        pass
      with gr.Column(scale=1):
        with gr.Row():
          clear = gr.Button(value="Clear", variant="secondary", size="sm")
          stop = gr.Button(value="Stop", variant="stop", size="sm")
          # submit = gr.Button(value="Send", variant="primary", size="sm")

    with gr.Accordion("Settings", open=False):
      with gr.Row():
        with gr.Column():
          temp = gr.Slider(minimum=0, maximum=2.0, step=0.1,
                           value=params['temperature'],
                           interactive=True, label="Temp")
        with gr.Column():
          top_p = gr.Slider(minimum=0, maximum=1, step=0.01,
                            value=params['top_p'],
                            interactive=True, label="Top-P")
        with gr.Column():
          top_k = gr.Slider(minimum=0, maximum=100, step=1,
                            value=params['top_k'],
                            interactive=True, label="Top-K")
      with gr.Row():
        with gr.Column():
          predictions = gr.Slider(minimum=-1, maximum=2048, step=1,
                                  value=params['n_predict'],
                                  interactive=True, label="Predictions")
        with gr.Column():
          penalty = gr.Slider(minimum=1.0, maximum=1.25, step=0.01,
                              value=params['repeat_penalty'],
                              interactive=True, label="Repeat penalty")
        with gr.Column():
          min_p = gr.Slider(minimum=0, maximum=0.5, step=0.01,
                            value=params['min_p'],
                            interactive=True, label="Min-p")
      with gr.Row():
        with gr.Column():
          ff = list(formats.keys())
          format_dropdown = gr.Dropdown(choices=ff, value=ff[0],
                                        interactive=True, label='Formats')
        with gr.Column():
          prompts = gr.Dropdown(choices=sysprompts, value=sysprompts[0],
                                interactive=True, label='Prompts')
      with gr.Row():
        system_prompt = gr.Textbox(label='System prompt', value=sysprompts[0],
                                   interactive=True)

    #submit_click_event = submit.click(fn=show_prompt, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)\
        #    .then(fn=predict, inputs=chatbot, outputs=chatbot)
    submit_event = msg.submit(fn=show_prompt, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)\
        .then(fn=update_params, inputs=[temp, top_p, top_k, predictions, penalty, min_p], queue=False)\
        .then(fn=predict, inputs=[chatbot, format_dropdown, system_prompt], outputs=chatbot)

    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)
    prompts.select(lambda x: x, inputs=[prompts], outputs=system_prompt)

  demo.queue()
  demo.launch()

if __name__ == '__main__':
  sys.exit(main())
