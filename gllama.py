from datetime import datetime
import gradio as gr
import json
import os
import requests
import sys
import yaml

from pypdf import PdfReader

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

def add_text(message, history):
  return '', history + [[message, '']]

def add_file(history, file):
  history = history + [((file.name,), None)]
  return history

def fetch_content(prompt, opt_params=None):
  payload = params.copy()
  if opt_params is not None:
    payload.update(opt_params)
  payload['prompt'] = prompt
  data = requests.request('POST', completion_url, data=json.dumps(payload),
                          stream=payload['stream'], headers=headers)
  for line in data.iter_lines():
    if line:
      decoded_line = line.decode('utf-8')
      if decoded_line.startswith('data:'):
        decoded_line = decoded_line[6:]
      d = json.loads(decoded_line)
      yield d['content']

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
  # messages += "\n".join(["\n".join([user_template.format(item[0]), bot_template.format(item[1])]) for item in history])
  for item in history:
    if type(item[0]) is tuple:
      print('Processing a file:', item[0][0])
      text = read_file(item[0][0])
      print('Processing done.')
      if text is not None:
        log_tokens(text)
        text = 'CONTEXT: """' + text + '"""\n'
        messages += add_linebreak(user_template.format(text))
    else:
      messages += add_linebreak(user_template.format(item[0]))
      messages += add_linebreak(bot_template.format(item[1]))
  messages = messages.rstrip(end_tag)
  messages = messages.rstrip()
  messages = add_linebreak(messages)

  history[-1][1] = ''
  for line in fetch_content(messages):
    history[-1][1] += line
    yield history

  save_history(history)
  return history

def save_history(history):
  print('save_history:', history[-1])
  filename = './history/' + datetime.now().strftime('%Y-%m-%d') + '.json'
  with open(filename, 'a') as f:
    f.write('\n' + json.dumps(history[-1]))

def update_params(temp, top_p, top_k, predictions, repeat_penalty, min_p):
  params['temperature'] = temp
  params['top_p'] = top_p
  params['top_k'] = top_k
  params['n_predict'] = predictions
  params['repeat_penalty'] = repeat_penalty
  params['min_p'] = min_p

def grammar_check(prompt):
  ss = f'Check the following line, which is enclosed by a double quotes for grammar and positional mistakes, and provide a revised line — “{prompt}".'
  params2 = {'stream': False, 'temperature': 0.1, 'n_keep': 0, 'cache_prompt': False}
  revised_prompt = ''.join(fetch_content(ss, params2))
  return prompt + revised_prompt

def read_file(filename):
  if filename.endswith('.txt'):
    with open(filename) as f:
      return ''.join(f.readlines())
  elif filename.endswith('.pdf'):
    reader = PdfReader(filename)
    text = ''
    for page in reader.pages:
      text += page.extract_text()
    return text
  return None

def add_linebreak(s):
  return s + '\n' if not s.endswith('\n') else s

def log_tokens(text):
  tokens = tokenize(text)
  if 'tokens' in tokens:
    print('log_tokens:', len(tokens['tokens']))
  else:
    print('log_tokens: unknown')


def main():
  CSS ="""
  #chatbot { min-height: 500px; }
  #chatbot .message { padding: 24px; }
  .compact-row { gap: 10px; padding: 0; }
  .custom-button { min-width: 40px; max-width: 40px; border-radius: 5px; }
  """

  with gr.Blocks(css=CSS, title='LocalLlama',
                 theme=gr.themes.Monochrome(text_size="sm")) as demo:
    with gr.Row():
      chatbot = gr.Chatbot(elem_id="chatbot",
                           layout='panel',
                           show_copy_button=True,
                           render_markdown=True,
                           bubble_full_width=False,
                           height=500,
                           container=False)
    with gr.Row(elem_classes='compact-row'):
      msg = gr.Textbox(autofocus=True,
                       lines=2,
                       show_label=False,
                       container=False)
      upload_btn = gr.UploadButton(label="📁", variant="secondary", size="sm",
                                   interactive=True,
                                   elem_classes="custom-button")
      submit_btn = gr.Button(value="▶", variant="primary", size="sm",
                             elem_classes="custom-button")
    with gr.Row():
      with gr.Column(scale=2):
        pass
      with gr.Column(scale=1):
        with gr.Row():
          with gr.Column(min_width=50, scale=1):
            check_btn = gr.Button(value="Check", size="sm")
          with gr.Column(min_width=50, scale=1):
            clear_btn = gr.Button(value="Clear", size="sm")
          with gr.Column(min_width=50, scale=1):
            stop_btn = gr.Button(value="Stop", variant="stop", size="sm")

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

    add_text_args = dict(
        fn=add_text, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    )
    update_params_args = dict(
        fn=update_params, inputs=[temp, top_p, top_k, predictions, penalty, min_p], queue=False
    )
    predict_args = dict(
        fn=predict, inputs=[chatbot, format_dropdown, system_prompt], outputs=chatbot
    )

    submit_event1 = msg.submit(**add_text_args).then(**update_params_args)\
        .then(**predict_args)
    submit_event2 = submit_btn.click(**add_text_args)\
        .then(**update_params_args).then(**predict_args)

    stop_btn.click(fn=None, inputs=None, outputs=None,
                   cancels=[submit_event1, submit_event2], queue=False)
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    prompts.select(lambda x: x, inputs=[prompts], outputs=system_prompt)

    check_btn.click(fn=grammar_check, inputs=[msg], outputs=[msg])
    upload_btn.upload(add_file, inputs=[chatbot, upload_btn], outputs=chatbot)

  demo.queue()
  demo.launch()

if __name__ == '__main__':
  sys.exit(main())
