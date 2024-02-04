import yaml

def messages(f):
  return f['sysprompt'].format('You are helpful assistant.') +\
         f['user_template'].format('What is capital of France?') + '\n' +\
         f['bot_template'].format('Paris')

with open('formats.yml', 'r') as f:
  ff = yaml.safe_load(f)
  for name in ff.keys():
    print('Format:', name)
    print(messages(ff[name]))
    print()
