import os
import time
import traceback

import telepot
from nltk.tokenize import WordPunctTokenizer

tokenizer_W = WordPunctTokenizer()


def tokenize(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())


import _thread


class PersonalTelBot(telepot.Bot):
    def __init__(self, api_token, password, whitelist):
        """

        :param api_token: API TOKEN for your telegram bot you can get from @BotFather  , you can register it there
        :param password:  Your password on your Linux/ MacOs system, it will be used for sudo commands or similar
        :param whitelist: path to whitelisted chat_id telegram accounts, you dont need to create, it will be created
                          on first run and then will be appended on every login
        """
        super(PersonalTelBot, self).__init__(api_token)
        self.password = password
        self.whitelist = whitelist


class BashCommandHandler(PersonalTelBot):

    def __init__(self, api_token, password, whitelist):
        if not os.path.isfile(whitelist):
            open(whitelist, 'w').write('')

        super(BashCommandHandler, self).__init__(api_token, password, whitelist)

    def process_command(self, translator, s, chat_id, ingroup):
        try:
            trans = translator.translate([s])
            print(trans)
        except Exception as e:
            trans = 'Error translating\nDetails\n' + e.args

        self.sendMessage(chat_id, '\n'.join(trans))


# bash_handler = BashCommandHandler(
#     api_token='HERE API TOKEN from @botfather @BotFather',
#     password='Your device password',
#     whitelist='path to whitelisted chat_id telegram accounts, you dont need to create, it will be created '
#               'on first run and then will be appended on every login'
# )

bash_handler = BashCommandHandler(
    api_token='1255989296:AAG97sKMA1TxjM7g7E5SRz0aENHWymr6hGc',
    password='',
    whitelist='./whitelist'
)
last_date_path = './last_date'

from translator import Translator


def timed_update(translator, command_text, chat_id, ingroup):
    if command_text.find('TranslateEzBot') == -1 and ingroup != False:
        return
    command_text = command_text.replace('@TranslateEzBot', '')
    bash_handler.process_command(translator, command_text, chat_id, ingroup)


if __name__ == '__main__':
    model_path = 'transformer.pth'
    en_field_path = 'EN_TEXT.field'
    ru_field_path = 'RU_TEXT.field'
    translator = Translator(model_path, en_field_path, ru_field_path)

    wait_time = 10  # this variable usually is 2 seconds but grows to 2 minutes if last time active is more
    # than 1 hour ago

    try:
        last_date = int(open(last_date_path).read())
    except:
        last_date = 0
        open(last_date_path, 'w').write('0')

    while True:
        # print(TelegramBot.getMe())
        try:
            bash_handler.deleteWebhook()
            messages = bash_handler.getUpdates(offset=last_date + 1)
            print('last date id is', last_date, len(messages))
            if len(messages) > 0:
                wait_time = 2
            else:
                wait_time = min(wait_time + 0.5, 120)
            for message_ in messages:
                if 'message' not in message_:
                    continue
                message = message_['message']
                ingroup = True
                try:
                    chat_id = message['chat']['id']
                    z = message['chat']['title']
                except Exception as e:
                    chat_id = message['from']['id']
                    ingroup = False

                print(message_)
                print(ingroup)

                if 'text' in message:
                    try:
                        print(message)
                        _thread.start_new_thread(timed_update, (translator, message['text'], chat_id, ingroup))
                    except Exception as e:
                        print(e.args)
                        continue
                else:
                    bash_handler.sendMessage(chat_id, 'Command error check logs on serverside (your device)')
            last_date = last_date if len(messages) == 0 else messages[-1]['update_id']
            open(last_date_path, 'w').write(str(last_date))
        except Exception as e:
            print(traceback.format_exc())

        time.sleep(wait_time)
