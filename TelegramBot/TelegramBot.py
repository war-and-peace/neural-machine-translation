import requests

from googletrans import Translator
from config import config


def get_name(message: dict):
    res = ''
    if 'first_name' in message.keys():
        res = message['first_name']
    if 'last_name' in message.keys():
        res = res + ' ' + message['last_name']
    return res


class TelegramBot:

    def __init__(self):
        """"
        Initializes an instance of the TelegramBot class.
        Attributes:
            chat_id:str: Chat ID of Telegram chat, used to identify which conversation outgoing messages should be send to.
            text:str: Text of Telegram chat
            first_name:str: First name of the user who sent the message
            last_name:str: Last name of the user who sent the message
        """

        self.chat_id = None
        self.text = None
        self.first_name = None
        self.last_name = None
        self.translator = Translator()

    def parse_webhook_data(self, data):
        """
        Parses Telegram JSON request from webhook and sets fields for conditional actions
        Args:
            data:str: JSON string of data
        """

        message = data['message']

        self.chat_id = message['chat']['id']
        try:
            self.incoming_message_text = message['text'].lower()
        except:
            return
        self.first_name = message['from']['first_name']
        self.last_name = message['from']['last_name']
        if 'forward_sender_name' in message.keys() or 'forward_from' in message.keys():
            try:
                self.forwarded_from = message['forward_from']
                print(self.forwarded_from, 'sip')
                self.forwarded_name = get_name(self.forwarded_from)
                print('done')
            except:
                self.forwarded_from = {
                    'forward_sender_name': message['forward_sender_name']
                }
                self.forwarded_name = message['forward_sender_name']
            self.forwarded_text = message['text']
        else:
            self.forwarded_from = None

    def action(self):
        """
        Conditional actions based on set webhook data.
        Returns:
            bool: True if the action was completed successfully else false
        """

        success = None

        if self.incoming_message_text == '/start':
            self.outgoing_message_text = "Hello {} {}!".format(self.first_name, self.last_name) + \
                '\n Send me only text ... if you send photos i will be fucked :)'
            success = self.send_message()
            return success
        print(self.forwarded_from)
        if self.forwarded_from:
            if not self.forwarded_text:
                self.outgoing_message_text = self.first_name + ' ' + self.last_name + ':\n' + self.translator.translate(
                    self.incoming_message_text).text
            else:
                self.outgoing_message_text = self.forwarded_name + ':\n' + self.translator.translate(
                    self.forwarded_text).text
            self.send_message()
        else:
            self.outgoing_message_text = self.translator.translate(self.incoming_message_text).text
            self.send_message()
        return success

    def send_message(self):
        """
        Sends message to Telegram servers.
        """
        conf = config()
        res = requests.get(conf.get_TELEGRAM_SEND_MESSAGE_URL().format(self.chat_id, self.outgoing_message_text))

        return True if res.status_code == 200 else False

    @staticmethod
    def init_webhook(url):
        """
        Initializes the webhook
        Args:
            url:str: Provides the telegram server with a endpoint for webhook data
        """

        requests.get(url)