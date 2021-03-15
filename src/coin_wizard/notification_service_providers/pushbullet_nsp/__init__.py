#!/usr/bin/python3

from pushbullet import Pushbullet

class NotificationServiceProvider(object):
    notification_service_provider_settings_fields = ['api_key']
    def __init__(self, nsp_settings):
        self.lines = []
        self.api_key = nsp_settings['api_key']
        self.pb = Pushbullet(self.api_key)
        # print(self.api_key)

    def pushImmediately(self, title, context):
        self.pb.push_note(title, context)

    def push(self, title):
        context = ''.join(self.lines)
        self.lines = []
        self.pb.push_note(title, context)

    def addLine(self, string_line):
        self.lines.append('  ' + string_line+'\n')
