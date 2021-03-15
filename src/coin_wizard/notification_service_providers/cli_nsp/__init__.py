#!/usr/bin/python3

class NotificationServiceProvider(object):
    notification_service_provider_settings_fields = []
    def __init__(self, nsp_settings):
        self.lines = []

    def pushImmediately(self, title, context):
        print('\n===== Notification(%s) ===== ' % (title))
        print(context)

    def push(self, title):
        context = ''.join(self.lines)
        self.lines = []
        print('\n===== Notification(%s) ===== ' % (title))
        print(context)

    def addLine(self, string_line):
        self.lines.append('  ' + string_line+'\n')
