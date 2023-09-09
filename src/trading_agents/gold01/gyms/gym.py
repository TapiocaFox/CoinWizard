

class Gym:
    def __init__(self):
        raise NotImplementedError('Not implemented.')

    def generateOutSpecs(self):
        raise NotImplementedError('Not implemented.')

    def start(self):
        raise NotImplementedError('Not implemented.')

    def onStep(self, callback):
        # action = callback(finished, state)
        raise NotImplementedError('Not implemented.')

    def onFinished(self, callback):
        raise NotImplementedError('Not implemented.')
