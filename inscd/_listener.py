class _Listener:
    def __init__(self):
        """
        Description:
        A singleton decorate type to collect valid data during training. Default it works as print(). If you want to
        listen to the valid metrics during training, you can pass your listener function via listener.update(print) or
        listener.update(wandb.log). If you want to stop listening, you can use listener.silence().
        """
        self.__collector = print

    def update(self, collector):
        self.__collector = collector

    def reset(self):
        self.__collector = print

    def silence(self):
        self.__collector = None

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if self.__collector is not None:
                self.__collector(result)
            return result
        return wrapper