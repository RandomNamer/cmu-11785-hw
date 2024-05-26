class Module:
    
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError