class BootstrappedLeaveClusterOut:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def split(self, X, y=None, groups=None):
        raise RuntimeError("BootstrappedLeaveClusterOut is not available in serving mode.")

    def get_n_splits(self, X=None, y=None, groups=None):
        return 0
