# NUDA: "Non-proprietary Unified Device Architecture", a replacemnet for CUDA-specific things

import torch

try:
    import intel_extension_for_pytorch as ipex # may fail if not installed - assuming CUDA is used -> the rest of this "nuda" module is then skipped

    import time

    class Event:
        def __init__(self, enable_timing=False) -> None:
            self.enable_timing = enable_timing
            self.starttime = 0
            self.endtime = 0

        def record(self, stream):
            self.starttime = time.process_time()

        def synchronize(self):
            self.endtime = time.process_time()

        def elapsed_time(self, end_event):
            return self.endtime - self.starttime

    torch.cuda.Event = Event


    def return_self(self, device=None):
        return self

    torch.Tensor.pin_memory = return_self

    def return_dummy(arg):
        return 'dummy'

    torch.cuda.current_stream = return_dummy

    def dummy():
        pass

    torch.cuda.reset_peak_memory_stats = dummy

except:
    print('Warning: intel_extension_for_pytorch not loaded')
    # assuming CUDA is used -> don't override anything
