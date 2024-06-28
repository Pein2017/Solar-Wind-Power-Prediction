import os


import torch  # noqa
from models import TimeMixer, SimpleMLP


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "TimeMixer": TimeMixer,
            "SimpleMLP": SimpleMLP,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_npu:
            import torch_npu  # noqa

            os.environ["NPU_VISIBLE_DEVICES"] = (
                str(self.args.npu) if not self.args.use_multi_npu else self.args.devices
            )
            device = torch.device("npu:{}".format(self.args.npu))
            if self.args.use_multi_npu:
                print("Use NPU: npu{}".format(self.args.device_ids))
            else:
                print("Use NPU: npu:{}".format(self.args.npu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device
