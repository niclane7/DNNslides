from torchsummary import summary

import re
import ast
from io import StringIO
import sys
from collections import OrderedDict

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def prod(nums):
    if len(nums) == 1:
        return int(nums[0])
    return int(nums[0]) * prod(nums[1:])

def num_mb(num_val):
    return num_val * 4 / (2 ** 20)

def num_params_and_output(model, input_shape):
    with Capturing() as output:
        summary(model, input_shape, device='cpu')
    layer_names = []
    n_params = []
    n_out = []
    start, end = [i for i, line in enumerate(output) if re.search('^=+$', line)]
    req_lines = output[start+1:end]
    re_layern = re.compile('[a-zA-Z]+\S*-\d?\d+?')
    re_out_sz = re.compile('\[.*\]')
    re_n_params = re.compile('(\d*,)*\d+$')
    for line in req_lines:
        layer_names.append(get_only_match(re_layern, line))
        n_p = int(get_only_match(re_n_params, line).replace(',', ''))
        n_params.append(n_p)
        output_sz = ast.literal_eval(get_only_match(re_out_sz, line))
        out_used = prod(output_sz[1:])
        n_out.append(out_used)
    return layer_names, n_params, n_out

def num_params(mode, input_shape):
    _, n_params, _= num_params_and_output(mode, input_shape)
    return n_params

def data_usage(model, input_shape, batch_size=1):
    layer_names, n_params, n_out = num_params_and_output(model, input_shape)
    return layer_names, [num_mb(p) for p in n_params], [num_mb(out) * batch_size for out in n_out]

def get_only_match(reg, line):
    return reg.search(line)[0]


if __name__ == '__main__':
    from torchvision.models import alexnet as net
    a = net()
    used = data_usage(a, (3,224,224), batch_size=128)
    print(used)
    print(len(used[0]))
    #summary(a, (3,224,224))
    #print(len(used))
    #print(len(used[0]), len(used[1]))

