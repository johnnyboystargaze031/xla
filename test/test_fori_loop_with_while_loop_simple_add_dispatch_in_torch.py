import os
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb
from torch_xla.experimental.fori_loop import while_loop


class WhileLoopTest(unittest.TestCase):

  def test_while_loop_tpu(self):

    def cond_fn(counter, x):
      return counter < xb.Op.scalar(x.builder(), 10, dtype=xb.Type.S32)

    def body_fn(counter, x):
      next_counter = counter + xb.Op.scalar(
          counter.builder(), 1, dtype=xb.Type.S32)
      x = x + xb.Op.scalar(x.builder(), 1, dtype=xb.Type.S32)
      return xb.Op.tuple((next_counter, x))

    device = xm.xla_device()
    x = torch.ones(1, dtype=torch.int, device=device)
    res = while_loop(cond_fn, body_fn, (x,))
    print("while_loop result: ", res)
    expected = torch.tensor(11, dtype=torch.int, device=device)
    print("expected result: ", expected)
    self.assertEqual(expected, res[0])


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
