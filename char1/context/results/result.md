
(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# python char1/context/context_experiment.py
项目根目录：/mnt/e/ai-agent-book
模型：qwen3.7-flash-2026-07-15
API 地址：https://dashscope.aliyuncs.com/compatible-mode/v1
实验组：['full', 'no_history', 'no_reasoning', 'no_tool_calls', 'no_tool_results']

[实验组] full
  第 1 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 2 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 1800000, 'from_currency': 'GBP', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'GBP', 'to_currency': 'USD', 'converted_amount': 2278481.01, 'reference_rate': '1.265822784810126582278481013', 'rate_type': 'fixed_example_not_live'}
  第 3 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 380000000, 'from_currency': 'JPY', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'JPY', 'to_currency': 'USD', 'converted_amount': 2541806.02, 'reference_rate': '0.006688963210702341137123745819', 'rate_type': 'fixed_example_not_live'}
  第 4 轮：工具调用 1 次，返回推理内容=True
    调用：calculate({'expression': '2500000+2282608.7+2278481.01+2541806.02'}) | 结果已隐藏=False | 返回={'value': '9602895.73', 'rounded_2': '9602895.73'}
  第 5 轮：工具调用 1 次，返回推理内容=True
    调用：calculate({'expression': '9602895.73/4'}) | 结果已隐藏=False | 返回={'value': '2400723.9325', 'rounded_2': '2400723.93'}
  第 6 轮：工具调用 0 次，返回推理内容=True

[实验组] no_history
  第 1 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 2 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 3 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 4 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 5 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 6 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 7 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 8 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 9 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 10 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}

[实验组] no_reasoning
  第 1 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 2 轮：工具调用 1 次，返回推理内容=False
    调用：convert_currency({'amount': 1800000, 'from_currency': 'GBP', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'GBP', 'to_currency': 'USD', 'converted_amount': 2278481.01, 'reference_rate': '1.265822784810126582278481013', 'rate_type': 'fixed_example_not_live'}
  第 3 轮：工具调用 1 次，返回推理内容=False
    调用：convert_currency({'amount': 380000000, 'from_currency': 'JPY', 'to_currency': 'USD'}) | 结果已隐藏=False | 返回={'from_currency': 'JPY', 'to_currency': 'USD', 'converted_amount': 2541806.02, 'reference_rate': '0.006688963210702341137123745819', 'rate_type': 'fixed_example_not_live'}
  第 4 轮：工具调用 1 次，返回推理内容=True
    调用：calculate({'expression': '2500000+2282608.7+2278481.01+2541806.02'}) | 结果已隐藏=False | 返回={'value': '9602895.73', 'rounded_2': '9602895.73'}
  第 5 轮：工具调用 1 次，返回推理内容=False
    调用：calculate({'expression': '9602895.73/4'}) | 结果已隐藏=False | 返回={'value': '2400723.9325', 'rounded_2': '2400723.93'}
  第 6 轮：工具调用 0 次，返回推理内容=False

[实验组] no_tool_calls
  第 1 轮：工具调用 0 次，返回推理内容=True

[实验组] no_tool_results
  第 1 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 2100000, 'from_currency': 'EUR', 'to_currency': 'USD'}) | 结果已隐藏=True | 返回={'from_currency': 'EUR', 'to_currency': 'USD', 'converted_amount': 2282608.7, 'reference_rate': '1.086956521739130434782608696', 'rate_type': 'fixed_example_not_live'}
  第 2 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 1800000, 'from_currency': 'GBP', 'to_currency': 'USD'}) | 结果已隐藏=True | 返回={'from_currency': 'GBP', 'to_currency': 'USD', 'converted_amount': 2278481.01, 'reference_rate': '1.265822784810126582278481013', 'rate_type': 'fixed_example_not_live'}
  第 3 轮：工具调用 1 次，返回推理内容=True
    调用：convert_currency({'amount': 380000000, 'from_currency': 'JPY', 'to_currency': 'USD'}) | 结果已隐藏=True | 返回={'from_currency': 'JPY', 'to_currency': 'USD', 'converted_amount': 2541806.02, 'reference_rate': '0.006688963210702341137123745819', 'rate_type': 'fixed_example_not_live'}
  第 4 轮：工具调用 0 次，返回推理内容=True

实验组            状态             调用   重复   数值正确  观测齐全  请求含历史推理
full             completed            5        0  True     True      5
no_history       iteration_limit     10        9  False    False     0
no_reasoning     completed            5        0  True     True      0
no_tool_calls    completed            0        0  False    False     0
no_tool_results  completed            3        0  False    False     3

结果已保存：/mnt/e/ai-agent-book/char1/context/results/context_ablation_20261010_145729.json
