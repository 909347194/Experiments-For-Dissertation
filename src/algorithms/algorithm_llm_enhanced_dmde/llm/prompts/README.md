# LLM 提示词管理

## 目标
将硬编码在模块中的提示词抽离出来，统一管理，支持：
1. 默认提示词（Python 模块定义）
2. 外部文件覆盖（实验配置目录）
3. 动态参数插值（f-string 风格）
4. 场景化提示词（N=M / N>M / N<M）

## 统一入口

`src/algorithms/algorithm_llm_enhanced_dmde/llm/prompts/__init__.py` 提供：

| 接口 | 用途 |
|---|---|
| `get_prompt(module_name, prompt_path, **kwargs)` | 统一入口 |
| `load_prompt_from_file(path, fallback)` | 文件加载工具 |
| `get_search_controller_prompt(model_type, cr_choices, f_choices, coupled, no_cr)` | 动态生成搜索控制器提示词 |
| `get_operator_selection_prompt(strategies)` | 动态生成算子选择提示词 |
| `PROMPT_REGISTRY` / `USER_PROMPT_REGISTRY` | 提示词注册表 |

提示词模板按**动作空间条件**参数化：默认（CR + F + GMR 解耦）、`coupled`（仅 CR，F/GMR 由公式 3-11/3-12 推导）、`no_cr`（CR 锁定，仅 F + GMR）。模型看不到自己无权控制的通道。

## 模块集成方式

### `llm/modules/search_controller.py`
```python
from ..prompts import get_search_controller_prompt

def __init__(self, llm_client, config):
    super().__init__(llm_client, config)
    prompt_path = self._config.get("system_prompt_path")
    cr_choices = self._config.get("cr_choices", CR_CHOICES)

    if prompt_path is None:
        self._system_prompt = get_search_controller_prompt(cr_choices=cr_choices)
    else:
        from pathlib import Path
        p = Path(prompt_path)
        self._system_prompt = (
            p.read_text() if p.exists()
            else get_search_controller_prompt(cr_choices=cr_choices)
        )
```

### `llm/modules/cr_control.py` / `llm/modules/operator_selection.py`（已废弃，统一走 search_controller）
```python
from ..prompts import get_prompt

def __init__(self, llm_client, config):
    super().__init__(llm_client, config)
    prompt_path = self._config.get("system_prompt_path")
    self._system_prompt = get_prompt("cr_control", prompt_path=prompt_path)
```

## 实验配置示例

### 目录结构
```
experiments/exp_llm_enhanced_dmde/exp_llm_dmde_01/
├── config/
│   ├── prompts/
│   │   └── search_controller.txt  # 自定义搜索控制器提示词
│   └── config.yaml
└── ...
```

### 配置文件示例 (config.yaml)
```yaml
llm_modules:
  search_controller:
    enabled: true
    system_prompt_path: "config/prompts/search_controller.txt"
    cr_choices: [0.1, 0.3, 0.5, 0.7, 0.9]
    interval: 5
```

## 使用方式

### 1. 使用默认提示词（无需配置）
```python
module = LLMSearchControllerModule(llm_client, {})
```

### 2. 使用外部文件覆盖
```python
module = LLMSearchControllerModule(llm_client, {
    "system_prompt_path": "/path/to/prompt.txt"
})
```

### 3. 动态参数配置
```python
module = LLMSearchControllerModule(llm_client, {
    "cr_choices": [0.2, 0.5, 0.8],  # 自定义 CR 候选值
})
```

## 优势

1. **统一管理**: 所有提示词集中在 `llm/prompts/__init__.py`
2. **代码解耦**: 模块代码不再包含大段提示词字符串
3. **实验灵活**: 修改提示词无需改动代码，支持快速迭代
4. **版本追踪**: 提示词文件可独立于代码进行版本控制
5. **场景定制**: 不同实验可使用不同的提示词文件
6. **动态参数**: 支持通过配置传递参数生成提示词
7. **向后兼容**: 保持原有的 `system_prompt_path` 配置方式

## 测试验证

```bash
# 测试提示词生成
python -c "from llm.prompts import get_search_controller_prompt; print(len(get_search_controller_prompt()))"

# 测试模块初始化
python -c "from llm.modules import create_module; m = create_module('search_controller', mock_llm, {}); print('OK')"

# 测试外部文件加载
python -c "from llm.prompts import get_prompt; p = get_prompt('search_controller', prompt_path='config/prompts/search_controller.txt'); print(len(p))"
```
