# LLM 提示词管理重构总结

## 改造目标
将硬编码在模块中的提示词抽离出来，统一管理，支持：
1. 默认提示词（Python 模块定义）
2. 外部文件覆盖（实验配置目录）
3. 动态参数插值（f-string 风格）
4. 场景化提示词（N=M/N>M/N<M）

## 新增文件

### 1. `src/algorithms/algorithm_llm_enhanced_dmde/llm/prompts/__init__.py`
统一的提示词管理模块，提供：
- `get_prompt(module_name, prompt_path, **kwargs)`: 统一入口
- `load_prompt_from_file(path, fallback)`: 文件加载工具
- `get_search_controller_prompt(cr_choices)`: 动态生成带参数的提示词
- `get_operator_selection_prompt(strategies)`: 动态生成带参数的提示词
- `PROMPT_REGISTRY`: 提示词注册表
- `POPULATION_INIT_SYSTEM_PROMPT`: population_init 默认提示词
- `CR_CONTROL_SYSTEM_PROMPT`: cr_control 默认提示词

## 修改的文件

### 2. `llm/modules/population_init.py`
**改造前**: 硬编码 `_SYSTEM_PROMPT` 字符串，手动加载外部文件
**改造后**: 
```python
from ..prompts import get_prompt

def __init__(self, llm_client, config):
    super().__init__(llm_client, config)
    prompt_path = self._config.get("system_prompt_path")
    self._system_prompt = get_prompt("population_init", prompt_path=prompt_path)
```

### 3. `llm/modules/cr_control.py`
**改造前**: 硬编码 `_SYSTEM_PROMPT` 字符串
**改造后**:
```python
from ..prompts import get_prompt

def __init__(self, llm_client, config):
    super().__init__(llm_client, config)
    prompt_path = self._config.get("system_prompt_path")
    self._system_prompt = get_prompt("cr_control", prompt_path=prompt_path)
```

### 4. `llm/modules/search_controller.py`
**改造前**: 使用 `.format()` 动态生成提示词，手动加载外部文件
**改造后**:
```python
from ..prompts import get_search_controller_prompt

def __init__(self, llm_client, config):
    super().__init__(llm_client, config)
    prompt_path = self._config.get("system_prompt_path")
    cr_choices = self._config.get("cr_choices", CR_CHOICES)
    
    if prompt_path is None:
        self._system_prompt = get_search_controller_prompt(cr_choices=cr_choices)
    else:
        # 从外部文件加载
        from pathlib import Path
        p = Path(prompt_path)
        self._system_prompt = p.read_text() if p.exists() else get_search_controller_prompt(cr_choices=cr_choices)
```

### 5. `llm/modules/operator_selection.py`
**改造前**: 使用 `.format()` 动态生成提示词
**改造后**:
```python
from ..prompts import get_operator_selection_prompt

def __init__(self, llm_client, config):
    super().__init__(llm_client, config)
    prompt_path = self._config.get("system_prompt_path")
    strategies = self._config.get("strategies", AVAILABLE_STRATEGIES)
    
    if prompt_path is None:
        self._system_prompt = get_operator_selection_prompt(strategies=strategies)
    else:
        # 从外部文件加载
        from pathlib import Path
        p = Path(prompt_path)
        self._system_prompt = p.read_text() if p.exists() else get_operator_selection_prompt(strategies=strategies)
```

## 实验配置示例

### 实验目录结构
```
experiments/exp_llm_enhanced_dmde/exp_llm_dmde_04/
├── config/
│   ├── prompts/
│   │   ├── population_init.txt    # 自定义 population_init 提示词
│   │   ├── cr_control.txt         # 自定义 cr_control 提示词
│   │   └── search_controller.txt  # 自定义 search_controller 提示词
│   └── config.yaml                # 实验配置文件
└── ...
```

### 配置文件示例 (config.yaml)
```yaml
llm_modules:
  population_init:
    enabled: true
    system_prompt_path: "config/prompts/population_init.txt"
    llm_init_ratio: 0.2
  
  search_controller:
    enabled: true
    system_prompt_path: "config/prompts/search_controller.txt"
    cr_choices: [0.1, 0.3, 0.5, 0.7, 0.9]
    interval: 5
```

## 使用方式

### 1. 使用默认提示词（无需配置）
```python
module = LLMPopulationInitModule(llm_client, {})
```

### 2. 使用外部文件覆盖
```python
module = LLMPopulationInitModule(llm_client, {
    "system_prompt_path": "/path/to/prompt.txt"
})
```

### 3. 动态参数配置
```python
module = LLMSearchControllerModule(llm_client, {
    "cr_choices": [0.2, 0.5, 0.8],  # 自定义 CR 候选值
})
```

### 4. 场景化提示词（通过外部文件实现）
```bash
# N=M 场景
cp prompts/balanced.txt exp_01/config/prompts/population_init.txt

# N>M 场景
cp prompts/overloaded.txt exp_02/config/prompts/population_init.txt

# N<M 场景
cp prompts/srp.txt exp_03/config/prompts/population_init.txt
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
# 测试提示词加载
python -c "from llm.prompts import get_prompt; print(get_prompt('population_init')[:100])"

# 测试模块初始化
python -c "from llm.modules import create_module; m = create_module('population_init', mock_llm, {}); print('OK')"

# 测试外部文件加载
python -c "from llm.prompts import get_prompt; p = get_prompt('population_init', prompt_path='exp_04/config/prompts/population_init.txt'); print(len(p))"
```
