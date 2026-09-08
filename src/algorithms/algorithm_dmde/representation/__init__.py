# -*- coding: utf-8 -*-
"""representation — 基因表征与空间映射机制

职责：
    本包实现 DMDE 的离散-连续混合表征体系，串联"离散三元组基因
    <i, j, k> → 连续空间差分进化搜索 → 离散解还原"三个环节，
    为上层算法提供统一的编码、映射与修复接口。

子模块清单：
    - encoder.py        : 统一三元组基因生成器（规则 3.1 - 3.3）
    - mapper.py         : 离散到连续的正向空间映射 phi（公式 3-5）
    - inverse_mapper.py : 连续到离散的反映射协调器 phi'（公式 3-7）
    - repair_rules/     : 反映射冲突消解策略簇（规则 3.4 - 3.6）

数据流约定：
    encoder（离散基因）--mapper.phi--> 连续个体
        --(DE 算子进化)--> 连续个体 --inverse_mapper.phi'--> 离散基因
    inverse_mapper 在还原过程中按需调用 repair_rules 消解各类冲突。
"""
