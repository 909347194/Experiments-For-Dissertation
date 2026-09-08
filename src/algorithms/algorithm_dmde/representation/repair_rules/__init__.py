# -*- coding: utf-8 -*-
"""repair_rules — 反映射冲突消解策略簇

职责：
    本包收纳 inverse_mapper 反映射后各类离散冲突的修补策略，
    每个模块对应论文中的一条具体规则，独立可插拔、可组合，
    供 inverse_mapper.phi' 统一协调调用。

子模块清单：
    - nearest_match.py : 规则 3.4 —— 最近邻空间欧氏距离匹配
    - unique_filter.py : 规则 3.5 —— 独占分配与掩码(Inf)行/列剔除
    - invalid_mutator.py: 规则 3.6 —— 越界、重复及无效值随机变异修补

约定：
    各策略模块应提供统一签名的修补入口（如 repair(target, context)），
    以便协调器以策略簇方式按需调用。
"""
