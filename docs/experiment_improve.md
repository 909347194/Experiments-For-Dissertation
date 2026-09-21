### Table 1

**Overall comparison**

| Algorithm | Best | Mean ± Std | Time |
| --------- | ---- | ---------- | ---- |

---

### Table 2

**Ablation**

| Configuration | Best | Mean ± Std | Median |
| ------------- | ---- | ---------- | ------ |

---

### Figure 1

**Convergence curves**

重点展示：

DMDE, +SearchCtrlDMDE,\\ +Search\\Ctrl---

### Figure 2

**LLM-guided CR trajectory**

CRt+fitnesstCR_t + fitness_t---

### Table 3

**Decoupled parameter analysis**

| Configuration | CR-F corr. | F mean ± std | GMR mode dist. |
| ------------- | ---------- | ------------ | -------------- |
| Vanilla DMDE (coupled) |  |  |  |
| Decoupled (CR+F+GMR)   |  |  |  |
| Coupled (CR only)      |  |  |  |
| No-CR (F+GMR)          |  |  |  |

---

### Table 4

**Scalability / three assignment models**

足够了。

# 最终论文我建议控制成这套实验矩阵

不要做得太大。

### Experiment 1：Overall comparison

<pre class="overflow-visible! px-0!" data-start="5625" data-end="5694"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>GA
DMDE
Random-DMDE
Rule-DMDE
DDQN-DMDE
LLM-MOEA
LLM-DMDE</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></pre>

这是你的主实验。

---

### Experiment 2：Ablation

<pre class="overflow-visible! px-0!" data-start="5738" data-end="5772"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>DMDE
+SearchCtrl
+SearchCtrl(coupled)
+SearchCtrl(no-CR)</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></pre>

核心指标：

- Best
- Mean ± Std
- Median
- convergence

---

### Experiment 3：Action-space ablation

<pre class="overflow-visible! px-0!" data-start="5865" data-end="5894"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>Decoupled (CR+F+GMR)
Coupled (CR only)
No-CR (F+GMR, CR=0.3 frozen)
Offline fixed CR=0.3</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></pre>

比较（三条件共用同一观测集、触发机制与冻结守卫，仅动作空间不同）：

- CR-F 相关系数（验证解耦）
- F / GMR 通道的实际使用率
- 与离线固定 CR=0.3 的差距（负面结果基线）

---

### Experiment 4：CR analysis

<pre class="overflow-visible! px-0!" data-start="5967" data-end="6026"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>Vanilla CR (formula 3-9)
Rule-based adaptive CR
LLM-guided CR (decoupled)
LLM-guided CR (coupled)
Offline fixed CR=0.3</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></pre>

比较：

- convergence curve
- convergence speed
- AUC
- CR trajectory

---

### Experiment 5：Scalability / generalization

至少：

N=M,N>M,N<MN=M,\\quad N>M,\\quad N<M最好再有一个 larger-scale
