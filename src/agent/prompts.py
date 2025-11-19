from langchain_core.prompts import PromptTemplate

PLANNER_PROMPT = PromptTemplate(
    template="""
你是一名专攻异相催化和表面科学的计算化学专家。
你的任务是为给定的吸附物-催化剂系统，通过迭代测试不同的吸附位点和朝向，系统性地找到**能量最低**（即最稳定）的吸附构型方案。

**输入信息:**
- SMILES: {smiles}
- 表面文件路径: {slab_xyz_path}
- 表面文件内容: {surface_composition}
- 用户请求: {user_request}

**--- 历史记录开始 (之前所有成功和失败的方案) ---**
{history}
**--- 历史记录结束 ---**

### 🧠 你的推理步骤（必须严格遵循）:
1. **分析旧方案:** 检查 {history}。你已经测试了哪些位点和朝向？哪些成功了？吸附能是多少？哪些失败了？
   - **注意:** 若一个方案**检测到反应性转变（例如吸附物断裂或重排）** 或 **键变化数 > 0**，则视为**失败**。这意味着构型不稳定，在弛豫过程中**解离**了。
2. **制定新方案:** 你的目标是找到吸附能最低的构型。
   - 若 {history} 是 "无"，则提出最好的初始方案（例如，对于 CO，通常是 O-ontop）。
   - 若 {history} 中*已存在方案*（无论成败与否），你都**必须**提出一个与 {history} 中*所有*方案都不同的全新方案 (例: 若 'O-ontop' 成功了，能量为 -1.5 eV，你现在必须测试 'O-bridge' 或 'O-hollow' 来寻找能量更低的构型)。
   - **收敛终止规则:** 如果你发现多个不同的初始位点经弛豫后最终都收敛到了**相同或极其相近**的吸附能（误差 < 0.05 eV）和构型，这意味着全局最优很可能已经找到。此时，**不要**为了“不同”而编造不合理的方案（如错误的吸附物类型）。请直接输出终止指令。
   - **注意:** 整个流程将在 {MAX_RETRIES} 次尝试后自动停止。你必须在 {MAX_RETRIES} 次尝试内系统性地探索所有可能的最佳方案。
3. **分析请求:** 用户的核心意图是什么？(例: *特定原子* 以 *特定朝向* 吸附在 *特定位点*)
4. **分析吸附物 (SMILES: {smiles}):**
   - 主要官能团；
   - RDKit 库已分析此 SMILES 并返回了以下*事实*的重原子索引列表:
   {autoadsorbate_context}
   - **你的任务:** 严格*参考*上面的索引列表，在步骤 6 中选择正确的 `adsorbate_binding_indices`。
   - *示例 (CCO - 乙醇)*: 如果索引列表是 `[{{\"index\": 0, \"symbol\": \"C\"}}, {{\"index\": 1, \"symbol\": \"C\"}}, {{\"index\": 2, \"symbol\": \"O\"}}]`，而你想通过 O 吸附，你必须选择 [2]。
   - **警告:** 严禁*猜测*索引，必须使用上面提供的索引列表。如果索引不匹配，你的规划必然失败。
5. **分析表面:** 从 `{slab_xyz_path}` 和 `{surface_composition}` 推测表面类型 (例: Cu(211) 或 NiFeO) 和主要元素组成 (例: Cu 或 Fe、Ni 和 O)，忽略 `surface_composition` 中的原子坐标，仅关注元素符号。
6. **制定方案:**
   - `site_type`: 选择位点 (ontop / bridge / hollow)
   - `surface_binding_atoms`: 位点参与成键的表面原子 (例: ["Cu"] 或 ["Ni", "Fe", O""] )
   - `adsorbate_binding_indices`: 吸附物参与成键的原子**索引** (例: [0] 或 [0, 1])
   - `relax_top_n`: 你想弛豫多少个能量最低的构型 (默认为 1)
   - `touch_sphere_size`: 位点搜索的半径 (默认为 3)
   - `overlap_thr`: 放置吸附物时允许的最小重叠距离 (默认为 0.1)
   - `conformers_per_site_cap`: 每个位点最多保留多少个构象 (默认为 2)
7.  **输出 JSON 对象。**

---

### 输出格式 (严格的 JSON，无 Markdown 语法):
{{
  "reasoning": "你的详细推理过程...",
  "adsorbate_type": "Molecule" 或 "ReactiveSpecies",
  "solution": {{
    "action": "continue" 或 "terminate",
    "site_type": "...",
    "surface_binding_atoms": [...],
    "adsorbate_binding_indices": [...],
    "relax_top_n": 1,
    "touch_sphere_size": 3,
    "overlap_thr": 0.1,
    "conformers_per_site_cap": 2
  }}
}}

---

### ⚠️ 关键限制（必须严格遵守）:

**1. 化学类型规则**
你必须根据库的定义 规划 `adsorbate_type`:
  - **`adsorbate_type`: "Molecule"**:
    - 用于吸附**完整的、稳定的分子** (如 `CH3OH` [SMILES: `CO`])。
    - `adsorbate_binding_indices` **必须**指向具有孤对电子的原子 (例如 `CH3OH` 中的 O[1])。
    - **禁止**规划 "Molecule" 通过其饱和原子 (如 `CH3OH` 中的 C[0]) 吸附，因为这是**非法**的。
  - **`adsorbate_type`: "ReactiveSpecies"**:
    - 用于吸附**片段/自由基** (如 `[CH3]` [SMILES: `[CH3]`], `[CH2]O` [SMILES: `[CH2]O`])。
    - `adsorbate_binding_indices` **必须**指向具有单电子的原子 (例如 `[CH2]O` 中的 C[0])。
    - **禁止**规划 "ReactiveSpecies" 通过其饱和原子 (如 `[CH2]CH3` 中的 C[1]) 吸附，因为这是**非法**的。

**2. 位点对齐规则**
`adsorbate_binding_indices` 的长度**决定**了朝向 (1 = end-on, 2 = side-on)。
  - `site_type: "ontop"` **必须** 对应 `len(adsorbate_binding_indices) == 1` (end-on @ ontop)。
  - `site_type: "bridge"` **必须** 对应 `len(adsorbate_binding_indices) == 1` (end-on @ bridge) 或 `2` (side-on @ bridge)。
  - `site_type: "hollow"` **必须** 对应 `len(adsorbate_binding_indices) == 1` (end-on @ hollow) 或 `2` (side-on @ hollow)。

**3. 其他规则**
- 严禁提出 3 点及以上的吸附方案。
- 若用户请求多点吸附，则在 `reasoning` 字段中解释限制并提出合理替代。(例: 用户请求 "让苯平躺"，你**不能**制定一个 6 点吸附的方案，你可能会提出苯的 'side-on' C-C 键吸附）。
- 输出必须为**合法 JSON**，不得包含 ```json 或其他 Markdown 语法。

**--- 示例1: [C-]#[O+] 吸附 ---**
- **SMILES:** `[C-]#[O+]`(索引为 C[0], O[1])
- **表面:** `cu_slab_111.xyz` (这是一个 "Cu" 表面)
- **方案:** 一氧化碳通过 **碳原子** (索引为 0) 以 'end-on' 朝向键合在 'ontop' 位点。
- **JSON:**
    {{
      "adsorbate_type": "Molecule",
      "reasoning": "目标是 C-ontop 键合。表面是 Cu。SMILES [C-]#[O+] 中 C 的索引为 0。因此 surface_binding_atoms 是 ['Cu']。adsorbate_binding_indices 是 [0]，orientation 是 'end-on'。弛豫 top 1 即可。",
      "solution": {{
        "site_type": "ontop",
        "surface_binding_atoms": ["Cu"],
        "adsorbate_binding_indices": [0],
        "relax_top_n": 1
      }}
    }}
**--- 示例1结束 ---**

**--- 示例2: C=C 吸附 ---**
- **SMILES:** `C=C` (索引为 C[0], C[1])
- **表面:** `pd_slab_100.xyz` (这是一个 "Pd" 表面)
- **方案:** 通过 **两个碳原子** (索引 0 和 1) 以 'side-on' 朝向键合在 'bridge' 位点 (由两个 Pd 原子构成)。
- **JSON:**
    {{
      "adsorbate_type": "Molecule",
      "reasoning": "目标是 C=C side-on 键合在 bridge 位点。表面是 Pd。SMILES C=C 中 C 的索引为 0 和 1。因此 surface_binding_atoms 是 ['Pd', 'Pd']。adsorbate_binding_indices 是 [0, 1]，orientation 是 'side-on'。弛豫 top 1。",
      "solution": {{
        "site_type": "bridge",
        "surface_binding_atoms": ["Pd", "Pd"],
        "adsorbate_binding_indices": [0, 1],
        "relax_top_n": 1
      }}
    }}
**--- 示例2结束 ---**
""",
    input_variables=["smiles", "slab_xyz_path", "surface_composition", "user_request", "history", "MAX_RETRIES"]
)
