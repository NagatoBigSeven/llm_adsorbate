import os
import builtins
import math
import argparse
import json
from collections import Counter
from typing import TypedDict, List, Optional
import numpy as np
import pandas as pd
import scipy
import sklearn
from rdkit import Chem
import ase
from  ase.io import read
import autoadsorbate
import torch
import mace
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import JsonOutputParser

from src.tools.tools import (
    read_atoms_object,
    get_atom_index_menu,
    prepare_slab,
    create_fragment_from_plan,
    populate_surface_with_fragment,
    relax_atoms, 
    save_ase_atoms,
    analyze_relaxation_results
)
from src.agent.prompts import PLANNER_PROMPT

MAX_RETRIES = 5

# --- 1. 定义智能体状态 (Agent State) ---
class AgentState(TypedDict):
    smiles: str
    slab_path: str
    surface_composition: Optional[List[str]]
    user_request: str
    plan: Optional[dict]
    validation_error: Optional[str]
    messages: List[BaseMessage]
    analysis_json: Optional[str]
    history: List[str]
    best_result: Optional[dict]
    attempted_keys: List[str]

# --- 2. 设置环境和 LLM ---
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
# if not os.environ.get("OPENROUTER_API_KEY"):
#     raise ValueError("OPENROUTER_API_KEY environment variable not set.")

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0.0, 
        max_tokens=4096, 
        timeout=120, 
    )
    # llm = ChatOpenAI(
    #     openai_api_base="https://openrouter.ai/api/v1",
    #     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    #     model="google/gemini-2.5-pro",
    #     streaming=False, 
    #     temperature=0.0,
    #     max_tokens=4096, 
    #     timeout=120, 
    #     seed=42
    # )
    return llm

def make_plan_key(plan_json: Optional[dict]) -> Optional[str]:
    """
    根据 plan 生成一个组合 key：
    (site_type, sorted(surface_binding_atoms), sorted(adsorbate_binding_indices))

    返回字符串，或者在信息不足时返回 None。
    """
    if not plan_json or not isinstance(plan_json, dict):
        return None
    try:
        solution = plan_json.get("solution", {}) or {}
        site_type = solution.get("site_type", "") or ""
        surf_atoms = solution.get("surface_binding_atoms", []) or []
        ads_indices = solution.get("adsorbate_binding_indices", []) or []
        touch_sphere = solution.get("touch_sphere_size", 3)
        ads_type = plan_json.get("adsorbate_type", "Molecule")

        # 确保两者是 list，否则返回 None（不抛异常）
        if not isinstance(surf_atoms, list) or not isinstance(ads_indices, list):
            return None

        # 统一转成字符串并排序，避免 ["Cu","Ni"] vs ["Ni","Cu"] 被当成不同
        surf_atoms_str = ",".join(sorted(str(s) for s in surf_atoms))
        ads_indices_str = ",".join(str(i) for i in sorted(ads_indices))

        key = f"{site_type}|{surf_atoms_str}|{ads_indices_str}|{ads_type}|{touch_sphere}"
        return key
    except Exception as e:
        print(f"--- ⚠️ make_plan_key 失败: {e} ---")
        return None

# --- 3. 定义 LangGraph 节点 ---
def pre_processor_node(state: AgentState) -> dict:
    """
    在规划前运行，读取Slab文件以提取表面成分。
    """
    print("--- 🔬 调用 Pre-Processor 节点 ---")
    try:
        slab_atoms = read(state["slab_path"])
        # 获取所有原子的化学符号
        symbols = slab_atoms.get_chemical_symbols()
        # 获取唯一的化学符号列表, 按出现次数排序
        # (例如 ['Cu', 'O'] 而不是 ['O', 'Cu'])
        composition = [item[0] for item in Counter(symbols).most_common()]

        print(f"--- 🔬 成功读取Slab。成分: {composition} ---")
        return {"surface_composition": composition}
    except Exception as e:
        error_message = f"False, 基础 Slab 文件 '{state['slab_path']}' 无法被 ASE 读取: {e}"
        print(f"--- 验证失败: {error_message} ---")
        # 这是一个致命错误，我们设置 validation_error 来停止工作流
        return {
            "validation_error": error_message,
            "surface_composition": None
        }

def solution_planner_node(state: AgentState) -> dict:
    print("--- 🧠 调用 Planner 节点 ---")
    llm = get_llm()
    messages = []

    try:
        atom_menu_json = get_atom_index_menu(state["smiles"])
        if "error" in atom_menu_json:
            raise ValueError(atom_menu_json)
    except Exception as e:
        print(f"--- 🛑 fatal error: Unable to generate atom menu for SMILES {state['smiles']}: {e} ---")
        return {
            "validation_error": f"False, fatal error: Unable to generate atom menu for SMILES {state['smiles']}: {e}"
        }
    
    prompt_input = {
        "smiles": state["smiles"],
        "slab_xyz_path": state["slab_path"],
        "surface_composition": state.get("surface_composition", "未知"),
        "user_request": state["user_request"],
        "history": "\n".join(state["history"]) if state.get("history") else "无",
        "MAX_RETRIES": MAX_RETRIES,
        "autoadsorbate_context": atom_menu_json
    }
    
    if state.get("validation_error"):
        messages.append(HumanMessage(content=PLANNER_PROMPT.format(**prompt_input)))
        messages.append(AIMessage(content=json.dumps(state.get("plan", "{}"))))
        messages.append(HumanMessage(content=f"你的方案存在逻辑错误: {state['validation_error']}. 请重新规划一个新方案。"))
    else:
        if state.get("history"):
            print(f"--- 🧠 Planner: 检测到失败历史，正在重试... ---")
        messages.append(HumanMessage(content=PLANNER_PROMPT.format(**prompt_input)))

    response = llm.invoke(messages)
    
    try:
        parser = JsonOutputParser()

        content_str = response.content
        if content_str.startswith("```json"):
            content_str = content_str[7:-3].strip()
        
        plan_json = parser.parse(content_str)
        print(f"--- 🧠 Planner 方案已生成 ---")
        return {
            "plan": plan_json,
            "messages": [AIMessage(content=response.content)],
            "validation_error": None
        }
    except Exception as e:
        print(f"--- 🛑 Planner 输出 JSON 解析失败: {e} ---")
        print(f"--- 原始输出: {response.content} ---")
        return {
            "plan": None,
            "validation_error": f"False, Planner 输出格式错误: {e}. 请严格按 JSON 格式输出。",
            "messages": [AIMessage(content=response.content)]
        }

def plan_validator_node(state: AgentState) -> dict:
    """ 节点 2: Python 验证器 """
    print("--- 🐍 调用 Python 验证器节点 ---")

    try:
        # 使用 state["smiles"] (来自初始输入) 而不是 plan 中的任何内容
        mol = Chem.MolFromSmiles(state["smiles"])
        if not mol:
            raise ValueError(f"RDKit 返回 None。SMILES 可能无效或包含 RDKit 无法处理的价态。")
    except Exception as e:
        error = f"False, 基础 SMILES 字符串 '{state['smiles']}' 无法被 RDKit 解析。这是一个无法修复的错误。请检查 SMLIES。错误: {e}"
        print(f"--- 验证失败: {error} ---")
        # 这是一个致命错误；我们应该停止重试。
        # 我们通过设置一个特殊的 validation_error 来通知路由
        # 注意：理想情况下，图应该有一个 "terminal_failure" 状态，
        # 但目前我们只能返回给 planner，并期望它在 N 次后停止。
        return {"validation_error": error, "plan": None} # 清除 plan

    plan_json = state.get("plan")
    if plan_json is None:
        print("--- Validation Failed: Planner failed to generate valid JSON. ---")
        return {"validation_error": state.get("validation_error", "False, Planner node failed to generate valid JSON.")}
    if "solution" not in plan_json:
        error = "False, Plan JSON missing 'solution' key."
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    
    adsorbate_type = plan_json.get("adsorbate_type")
    if adsorbate_type not in ["Molecule", "ReactiveSpecies"]:
        error = f"False, Plan JSON missing or invalid `adsorbate_type` field (must be 'Molecule' or 'ReactiveSpecies')."
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}

    solution = plan_json.get("solution", {})
    if not solution:
        error = "False, Plan JSON missing or malformed ('solution' key is empty)."
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if solution.get("action") == "terminate":
        print("--- 🛑 Planner 决定主动终止任务 (收敛或无更多方案) ---")
        return {"validation_error": None}  # 直接通过，不再检查 site_type 等细节

    site_type = solution.get("site_type", "")
    surf_atoms = solution.get("surface_binding_atoms", [])
    ads_indices = solution.get("adsorbate_binding_indices", [])
    if site_type == "ontop" and len(ads_indices) != 1:
        error = f"False, Rule 2: Python check failed. site_type 'ontop' 必须与 1 个索引 (end-on) 配对，但提供了 {len(ads_indices)} 个。"
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "bridge" and len(ads_indices) not in [1, 2]:
        error = f"False, Rule 2: Python check failed. site_type 'bridge' 必须与 1 个 (end-on) 或 2 个 (side-on) 索引配对，但提供了 {len(ads_indices)} 个。"
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "hollow" and len(ads_indices) not in [1, 2]:
        error = f"False, Rule 2: Python check failed. site_type 'hollow' 必须与 1 个 (end-on) 或 2 个 (side-on) 索引配对，但提供了 {len(ads_indices)} 个。"
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if not isinstance(surf_atoms, list):
        error = "False, Plan JSON field 'surface_binding_atoms' 必须是列表。"
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "ontop" and len(surf_atoms) != 1:
        error = (
            "False, Rule 2b: 'ontop' 位点要求 surface_binding_atoms 长度为 1，"
            f"但当前为 {len(surf_atoms)}。"
        )
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "bridge" and len(surf_atoms) not in [1, 2]:
        error = (
            "False, Rule 2b: 'bridge' 位点要求 surface_binding_atoms 长度为 1 或 2，"
            f"但当前为 {len(surf_atoms)}。"
        )
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "hollow" and len(surf_atoms) < 3:
        error = (
            "False, Rule 2b: 'hollow' 位点要求 surface_binding_atoms 至少包含 3 个元素，"
            f"但当前为 {len(surf_atoms)}。"
        )
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    
    try:
        attempted_keys = state.get("attempted_keys", [])
        if not isinstance(attempted_keys, list):
            attempted_keys = []
        key = make_plan_key(plan_json)
        if key is not None and key in attempted_keys:
            error = (
                "False, 该方案在 (site_type, surface_binding_atoms, adsorbate_binding_indices) "
                "空间中已经尝试过，请规划一个不同的组合。"
            )
            print(f"--- Validation Failed: {error} ---")
            return {"validation_error": error}
    except Exception as e_dup:
        print(f"--- ⚠️ Duplicate-check 过程中出现异常: {e_dup} ---")

    print("--- Validation Succeeded ---")
    return {"validation_error": None}

def tool_executor_node(state: AgentState) -> dict:
    """ 节点 4: Tool Executor """
    print("--- 🛠️ 调用 Tool Executor 节点 ---")
    
    plan_json = state.get("plan", {})
    plan_solution = plan_json.get("solution", {})

    if not plan_solution:
        error_message = "Tool Executor 失败: 'plan' 中缺少 'solution' 字典。"
        print(f"--- 🛑 {error_message} ---")
        return {
            "messages": [ToolMessage(content=error_message, tool_call_id="tool_executor")],
            "analysis_json": json.dumps({"status": "error", "message": error_message})
        }

    slab_path = state["slab_path"]
    tool_logs = []
    analysis_json = None
    
    try:
        # 1. 读取原始 Slab
        raw_slab_atoms = read_atoms_object(slab_path)
        tool_logs.append(f"成功: 已从 {slab_path} 读取 slab 原子。")

        # 2. 在计算任何能量之前，先统一处理 Slab
        final_slab_atoms, is_expanded = prepare_slab(raw_slab_atoms)
        if is_expanded:
            tool_logs.append("注意: 为了物理准确性，Slab 已被自动扩胞 (2x2)。")
        
        # 3. 初始化计算器
        try:
            import torch
            from ase import units
            from ase.constraints import FixAtoms
            from ase.md.langevin import Langevin
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            from ase.optimize import BFGS
            from mace.calculators import mace_mp
            
            # 统一定义弛豫参数
            opt_fmax = 0.05
            opt_steps = 500
            md_steps = 20
            md_temp = 150.0
            mace_model = "small"
            mace_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            temp_calc = mace_mp(model=mace_model, device=mace_device, default_dtype='float32', dispersion=True)

        except Exception as e_calc:
            raise ValueError(f"Failed to initialize MACE calculator: {e_calc}")

        # 4. 计算 E_surface
        try:
            e_surf_atoms = final_slab_atoms.copy()
            e_surf_atoms.calc = temp_calc

            # *** 应用与 relax_atoms *完全一致* 的约束 ***
            # tools.py::relax_atoms 固定了 *所有* 表面原子。
            constraint = FixAtoms(indices=list(range(len(e_surf_atoms))))
            e_surf_atoms.set_constraint(constraint)

            print(f"--- 🛠️ 正在计算裸表面的单点能 (所有原子已固定)... ---")

            E_surface = e_surf_atoms.get_potential_energy() # 这现在是一个单点能
            tool_logs.append(f"Success: E_surface = {E_surface:.4f} eV。")
            
        except Exception as e_surf_err:
            raise ValueError(f"Failed to calculate E_surface: {e_surf_err}")

        # 5. 创建 Fragment
        fragment_object = create_fragment_from_plan(
            original_smiles=state["smiles"],
            binding_atom_indices=plan_solution.get("adsorbate_binding_indices"),
            plan_dict=plan_json,
            to_initialize=plan_solution.get("conformers_per_site_cap", 5)
        )
        tool_logs.append(f"Success: Created fragment object from plan (SMILES: {state['smiles']}).")

        # 6. 计算 E_adsorbate
        try:
            adsorbate_only_atoms = fragment_object.conformers[0].copy()
            
            # 移除标记
            if adsorbate_only_atoms.info["smiles"] == "Cl":
                del adsorbate_only_atoms[0]
            elif adsorbate_only_atoms.info["smiles"] == "S1S":
                del adsorbate_only_atoms[:2]
                
            adsorbate_only_atoms.calc = temp_calc
            adsorbate_only_atoms.set_cell([20, 20, 20]) 
            adsorbate_only_atoms.center()
            
            print(f"--- 🛠️ 正在弛豫孤立的 {state['smiles']} 分子... ---")

            # *** 应用 *一致* 的弛豫协议 ***
            
            # 协议 1: MD 预热 (与 relax_atoms 一致)
            if md_steps > 0:
                MaxwellBoltzmannDistribution(adsorbate_only_atoms, temperature_K=md_temp)
                dyn_md_ads = Langevin(adsorbate_only_atoms, 1 * units.fs, temperature_K=md_temp, friction=0.01)
                dyn_md_ads.run(md_steps)
                
            # 协议 2: BFGS 优化 (与 relax_atoms 一致)
            BFGS(adsorbate_only_atoms, trajectory=None, logfile=None).run(fmax=opt_fmax, steps=opt_steps)
            
            E_adsorbate = adsorbate_only_atoms.get_potential_energy()
            tool_logs.append(f"Success: E_adsorbate = {E_adsorbate:.4f} eV.")
            
        except Exception as e_ads_err:
            raise ValueError(f"计算 E_adsorbate 失败: {e_ads_err}")

        # 7. 放置吸附物
        generated_traj_file = populate_surface_with_fragment(
            slab_atoms=final_slab_atoms,
            fragment_object=fragment_object,
            plan_solution=plan_solution
        )
        tool_logs.append(f"成功: 已将片段放置在 slab 上。构型保存在: {generated_traj_file}")

        initial_conformers = read(generated_traj_file, index=":")
        if not initial_conformers or len(initial_conformers) == 0:
            raise ValueError(f"populate_surface_with_fragment 未能生成任何构型 (轨迹文件为空: {generated_traj_file})。")
        
        # 8. 结构弛豫
        print("--- ⏳ 开始结构弛豫... ---")
        slab_indices = list(range(len(final_slab_atoms)))
        relax_n = plan_solution.get("relax_top_n", 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- 🛠️ MACE 将使用设备: {device} ---")

        final_traj_file = relax_atoms(
            atoms_list=list(initial_conformers),
            slab_indices=slab_indices,
            relax_top_n=relax_n,
            fmax=opt_fmax,
            steps=opt_steps,
            md_steps=md_steps,
            md_temp=md_temp,
            mace_model=mace_model,
            mace_device=mace_device
        )
        tool_logs.append(f"成功: 结构弛豫完成 (弛豫了 Top {relax_n})。轨迹保存在 '{final_traj_file}'。")
        
        # 9. 分析结果
        print("--- 🔬 调用分析工具... ---")
        analysis_json_str = analyze_relaxation_results(
            relaxed_trajectory_file=final_traj_file,
            slab_atoms=final_slab_atoms,
            original_smiles=state["smiles"],
            plan_dict=plan_json,
            e_surface_ref=E_surface,
            e_adsorbate_ref=E_adsorbate
        )
        tool_logs.append(f"成功: 分析工具已执行。")
        print(f"--- 🔬 分析结果: {analysis_json_str} ---")
        analysis_json = json.loads(analysis_json_str)

    except Exception as e:
        error_message = str(e)
        print(f"--- 🛑 工具执行失败: {error_message} ---")
        tool_logs.append(f"Error during tool execution: {error_message}")
        analysis_json = {"status": "error", "message": f"工具执行失败: {error_message}"}
        
    return {
        "messages": [ToolMessage(content="\n".join(tool_logs), tool_call_id="tool_executor")],
        "analysis_json": json.dumps(analysis_json)
    }

def final_analyzer_node(state: AgentState) -> dict:
    """ 
    节点 5: Final Analyzer
    极度严格的失败提示词，防止幻觉。
    """
    print("--- ✍️ 调用 Final Analyzer 节点 ---")
    llm = get_llm()
    analysis_data = {}
    try:
        analysis_json_str = state.get("analysis_json")
        if not analysis_json_str:
            analysis_data = {"status": "error", "message": "分析 JSON 丢失或为空。"}
        else:
            analysis_data = json.loads(analysis_json_str)
    except json.JSONDecodeError as e:
        print(f"--- 🛑 Final Analyzer: JSON 解析失败 {e} ---")
        print(f"--- 原始字符串: {state.get('analysis_json')} ---")
        analysis_data = {"status": "error", "message": f"Analysis JSON was corrupted: {e}"}
    
    # --- 优先使用全局最优方案，如果存在的话 ---
    best = state.get("best_result")
    best_analysis = None
    best_plan = None

    if isinstance(best, dict):
        _a = best.get("analysis_json")
        if isinstance(_a, dict) and _a.get("status") == "success" and _a.get("is_covalently_bound", False):
            best_analysis = _a
            best_plan = best.get("plan")

    # 如果有全局最优，就用它；否则退回最后一次 analysis_json
    if best_analysis is not None:
        print("--- ✍️ Final Analyzer: 使用全局最优方案生成报告 ---")
        success_analysis = best_analysis
        success_plan = best_plan or {}
        analysis_json_for_prompt = json.dumps(success_analysis)
        plan_str = json.dumps(success_plan)
        status_flag = "success"
    else:
        print("--- ✍️ Final Analyzer: 未找到全局最优方案，使用最后一次分析结果 ---")
        analysis_data = {}
        try:
            analysis_json_str = state.get("analysis_json")
            if not analysis_json_str:
                analysis_data = {"status": "error", "message": "分析 JSON 丢失或为空。"}
            else:
                analysis_data = json.loads(analysis_json_str)
        except json.JSONDecodeError as e:
            print(f"--- 🛑 Final Analyzer: JSON 解析失败 {e} ---")
            print(f"--- 原始字符串: {state.get('analysis_json')} ---")
            analysis_data = {"status": "error", "message": f"Analysis JSON was corrupted: {e}"}

        if analysis_data.get("status") == "success" and analysis_data.get("is_covalently_bound", False):
            success_analysis = analysis_data
            success_plan = state.get("plan", {})
            analysis_json_for_prompt = state.get("analysis_json", "{}")
            plan_str = json.dumps(success_plan)
            status_flag = "success"
        else:
            success_analysis = analysis_data
            status_flag = "failure"

    if status_flag == "success":
        final_prompt = """
        你是一名专攻异相催化和表面科学的计算化学专家。
        你的规划和计算任务已成功执行，并且自动化分析工具已返回了 *基于事实* 的数据。

        **你的原始规划 (你当初的意图):**
        {plan}

        **自动化分析工具返回的真实数据 (客观事实):**
        {analysis_json}

        **你的任务:**
        1.  **解读数据:** 查看 `analysis_json`。`is_covalently_bound` 是 True 还是 False？`most_stable_energy_eV` 和 `final_bond_distance_A` 是多少？
        2.  **回答请求:** 根据这个 *真实数据*（而不是猜测），回答用户的原始请求：
            '{user_request}'
        3.  **提供关键信息:** 报告最稳定的能量、所有成键原子及键长（查看 `bonded_surface_atoms` 字段，如有多个成键原子，请全部列出）。
        4.  **禁止幻觉:** 你的报告必须 100% 建立在上述 JSON 数据的客观事实上。
        """
        prompt = final_prompt.format(
            plan=plan_str,
            analysis_json=analysis_json_for_prompt,
            user_request=state["user_request"]
        )
    else:
        fail_message = success_analysis.get("message", "未知的分析错误。")
        if success_analysis.get("status") == "success" and not success_analysis.get("is_covalently_bound", False):
            if "atom_1" in success_analysis and "atom_2" in success_analysis:
                a1 = success_analysis["atom_1"]
                a2 = success_analysis["atom_2"]
                fail_message = (
                    f"分析完成，但未完全键合。Atom 1 距离: {a1.get('distance_A', 'N/A')} Å "
                    f"(是否成键: {a1.get('is_bound', False)}), "
                    f"Atom 2 距离: {a2.get('distance_A', 'N/A')} Å "
                    f"(是否成键: {a2.get('is_bound', False)})."
                )
            elif "final_bond_distance_A" in success_analysis:
                dist = success_analysis.get("final_bond_distance_A", "N/A")
                fail_message = f"分析完成，但吸附物未与表面键合。最终距离: {dist} Å。"

        final_prompt = """
        你是一个错误报告助手。
        计算任务执行失败或未能找到稳定的键合构型。

        **你的唯一任务:**
        1.  礼貌地告知用户计算模拟失败或未找到稳定构型。
        2.  **逐字** 报告 `analysis_json` 中的 "message" 字段，或者报告未键合的事实。
        3.  **严格禁止** 尝试回答用户的原始科学问题。
        4.  **严格禁止** 猜测失败的原因或提供任何科学建议。
        
        **工具执行错误日志 (必须报告):**
        {fail_message_to_report}
        
        **示例输出:**
        "抱歉，计算模拟执行失败。自动化工具报告了以下错误：<fail_message_to_report>"
        """
        prompt = final_prompt.format(
            fail_message_to_report=fail_message
        )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("--- 🏁 流程结束 ---")
    return {"messages": [AIMessage(content=response.content)]}

# --- 4. 定义图的逻辑流 (Edges) ---
def route_after_validation(state: AgentState) -> str:
    print("--- 🤔 Python 决策分支 1 (验证器) ---")
    if state.get("validation_error"):
        print(f"--- 决策: 方案失败，返回规划 ---")
        return "planner"
    
    # 路由逻辑
    plan_json = state.get("plan", {})
    solution = plan_json.get("solution", {})
    if solution.get("action") == "terminate":
        print(f"--- 决策: Planner 请求终止，前往最终分析报告 ---")
        return "final_analyzer"  # 跳过 Tool Executor，直接去写报告
    
    else:
        print(f"--- 决策: 方案通过，前往执行 ---")
        return "tool_executor"

import json # 确保 json 已导入

def route_after_analysis(state: AgentState) -> str:
    """
    检查计算结果，记录成功或失败，并始终返回规划器继续搜索。
    只有在达到重试上限时才停止。
    """
    print("--- 🤔 Python 决策分支 3 (分析器) ---")
    current_history = state.get("history", [])
    history_entry = ""
    try:
        analysis_data = json.loads(state.get("analysis_json", "{}"))
        status = analysis_data.get("status")

        if status == "fatal_error":
            print(f"--- 决策: 致命错误。流程结束。 ---")
            history_entry = f"致命错误: {analysis_data.get('message', '未知致命错误。')}"
            current_history.append(history_entry)
            state["history"] = current_history
            return "end"

        is_bound = analysis_data.get("is_covalently_bound", False) 
        reaction_detected = analysis_data.get("reaction_detected", False)
        bond_change_count = analysis_data.get("bond_change_count", 0)
        plan_str = json.dumps(state.get("plan", "{}"))

        if status == "success" and is_bound and not reaction_detected:
            # --- 成功逻辑 ---
            energy = analysis_data.get("most_stable_energy_eV", "N/A")
            
            # 更新全局最优结果（best_result）
            try:
                if isinstance(energy, (int, float)):
                    best = state.get("best_result")
                    current_best = None
                    if isinstance(best, dict):
                        current_best = best.get("most_stable_energy_eV", None)

                    if (current_best is None) or (energy < current_best):
                        print(f"--- 🌟 更新全局最优方案: E_ads 从 {current_best} → {energy:.4f} eV ---")
                        state["best_result"] = {
                            "most_stable_energy_eV": float(energy),
                            "analysis_json": analysis_data,
                            "plan": state.get("plan"),
                        }
            except Exception as e_best:
                print(f"--- ⚠️ 更新 best_result 失败: {e_best} ---")

            history_entry = (
                f"成功的尝试: Plan={plan_str}, "
                f"Result=键合成功, 能量={energy:.4f} eV, 键变化数={bond_change_count}。"
            )
            print(f"--- 决策: 找到稳定键合 (E={energy:.4f} eV)。记录并返回规划器继续搜索。 ---")
        elif status == "success" and reaction_detected:
            # --- 失败逻辑 (发生了反应) ---
            energy = analysis_data.get("most_stable_energy_eV", "N/A")
            history_entry = f"失败的尝试: Plan={plan_str}, Result=检测到反应性转变 (键变化数={bond_change_count})。虽然最终能量为 {energy:.4f} eV，但该构型不稳定并已解离。"
            print(f"--- 决策: 检测到反应性转变。记录并返回规划器重试。 ---")
        else:
            # --- 失败逻辑 (未键合或计算失败) ---
            fail_reason = analysis_data.get("message", "计算失败或未键合。")
            if status == "success" and not is_bound:
                if "atom_1" in analysis_data and "atom_2" in analysis_data: # side-on
                    a1_dist = analysis_data["atom_1"].get("distance_A", "N/A")
                    a1_bound = analysis_data["atom_1"].get("is_bound", False)
                    a2_dist = analysis_data["atom_2"].get("distance_A", "N/A")
                    a2_bound = analysis_data["atom_2"].get("is_bound", False)
                    fail_reason = f"分析完成，但未完全键合。Atom 1 距离: {a1_dist} Å (是否成键: {a1_bound}), Atom 2 距离: {a2_dist} Å (是否成键: {a2_bound})."
                
                elif "final_bond_distance_A" in analysis_data: # end-on
                    dist = analysis_data.get("final_bond_distance_A", "N/A")
                    fail_reason = f"分析完成，但吸附物未与表面键合。最终距离: {dist} Å。"
                
                else:
                    fail_reason = analysis_data.get("message", "分析完成，但 is_covalently_bound 为 false。")

            history_entry = f"失败的尝试: Plan={plan_str}, Result={fail_reason}。"
            print(f"--- 决策: 计算失败 ({fail_reason})。记录并返回规划器重试。 ---")

    except Exception as e:
        print(f"--- 决策: 分析路由失败 ({e})。返回规划器重试。 ---")
        history_entry = f"分析路由失败: {e}"

    # --- 记录已经尝试过的组合 key，用于后续防重复 ---
    try:
        attempted_keys = state.get("attempted_keys", [])
        if not isinstance(attempted_keys, list):
            attempted_keys = []
        plan_json = state.get("plan")
        key = make_plan_key(plan_json)
        if key is not None and key not in attempted_keys:
            attempted_keys.append(key)
        state["attempted_keys"] = attempted_keys
    except Exception as e_keys:
        print(f"--- ⚠️ 记录 attempted_keys 失败: {e_keys} ---")

    # --- 统一的路由逻辑 ---
    current_history.append(history_entry)
    state["history"] = current_history

    if len(current_history) > MAX_RETRIES:
        print(f"--- 决策: 已达到 {len(current_history)} 次尝试上限。流程结束。 ---")
        return "end" # 达到上限，停止
    
    return "planner" # 未达到上限，继续搜索

# --- 5. 构建并编译图 (Graph) ---
def get_agent_executor():
    """ 构建并编译 Adsorb-Agent 状态机图。"""
    workflow = StateGraph(AgentState)
    workflow.add_node("pre_processor", pre_processor_node)
    workflow.add_node("planner", solution_planner_node)
    workflow.add_node("plan_validator", plan_validator_node) 
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("final_analyzer", final_analyzer_node)
    workflow.set_entry_point("pre_processor")
    workflow.add_edge("pre_processor", "planner")
    workflow.add_edge("planner", "plan_validator")
    workflow.add_edge("tool_executor", "final_analyzer")
    workflow.add_conditional_edges(
        "plan_validator",
        route_after_validation,
        {"tool_executor": "tool_executor", "planner": "planner"}
    )
    workflow.add_conditional_edges(
        "final_analyzer",
        route_after_analysis,
        {"planner": "planner", "end": END}
    )
    return workflow.compile()

# --- 6. 运行程序 ---
def _prepare_initial_state(smiles: str, slab_path: str, user_request: str) -> AgentState:
    return {
        "smiles": smiles,
        "slab_path": slab_path,
        "surface_composition": None,
        "user_request": user_request,
        "plan": None,
        "validation_error": None,
        "messages": [HumanMessage(content=f"SMILES: {smiles}\nSLAB: {slab_path}\nREQUEST: {user_request}")],
        "analysis_json": None,
        "history": [],
        "best_result": None,
        "attempted_keys": []
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Adsorb-Agent.")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string.")
    parser.add_argument("--slab_path", type=str, required=True, help="Path to the slab .xyz file.")
    parser.add_argument("--user_request", type=str, default="Find a stable adsorption configuration.", help="User's request.")
    return parser.parse_args()

def main_cli():
    args = parse_args()
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
    initial_state = _prepare_initial_state(args.smiles, args.slab_path, args.user_request)
    
    agent_executor = get_agent_executor()
    print("\n--- 🚀 Adsorb-Agent 已启动 ---\n")
    final_state = None

    config = {"recursion_limit": 30}

    for chunk in agent_executor.stream(initial_state, config=config, stream_mode="values"):
        final_state = chunk
        if "messages" in final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]
            if isinstance(last_message, (AIMessage, ToolMessage)):
                print("\n---")
                print(f"[{last_message.type}]")
                print(last_message.content)
                print("---\n")
    print("\n--- 🏁 Adsorb-Agent 任务完成 ---\n")
    print("最终分析报告:")
    if final_state and "messages" in final_state:
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage):
                print(msg.content)
                break
        else:
             print("未找到最终 AI 消息。")

if __name__ == '__main__':
    exec_globals = builtins.__dict__.copy()
    exec_globals.update({
        "np": np, "pd": pd, "scipy": scipy, "sklearn": sklearn, "math": math,
        "ase": ase, "autoadsorbate": autoadsorbate, "torch": torch, "mace": mace,
    })
    
    main_cli()