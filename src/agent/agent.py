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
    analyze_surface_sites,
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
    best_dissociated_result: Optional[dict]
    attempted_keys: List[str]
    available_sites_description: Optional[str]

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

        # 统一转成字符串，保留顺序以区分异核双点吸附的方向 (如 Mo-Pd vs Pd-Mo)
        surf_atoms_str = ",".join(str(s) for s in surf_atoms)
        ads_indices_str = ",".join(str(i) for i in ads_indices)

        key = f"{site_type}|{surf_atoms_str}|{ads_indices_str}|{ads_type}|{touch_sphere}"
        return key
    except Exception as e:
        print(f"--- ⚠️ make_plan_key 失败: {e} ---")
        return None

# --- 3. 定义 LangGraph 节点 ---
def pre_processor_node(state: AgentState) -> dict:
    print("--- 🔬 调用 Pre-Processor 节点 ---")
    try:
        analysis = analyze_surface_sites(state["slab_path"])
        return {
            "surface_composition": analysis["surface_composition"],
            "available_sites_description": analysis["available_sites_description"]
        }
    except Exception as e:
        error_message = f"错误: 无法读取 Slab 文件 '{state['slab_path']}': {e}"
        print(f"--- 验证失败: {error_message} ---")
        return {
            "validation_error": error_message,
            "surface_composition": None,
            "available_sites_description": None
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
        "autoadsorbate_context": atom_menu_json,
        "available_sites_description": state.get("available_sites_description", "无"),
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

    new_best_molecular = state.get("best_result")
    new_best_dissociated = state.get("best_dissociated_result")
    
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

            # 检测单原子分子。单原子在真空中没有内部自由度，势能面平坦，导致 BFGS 算法因力变化为0而除以零崩溃。
            if len(adsorbate_only_atoms) > 1:
                # 协议 1: MD 预热 (与 relax_atoms 一致)
                if md_steps > 0:
                    MaxwellBoltzmannDistribution(adsorbate_only_atoms, temperature_K=md_temp)
                    dyn_md_ads = Langevin(adsorbate_only_atoms, 1 * units.fs, temperature_K=md_temp, friction=0.01)
                    dyn_md_ads.run(md_steps)
                    
                # 协议 2: BFGS 优化 (与 relax_atoms 一致)
                BFGS(adsorbate_only_atoms, trajectory=None, logfile=None).run(fmax=opt_fmax, steps=opt_steps)
            else:
                print(f"--- 🛠️ 检测到单原子吸附物 ({len(adsorbate_only_atoms)} atom)，跳过真空弛豫（物理上无需优化）。 ---")
            
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

        if analysis_json.get("status") == "success":
            e_new = analysis_json.get("most_stable_energy_eV")
            is_dissociated = analysis_json.get("is_dissociated")

            # 逻辑分支 A: 如果是完整的分子 (Molecular State)
            if not is_dissociated:
                e_old_mol = new_best_molecular.get("most_stable_energy_eV", float('inf')) if new_best_molecular else float('inf')
                if isinstance(e_new, (int, float)) and e_new < e_old_mol:
                    print(f"--- 🌟 发现新最优 [分子态]: {e_new:.4f} eV ---")
                    new_best_molecular = {
                        "most_stable_energy_eV": e_new,
                        "analysis_json": analysis_json,
                        "plan": state.get("plan"),
                        "result_type": "Perfect" if analysis_json.get("bond_change_count")==0 else "Isomerized"
                    }

            # 逻辑分支 B: 如果是解离态 (Dissociated State) - [新增]
            else:
                e_old_diss = new_best_dissociated.get("most_stable_energy_eV", float('inf')) if new_best_dissociated else float('inf')
                if isinstance(e_new, (int, float)) and e_new < e_old_diss:
                    print(f"--- ⚠️ 发现更稳定的 [解离态]: {e_new:.4f} eV (将作为热力学参考) ---")
                    new_best_dissociated = {
                        "most_stable_energy_eV": e_new,
                        "analysis_json": analysis_json,
                        "plan": state.get("plan"),
                        "result_type": "Dissociated"
                    }

    except Exception as e:
        error_message = str(e)
        print(f"--- 🛑 工具执行失败: {error_message} ---")
        tool_logs.append(f"Error during tool execution: {error_message}")
        analysis_json = {"status": "error", "message": f"工具执行失败: {error_message}"}
        
    return {
        "messages": [ToolMessage(content="\n".join(tool_logs), tool_call_id="tool_executor")],
        "analysis_json": json.dumps(analysis_json),
        "best_result": new_best_molecular,
        "best_dissociated_result": new_best_dissociated
    }

def final_analyzer_node(state: AgentState) -> dict:
    """ 
    节点 5: Final Analyzer
    功能：基于全局最优结果生成报告，并区分完美吸附与分子内重排。
    """
    print("--- ✍️ 调用 Final Analyzer 节点 ---")
    llm = get_llm()
    
    # 1. 提取数据源
    best_result = state.get("best_result")
    best_dissociated = state.get("best_dissociated_result")
    last_analysis_json_str = state.get("analysis_json", "{}")
    
    try:
        last_analysis = json.loads(last_analysis_json_str)
    except:
        last_analysis = {}

    # 2. 决策：汇报哪个数据？
    target_data = None
    plan_used = None
    source_type = "failure"
    result_label = "Unknown" # 用于提示 LLM 结果类型

    # 优先级 1: 历史最优
    if best_result and isinstance(best_result, dict):
        print(f"--- ✍️ Final Analyzer: 锁定全局最优方案 (E={best_result.get('most_stable_energy_eV')} eV) ---")
        target_data = best_result.get("analysis_json")
        plan_used = best_result.get("plan")
        # 如果 route_after_analysis 保存了 result_type，则读取它
        result_label = best_result.get("result_type", "Best History")
        source_type = "success"
    
    # 优先级 2: 最后一次尝试成功
    elif last_analysis.get("status") == "success" and last_analysis.get("is_covalently_bound"):
        print("--- ✍️ Final Analyzer: 无历史最优，使用最后一步的成功结果 ---")
        target_data = last_analysis
        plan_used = state.get("plan")
        result_label = "Last Attempt"
        source_type = "success"
    
    else:
        print("--- ✍️ Final Analyzer: 所有尝试均失败 ---")
        source_type = "failure"

    # 3. 构建 Prompt
    if source_type == "success":
        data_str = json.dumps(target_data, indent=2, ensure_ascii=False)
        plan_str = json.dumps(plan_used, indent=2, ensure_ascii=False)
        
        # [新增] 准备解离态对比数据
        diss_warning_context = ""
        if best_dissociated:
            e_mol = target_data.get("most_stable_energy_eV", 999)
            e_diss = best_dissociated.get("most_stable_energy_eV", 999)
            if e_diss < e_mol:
                delta_E = e_diss - e_mol
                diss_warning_context = (
                    f"\n*** 严重热力学警告数据 ***\n"
                    f"虽然用户要求寻找分子吸附，但系统在历史计算中发现了能量更低的解离态。\n"
                    f"- 分子态能量: {e_mol:.3f} eV\n"
                    f"- 解离态能量: {e_diss:.3f} eV (更稳定 {abs(delta_E):.3f} eV)\n"
                    f"这意味着报告的分子态在热力学上是亚稳的，容易自发解离。"
                )

        final_prompt = f"""
        你是一名严谨的计算化学家。你的任务是根据提供的【客观事实数据】撰写最终实验报告。

        !!! 严重警告 !!!
        你必须 **严格忠实** 于以下 JSON 数据。
        - **严禁编造** 任何数字。
        - **严禁编造** 吸附位点名称（以 `actual_site_type` 为准）。
        
        **用户请求:** {state['user_request']}

        **最佳吸附构型数据:**
        ```json
        {data_str}
        ```

        {diss_warning_context}

        **初始规划:**
        ```json
        {plan_str}
        ```

        **撰写要求:**
        1.  **结论:** 直接回答用户请求。
        2.  **数据支撑:** 列出 `most_stable_energy_eV` (保留3位小数) 和 `final_bond_distance_A`。
        3.  **几何细节:** 描述 `bonded_surface_atoms` 中的原子和距离。
        4.  **位点纠正:** 如果 `actual_site_type` 与 `planned_site_type` 不符，明确指出发生了“位点滑移”。
        5.  **化学状态判定 (重要):** 请检查 JSON 中的 `bond_change_count` 和 `reaction_detected` 字段：
            - **完美吸附**: 如果 `bond_change_count == 0`，请报告为“分子以完整构型稳定吸附”。
            - **异构化/重排**: 如果 `bond_change_count > 0` 但 `is_dissociated == False`，请特别强调：“吸附物在表面发生了 **分子内重排/异构化**（键变化数: {{bond_change_count}}），形成了更稳定的新构型。”这应被视为一个重要的化学发现。
            - **解离**: 如果 `is_dissociated == True`，请报告为“吸附物发生了解离”。
        6. **科学完整性与热力学警告 (至关重要):**
            - 如果提供了【严重热力学警告数据】，你必须在报告的“结论”或“讨论”部分以醒目的方式指出：
              “尽管找到了稳定的分子吸附态，但计算显示该分子在该表面发生解离在热力学上更有利（能量低 X eV）。因此，报告的构型可能仅在动力学上稳定（亚稳态）。”
            - 严禁隐瞒这一事实，这关乎科学诚信。
        """
    else:
        fail_reason = last_analysis.get("message", "未找到稳定构型。")
        final_prompt = f"""
        你是一个错误报告助手。
        任务：礼貌地告知用户，在经过多次尝试后，未能找到符合要求的稳定吸附构型。
        错误日志："{fail_reason}"
        请建议用户检查 SMILES 或更换表面模型。严禁捏造结果。
        """

    # 4. 调用 LLM
    response = llm.invoke([HumanMessage(content=final_prompt)])
    
    print("--- 🏁 最终报告生成完毕 ---")
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

def route_after_analysis(state: AgentState) -> str:
    """
    简化的路由器：生成富含信息的历史记录，并决定下一步方向。
    注意：不要在此处更新 state["best_result"]，该操作已在 tool_executor 中完成。
    """
    print("--- 🤔 Python 决策分支 3 (分析器) ---")
    current_history = state.get("history", [])
    
    try:
        analysis_data = json.loads(state.get("analysis_json", "{}"))
        status = analysis_data.get("status")
        
        # 提取规划描述
        plan = state.get("plan", {}).get("solution", {})
        plan_desc = f"{plan.get('site_type')} @ {plan.get('surface_binding_atoms')} (Index {plan.get('adsorbate_binding_indices')})"
        
        if status == "fatal_error":
            state["history"].append(f"【致命错误】 方案: {plan_desc} -> {analysis_data.get('message')}")
            return "end"

        # 1. 提取关键指标
        energy = analysis_data.get("most_stable_energy_eV", "N/A")
        bond_change = analysis_data.get("bond_change_count", 0)
        is_dissociated = analysis_data.get("is_dissociated", False)
        
        # 2. [关键增强] 提取位点滑移信息
        # 这能告诉 Planner："你原本想去 Bridge，但实际去了 Hollow"
        # 提取位点分析数据
        site_info = analysis_data.get("site_analysis", {})
        actual_site = site_info.get("actual_site_type", "unknown")
        planned_site = site_info.get("planned_site_type", "unknown")
        
        # 处理化学滑移
        is_chem_slip = site_info.get("is_chemical_slip", False)
        planned_syms = site_info.get("planned_symbols", [])
        actual_syms = site_info.get("actual_symbols", [])

        site_msg = f"位点: {actual_site} ({','.join(actual_syms)})"

        # 强化滑移的负反馈
        if is_chem_slip:
            # 极其强烈地告知 Planner：原计划是失败/不稳定的
            # 将 planned_syms 转为字符串，如 "Cu-Pd-Pd"
            planned_str = "-".join(planned_syms)
            actual_str = "-".join(actual_syms)
            
            site_msg = (
                f"⚠️【不稳定位点警告】⚠️: "
                f"规划的 {planned_site} ({planned_str}) 不稳定，吸附物自发滑移到了 {actual_site} ({actual_str})。"
                f"这意味着 {planned_str} 对该吸附物亲和力不足，后续请**禁止**再次测试 {planned_str} 类位点！"
            )
        
        elif actual_site != "unknown" and planned_site != "unknown" and actual_site != planned_site:
            # 普通警告：只是几何变了 (如 hollow -> ontop，但原子没变)
            site_msg = f"⚠️ 几何滑移: {planned_site} -> {actual_site}"

        # 3. 构建历史条目
        if status == "success":
            if is_dissociated:
                res_str = "❌ 分子解离"
            elif bond_change > 0:
                res_str = f"⚠️ 分子内重排(BC={bond_change})"
            else:
                res_str = "✅ 完美吸附"
                
            # 格式：[结果] 方案 -> 实际位点 | 能量
            history_entry = (
                f"【{res_str}】 {plan_desc} "
                f"-> {site_msg} | "
                f"E={energy:.3f} eV"
            )
        else:
            history_entry = f"【计算失败】 {plan_desc} -> 原因: {analysis_data.get('message')}"
            
        current_history.append(history_entry)

    except Exception as e:
        current_history.append(f"历史记录生成异常: {e}")

    # 更新历史记录
    state["history"] = current_history

    # 4. 决策逻辑
    if len(current_history) >= MAX_RETRIES:
        print(f"--- 决策: 已达到 {len(current_history)} 次尝试上限。流程结束。 ---")
        return "end"
    
    return "planner"

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
        "best_dissociated_result": None,
        "attempted_keys": [],
        "available_sites_description": None
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