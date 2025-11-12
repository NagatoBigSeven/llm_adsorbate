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
    create_fragment_from_plan,
    populate_surface_with_fragment,
    relax_atoms, 
    save_ase_atoms,
    analyze_relaxation_results,
)
from src.agent.prompts import PLANNER_PROMPT

MAX_RETRIES = 4

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
    
    prompt_input = {
        "smiles": state["smiles"],
        "slab_xyz_path": state["slab_path"],
        "surface_composition": state.get("surface_composition", "未知"),
        "user_request": state["user_request"],
        "history": "\n".join(state["history"]) if state.get("history") else "无",
        "MAX_RETRIES": MAX_RETRIES
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
        print("--- 验证失败: Planner未能生成有效JSON。---")
        return {"validation_error": state.get("validation_error", "False, Planner 节点未能生成 JSON。")}
    
    if "solution" not in plan_json:
        error = "False, 方案 JSON 丢失 'solution' 键。"
        print(f"--- 验证失败: {error} ---")
        return {"validation_error": error}
        
    plan = plan_json.get("solution", {})
    if not plan:
        error = "False, 方案 JSON 丢失或格式错误（'solution' 键为空）。"
        print(f"--- 验证失败: {error} ---")
        return {"validation_error": error}

    orientation = plan.get("orientation", "")
    site_type = plan.get("site_type", "")
    surf_atoms = plan.get("surface_binding_atoms", [])
    ads_indices = plan.get("adsorbate_binding_indices", [])
    if site_type == "ontop" and len(surf_atoms) != 1:
        error = f"False, Rule 1: Python check failed. site_type is 'ontop' but surface_binding_atoms has {len(surf_atoms)} members (should be 1)."
        print(f"--- 验证失败: {error} ---")
        return {"validation_error": error}
    if site_type == "bridge" and len(surf_atoms) != 2:
        error = f"False, Rule 1: Python check failed. site_type is 'bridge' but surface_binding_atoms has {len(surf_atoms)} members (should be 2)."
        print(f"--- 验证失败: {error} ---")
        return {"validation_error": error}
    if site_type == "hollow" and len(surf_atoms) < 3: 
        error = f"False, Rule 1: Python check failed. site_type is 'hollow' but surface_binding_atoms has {len(surf_atoms)} members (should be >= 3)."
        print(f"--- 验证失败: {error} ---")
        return {"validation_error": error}
    if orientation == "end-on" and len(ads_indices) != 1:
        error = f"False, Rule 2: Python check failed. orientation is 'end-on' but adsorbate_binding_indices has {len(ads_indices)} members (should be 1)."
        print(f"--- 验证失败: {error} ---")
        return {"validation_error": error}
    if orientation == "side-on":
        if len(ads_indices) != 2:
            error = f"False, Rule 2: Python check failed. orientation is 'side-on' but adsorbate_binding_indices has {len(ads_indices)} members (should be 2)."
            print(f"--- 验证失败: {error} ---")
            return {"validation_error": error}
        elif site_type not in ["bridge", "hollow"]:
            error = f"False, Rule 3: Python check failed. orientation 'side-on' is physically incompatible with site_type '{site_type}'. 'side-on' must use 'bridge' or 'hollow'."
            print(f"--- 验证失败: {error} ---")
            return {"validation_error": error}
    print("--- 验证成功 ---")
    return {"validation_error": None}

def tool_executor_node(state: AgentState) -> dict:
    """ 节点 4: Tool Executor """
    print("--- 🛠️ 调用 Tool Executor 节点 ---")
    
    plan_solution = state["plan"].get("solution", {})

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
        slab_atoms = read_atoms_object(slab_path)
        tool_logs.append(f"成功: 已从 {slab_path} 读取 slab 原子。")
    
        # --- 计算参考态能量 (E_surface 和 E_adsorbate) ---
        # 1. 初始化一个统一的计算器和 *一致的* 弛豫参数
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

        # 2. 计算 E_surface
        try:
            e_surf_atoms = slab_atoms.copy()
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
    
        fragment_object = create_fragment_from_plan(
            original_smiles=state["smiles"],
            binding_atom_indices=plan_solution.get("adsorbate_binding_indices"),
            orientation=plan_solution.get("orientation"),
            to_initialize=plan_solution.get("conformers_per_site_cap", 5)
        )
        tool_logs.append(f"Success: Created fragment object from plan (SMILES: {state['smiles']}).")

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
            BFGS(adsorbate_only_atoms).run(fmax=opt_fmax, steps=opt_steps)
            
            E_adsorbate = adsorbate_only_atoms.get_potential_energy()
            tool_logs.append(f"成功: E_adsorbate = {E_adsorbate:.4f} eV。")
            
        except Exception as e_ads_err:
            raise ValueError(f"计算 E_adsorbate 失败: {e_ads_err}")

        generated_traj_file = populate_surface_with_fragment(
            slab_atoms=slab_atoms,
            fragment_object=fragment_object,
            plan_solution=plan_solution
        )
        tool_logs.append(f"成功: 已将片段放置在 slab 上。构型保存在: {generated_traj_file}")

        initial_conformers = read(generated_traj_file, index=":")
        if not initial_conformers or len(initial_conformers) == 0:
            raise ValueError(f"populate_surface_with_fragment 未能生成任何构型 (轨迹文件为空: {generated_traj_file})。")
        
        print("--- ⏳ 开始结构弛豫... ---")
        slab_indices = list(range(len(slab_atoms)))
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
        
        print("--- 🔬 调用分析工具... ---")
        analysis_json_str = analyze_relaxation_results(
            relaxed_trajectory_file=final_traj_file,
            slab_atoms=slab_atoms,
            original_smiles=state["smiles"],
            binding_atom_indices=plan_solution.get("adsorbate_binding_indices"),
            orientation=plan_solution.get("orientation"),
            e_surface_ref=E_surface,
            e_adsorbate_ref=E_adsorbate
        )
        tool_logs.append(f"成功: 分析工具已执行。")
        print(f"--- 🔬 分析结果: {analysis_json_str} ---")
        analysis_json = json.loads(analysis_json_str)
        
    except ValueError as e: # 特别捕获 _get_fragment 的失败
        if "RDKit" in str(e):
            # 这是一个致命的、不可重试的 SMILES 错误
            error_message = f"致命错误：RDKit 无法为 SMILES '{state['smiles']}' 生成构象: {e}"
            print(f"--- 🛑 {error_message} ---")
            analysis_json = {"status": "fatal_error", "message": error_message}
            # 不要抛出异常，而是返回这个特殊的 analysis_json
            return {
                "messages": [ToolMessage(content=error_message, tool_call_id="tool_executor")],
                "analysis_json": json.dumps(analysis_json)
            }
        else:
            raise e # 重新抛出，让外层捕获

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
    
    if analysis_data.get("status") == "success" and analysis_data.get("is_covalently_bound", False):
        final_prompt = """
        你是一名计算化学专家。
        你的规划和计算任务已成功执行，并且自动化分析工具已返回了 *基于事实* 的数据。

        **你的原始规划 (你当初的意图):**
        {plan}

        **自动化分析工具返回的真实数据 (客观事实):**
        {analysis_json}

        **你的任务:**
        1.  **解读数据:** 查看 `analysis_json`。`is_covalently_bound` 是 True 还是 False？`most_stable_energy_eV` 和 `final_bond_distance_A` 是多少？
        2.  **回答请求:** 根据这个 *真实数据*（而不是猜测），回答用户的原始请求：
            '{user_request}'
        3.  **提供关键信息:** 报告最稳定的能量、键长和保存的最佳结构文件名 (`best_structure_file`)。
        4.  **禁止幻觉:** 你的报告必须 100% 建立在上述 JSON 数据的客观事实上。
        """
        plan_str = json.dumps(state.get("plan", "{}"))
        prompt = final_prompt.format(
            plan=plan_str, 
            analysis_json=state["analysis_json"], 
            user_request=state["user_request"]
        )
    
    else:
        fail_message = analysis_data.get("message", "未知的分析错误。")
        if analysis_data.get("status") == "success" and not analysis_data.get("is_covalently_bound", False):
             fail_message = f"分析完成，但吸附物未与表面键合 (is_covalently_bound: false)。最终距离: {analysis_data.get('final_bond_distance_A', 'N/A')} Å。"
        
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
    else:
        print(f"--- 决策: 方案通过，前往执行 ---")
        return "tool_executor"

import json # 确保 json 已导入
...

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
            return "end"

        is_bound = analysis_data.get("is_covalently_bound", False) 
        plan_str = json.dumps(state.get("plan", "{}"))

        if status == "success" and is_bound:
            # --- 成功逻辑 ---
            energy = analysis_data.get("most_stable_energy_eV", "N/A")
            history_entry = f"成功的尝试: Plan={plan_str}, Result=键合成功, 能量={energy:.4f} eV。"
            print(f"--- 决策: 找到稳定键合 (E={energy:.4f} eV)。记录并返回规划器继续搜索... ---")

        else:
            # --- 失败逻辑 ---
            fail_reason = analysis_data.get("message", "计算失败或未键合。")
            if status == "success" and not is_bound:
                if "atom_1" in analysis_data and "atom_2" in analysis_data: # side-on
                    a1_dist = analysis_data["atom_1"].get("distance_A", "N/A")
                    a1_bound = analysis_data["atom_1"].get("is_bound", False)
                    a2_dist = analysis_data["atom_2"].get("distance_A", "N/A")
                    a2_bound = analysis_data["atom_2"].get("is_bound", False)
                    fail_reason = f"分析完成 (side-on)，但未完全键合。Atom 1 距离: {a1_dist} Å (Bound: {a1_bound}), Atom 2 距离: {a2_dist} Å (Bound: {a2_bound})."
                
                elif "final_bond_distance_A" in analysis_data: # end-on
                    dist = analysis_data.get("final_bond_distance_A", "N/A")
                    fail_reason = f"分析完成 (end-on)，但吸附物未与表面键合。最终距离: {dist} Å。"
                
                else:
                    fail_reason = analysis_data.get("message", "分析完成，但 is_covalently_bound 为 false。")

            history_entry = f"失败的尝试: Plan={plan_str}, Result={fail_reason}"
            print(f"--- 决策: 计算失败 ({fail_reason})。记录并返回规划器重试。 ---")

    except Exception as e:
        print(f"--- 决策: 分析路由失败 ({e})。返回规划器重试。 ---")
        history_entry = f"分析路由失败: {e}"

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
        "user_request": user_request,
        "plan": None,
        "validation_error": None,
        "messages": [HumanMessage(content=f"SMILES: {smiles}\nSLAB: {slab_path}\nREQUEST: {user_request}")],
        "analysis_json": None,
        "history": []
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
    for chunk in agent_executor.stream(initial_state, stream_mode="values"):
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