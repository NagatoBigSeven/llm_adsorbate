import os
import builtins
import contextlib
import io
import math
import argparse
import json
from typing import TypedDict, List, Optional, Any
import numpy as np
import pandas as pd
import scipy
import sklearn
import ase
import autoadsorbate
import torch
import mace
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import JsonOutputParser

from src.tools.tools import (
    read_atoms_object, 
    get_sites_from_atoms, 
    get_fragment, 
    get_ads_slab, 
    relax_atoms, 
    save_ase_atoms,
    analyze_relaxation_results,
    generate_surrogate_smiles
)
from src.agent.prompts import PLANNER_PROMPT

# --- 1. 定义智能体状态 (Agent State) ---
class AgentState(TypedDict):
    smiles: str
    slab_path: str
    user_request: str
    plan: Optional[dict]
    validation_error: Optional[str]
    messages: List[BaseMessage]
    analysis_json: Optional[str]
    surrogate_smiles: Optional[str] 

# --- 2. 设置环境和 LLM ---
load_dotenv()

if not os.environ.get("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

def get_llm():
    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="ibm-granite/granite-4.0-h-micro",
        streaming=False, 
        max_completion_tokens=20000, 
        request_timeout=600, 
        seed=420
    )
    return llm

# --- 3. 定义 LangGraph 节点 (Nodes) ---
def solution_planner_node(state: AgentState) -> dict:
    print("--- 🧠 调用 Planner 节点 ---")
    llm = get_llm()
    messages = []
    
    prompt_input = {
        "smiles": state["smiles"],
        "slab_xyz_path": state["slab_path"],
        "user_request": state["user_request"]
    }
    
    if state.get("validation_error"):
        messages.append(HumanMessage(content=PLANNER_PROMPT.format(**prompt_input)))
        messages.append(AIMessage(content=json.dumps(state.get("plan", "{}"))))
        messages.append(HumanMessage(content=f"你的方案存在逻辑错误: {state['validation_error']}. 请重新规划一个新方案。"))
    else:
        messages.append(HumanMessage(content=PLANNER_PROMPT.format(**prompt_input)))

    response = llm.invoke(messages)
    
    try:
        parser = JsonOutputParser()
        plan_json = parser.parse(response.content)
        print(f"--- 🧠 Planner 方案已生成 ---")
        return {
            "plan": plan_json,
            "messages": [AIMessage(content=response.content)],
            "validation_error": None
        }
    except Exception as e:
        print(f"--- 🛑 Planner 输出 JSON 解析失败: {e} ---")
        return {
            "plan": None,
            "validation_error": f"False, Planner 输出格式错误: {e}. 请严格按 JSON 格式输出。",
            "messages": [AIMessage(content=response.content)]
        }

def plan_validator_node(state: AgentState) -> dict:
    """ 节点 2: Python 验证器 """
    print("--- 🐍 调用 Python 验证器节点 ---")
    plan_json = state.get("plan")
    if plan_json is None:
        print("--- 验证失败: Planner未能生成有效JSON。---")
        return {"validation_error": state.get("validation_error", "False, Planner 节点未能生成 JSON。")}
    plan = plan_json.get("solution", {})
    if not plan:
        error = "False, 方案 JSON 丢失或格式错误（未找到 'solution' 键）。"
        print(f"--- 验证失败: {error} ---")
        return {"validation_error": error}
    orientation = plan.get("orientation", "")
    site_type = plan.get("site_type", "")
    surf_atoms = plan.get("surface_binding_atoms", [])
    ads_atoms = plan.get("adsorbate_binding_atoms", [])
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
    if orientation == "end-on" and len(ads_atoms) != 1:
        error = f"False, Rule 2: Python check failed. orientation is 'end-on' but adsorbate_binding_atoms has {len(ads_atoms)} members (should be 1)."
        print(f"--- 验证失败: {error} ---")
        return {"validation_error": error}
    if orientation == "side-on" and len(ads_atoms) < 2:
        error = f"False, Rule 2: Python check failed. orientation is 'side-on' but adsorbate_binding_atoms has {len(ads_atoms)} members (should be < 2)."
        print(f"--- 验证失败: {error} ---")
        return {"validation_error": error}
    print("--- 验证成功 ---")
    return {"validation_error": None}


def smiles_translator_node(state: AgentState) -> dict:
    """ 节点 3: SMILES 翻译器 """
    print("--- 🔬 调用 SMILES 翻译器节点 ---")
    try:
        plan = state["plan"]["solution"]
        original_smiles = state["smiles"]
        surrogate_smiles = generate_surrogate_smiles(
            original_smiles=original_smiles,
            binding_atoms=plan["adsorbate_binding_atoms"],
            orientation=plan["orientation"]
        )
        return {
            "surrogate_smiles": surrogate_smiles,
            "messages": [ToolMessage(content=f"SMILES 翻译成功: {surrogate_smiles}", tool_call_id="smiles_translator")]
        }
    except Exception as e:
        print(f"--- 🛑 SMILES 翻译失败: {e} ---")
        return {
            "validation_error": f"False, SMILES 翻译器失败: {e}. 这可能是一个无效的键合方案（例如，在分子中未找到 '{plan.get('adsorbate_binding_atoms', ['N/A'])[0]}'）。请重新规划。",
            "messages": [ToolMessage(content=f"SMILES 翻译失败: {e}", tool_call_id="smiles_translator")]
        }

def tool_executor_node(state: AgentState) -> dict:
    """ 节点 4: Tool Executor """
    print("--- 🛠️ 调用 Tool Executor 节点 ---")
    plan = state["plan"]
    slab_path = state["slab_path"]
    surrogate_smiles = state["surrogate_smiles"]
    tool_logs = []
    analysis_json = None
    try:
        slab_atoms = read_atoms_object(slab_path)
        tool_logs.append(f"成功: 已从 {slab_path} 读取 slab 原子。")
        fragment_atoms = get_fragment(SMILES=surrogate_smiles)
        if fragment_atoms is None:
            raise ValueError(f"RDKit failed to parse the surrogate_smiles: '{surrogate_smiles}'.")
        tool_logs.append(f"成功: 已从 *SMILES '{surrogate_smiles}' 生成片段。")
        site_df = get_sites_from_atoms(slab_atoms)
        if plan["solution"]["site_type"] == "ontop" and not site_df[site_df.connectivity == 1].empty:
             selected_site_dict = site_df[site_df.connectivity == 1].iloc[0].to_dict()
             tool_logs.append(f"成功: 已过滤并选择第一个 'ontop' 位点。")
        else:
            selected_site_dict = site_df.iloc[0].to_dict()
            tool_logs.append(f"注意: 未找到精确 'ontop' 位点，已选择第一个可用位点。")
        ads_slab_atoms = get_ads_slab(slab_atoms, fragment_atoms, selected_site_dict)
        tool_logs.append(f"成功: 已将片段放置在 slab 上。")
        print("--- ⏳ 开始结构弛豫... ---")
        relaxed_atoms = relax_atoms(ads_slab_atoms, output_dir='./outputs')
        tool_logs.append(f"成功: 结构弛豫完成。弛豫轨迹保存在 './outputs/relax.traj'。")
        relaxed_xyz_path = './outputs/relaxed_ads_slab.xyz'
        save_ase_atoms(relaxed_atoms, relaxed_xyz_path)
        tool_logs.append(f"成功: 最终弛豫结构已保存到 '{relaxed_xyz_path}'。")
        print("--- 🔬 调用分析工具... ---")
        analysis_json = analyze_relaxation_results(
            plan=plan,
            relaxed_xyz_path=relaxed_xyz_path,
            original_slab_path=slab_path
        )
        tool_logs.append(f"成功: 分析工具已执行。")
        print(f"--- 🔬 分析结果: {analysis_json} ---")
        print("--- ✅ 工具执行完毕 ---")
    except Exception as e:
        print(f"--- 🛑 工具执行失败: {e} ---")
        tool_logs.append(f"Error during tool execution: {str(e)}")
        analysis_json = json.dumps({"status": "error", "message": f"工具执行失败: {str(e)}"})
    return {
        "messages": [ToolMessage(content="\n".join(tool_logs), tool_call_id="executor_run")],
        "analysis_json": analysis_json
    }

def final_analysis_node(state: AgentState) -> dict:
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
    except json.JSONDecodeError:
        analysis_data = {"status": "error", "message": "Analysis JSON was corrupted."}
    
    if analysis_data.get("status") == "success":
        # 成功路径：我们有真实数据
        final_prompt = """
        你是一名计算化学专家。
        你的规划和计算任务已成功执行，并且自动化分析工具已返回了 *基于事实* 的数据。

        **你的原始规划 (你当初的意图):**
        {plan}

        **自动化分析工具返回的真实数据 (客观事实):**
        {analysis_json}

        **你的任务:**
        1.  **解读数据:** 查看 `analysis_json`。`is_covalently_bound` 是 True 还是 False？`final_bond_distance_A` 是多少？
        2.  **回答请求:** 根据这个 *真实数据*（而不是猜测），回答用户的原始请求：
            '{user_request}'
        3.  **禁止幻觉:** 你的报告必须 100% 建立在上述 JSON 数据的客观事实上。
        """
        plan_str = json.dumps(state.get("plan", "{}"))
        prompt = final_prompt.format(
            plan=plan_str, 
            analysis_json=state["analysis_json"], 
            user_request=state["user_request"]
        )
    
    else:
        # 失败路径 - 严格禁止幻觉
        final_prompt = """
        你是一个错误报告助手。
        计算任务执行失败了。

        **你的唯一任务:**
        1.  礼貌地告知用户计算模拟失败。
        2.  **逐字** 报告 `analysis_json` 中的 "message" 字段。
        3.  **严格禁止** 尝试回答用户的原始科学问题。
        4.  **严格禁止** 猜测失败的原因或提供任何科学建议。
        
        **工具执行错误日志 (必须报告):**
        {analysis_json}
        
        **示例输出:**
        "抱歉，计算模拟执行失败。自动化工具报告了以下错误：<analysis_json["message"]>"
        """
        prompt = final_prompt.format(
            analysis_json=state.get("analysis_json", '{"status": "error", "message": "未知的分析错误。"}')
            # 移除了 {user_request} 来防止幻觉
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
        print(f"--- 决策: 方案通过，前往翻译 ---")
        return "smiles_translator"

def route_after_translation(state: AgentState) -> str:
    print("--- 🤔 Python 决策分支 2 (翻译器) ---")
    if state.get("validation_error"):
        print(f"--- 决策: 翻译失败，返回规划 ---")
        return "planner"
    else:
        print(f"--- 决策: 翻译成功，前往执行 ---")
        return "tool_executor"

# --- 5. 构建并编译图 (Graph) ---
def get_agent_executor():
    """ 构建并编译 Adsorb-Agent 状态机图。"""
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", solution_planner_node)
    workflow.add_node("plan_validator", plan_validator_node) 
    workflow.add_node("smiles_translator", smiles_translator_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("final_analyzer", final_analysis_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "plan_validator")
    workflow.add_edge("tool_executor", "final_analyzer")
    workflow.add_edge("final_analyzer", END)
    workflow.add_conditional_edges(
        "plan_validator",
        route_after_validation,
        {"smiles_translator": "smiles_translator", "planner": "planner"}
    )
    workflow.add_conditional_edges(
        "smiles_translator",
        route_after_translation,
        {"tool_executor": "tool_executor", "planner": "planner"}
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
        "surrogate_smiles": None
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Adsorb-Agent.")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string.")
    parser.add_argument("--slab_path", type=str, required=True, help="Path to the slab .xyz file.")
    parser.add_argument("--user_request", type=str, default="Find a stable adsorption configuration.", help="User's request.")
    return parser.parse_args()

# @weave.op()
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
        print(final_state["messages"][-1].content)

if __name__ == '__main__':
    exec_globals = builtins.__dict__.copy()
    exec_globals.update({
        "np": np, "pd": pd, "scipy": scipy, "sklearn": sklearn, "math": math,
        "ase": ase, "autoadsorbate": autoadsorbate, "torch": torch, "mace": mace,
    })
    
    main_cli()