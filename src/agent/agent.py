import os
import builtins
import math
import platform
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
from ase import units
from ase.constraints import FixAtoms
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
import autoadsorbate
import torch
from mace.calculators import mace_mp
from dotenv import load_dotenv

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

# --- 1. Define Agent State ---
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

# --- 2. Setup Environment and LLM ---
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
        touch_sphere = solution.get("touch_sphere_size", 2)
        ads_type = plan_json.get("adsorbate_type", "Molecule")

        # Ensure both are lists, otherwise return None (no exception raised)
        if not isinstance(surf_atoms, list) or not isinstance(ads_indices, list):
            return None

        # Convert to string, preserving order to distinguish heteronuclear dual-point adsorption direction (e.g., Mo-Pd vs Pd-Mo)
        surf_atoms_str = ",".join(str(s) for s in surf_atoms)
        ads_indices_str = ",".join(str(i) for i in ads_indices)

        key = f"{site_type}|{surf_atoms_str}|{ads_indices_str}|{ads_type}|{float(touch_sphere):.1f}"
        return key
    except Exception as e:
        print(f"--- ⚠️ make_plan_key failed: {e} ---")
        return None

# --- 3. Define LangGraph Nodes ---
def pre_processor_node(state: AgentState) -> dict:
    print("--- 🔬 Calling Pre-Processor Node ---")
    try:
        analysis = analyze_surface_sites(state["slab_path"])
        return {
            "surface_composition": analysis["surface_composition"],
            "available_sites_description": analysis["available_sites_description"]
        }
    except Exception as e:
        error_message = f"Error: Unable to read Slab file '{state['slab_path']}': {e}"
        print(f"--- Validation Failed: {error_message} ---")
        return {
            "validation_error": error_message,
            "surface_composition": None,
            "available_sites_description": None
        }

def solution_planner_node(state: AgentState) -> dict:
    print("--- 🧠 Calling Planner Node ---")
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
        "surface_composition": state.get("surface_composition", "Unknown"),
        "user_request": state["user_request"],
        "history": "\n".join(state["history"]) if state.get("history") else "None",
        "MAX_RETRIES": MAX_RETRIES,
        "autoadsorbate_context": atom_menu_json,
        "available_sites_description": state.get("available_sites_description", "None"),
    }
    
    if state.get("validation_error"):
        messages.append(HumanMessage(content=PLANNER_PROMPT.format(**prompt_input)))
        messages.append(AIMessage(content=json.dumps(state.get("plan", "{}"))))
        messages.append(HumanMessage(content=f"Your plan has logical errors: {state['validation_error']}. Please replan."))
    else:
        if state.get("history"):
            print(f"--- 🧠 Planner: Detected failure history, retrying... ---")
        messages.append(HumanMessage(content=PLANNER_PROMPT.format(**prompt_input)))

    response = llm.invoke(messages)
    
    try:
        parser = JsonOutputParser()

        content_str = response.content
        if content_str.startswith("```json"):
            content_str = content_str[7:-3].strip()
        
        plan_json = parser.parse(content_str)
        print(f"--- 🧠 Planner Plan Generated ---")
        return {
            "plan": plan_json,
            "messages": [AIMessage(content=response.content)],
            "validation_error": None
        }
    except Exception as e:
        print(f"--- 🛑 Planner Output JSON Parse Failed: {e} ---")
        print(f"--- Raw Output: {response.content} ---")
        return {
            "plan": None,
            "validation_error": f"False, Planner output format error: {e}. Please output strictly in JSON format.",
            "messages": [AIMessage(content=response.content)]
        }

def plan_validator_node(state: AgentState) -> dict:
    """ Node 2: Python Validator """
    print("--- 🐍 Calling Python Validator Node ---")

    try:
        # Use state["smiles"] (from initial input) instead of anything in the plan
        mol = Chem.MolFromSmiles(state["smiles"])
        if not mol:
            raise ValueError(f"RDKit returned None. SMILES might be invalid or contain valences RDKit cannot handle.")
    except Exception as e:
        error = f"False, Base SMILES string '{state['smiles']}' cannot be parsed by RDKit. This is an unrecoverable error. Please check SMILES. Error: {e}"
        print(f"--- Validation Failed: {error} ---")
        # This is a fatal error; we should stop retrying.
        # We notify the router by setting a special validation_error
        # Note: Ideally, the graph should have a "terminal_failure" state,
        # but currently we can only return to the planner and expect it to stop after N attempts.
        return {"validation_error": error, "plan": None} # Clear plan

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
        print("--- 🛑 Planner decided to terminate (converged or no more plans) ---")
        return {"validation_error": None}  # Pass directly

    site_type = solution.get("site_type", "")
    surf_atoms = solution.get("surface_binding_atoms", [])
    ads_indices = solution.get("adsorbate_binding_indices", [])
    if site_type == "ontop" and len(ads_indices) != 1:
        error = f"False, Rule 2: Python check failed. site_type 'ontop' must pair with 1 index (end-on), but got {len(ads_indices)}."
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "bridge" and len(ads_indices) not in [1, 2]:
        error = f"False, Rule 2: Python check failed. site_type 'bridge' must pair with 1 (end-on) or 2 (side-on) indices, but got {len(ads_indices)}."
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "hollow" and len(ads_indices) not in [1, 2]:
        error = f"False, Rule 2: Python check failed. site_type 'hollow' must pair with 1 (end-on) or 2 (side-on) indices, but got {len(ads_indices)}."
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if not isinstance(surf_atoms, list):
        error = "False, Plan JSON field 'surface_binding_atoms' must be a list."
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "ontop" and len(surf_atoms) != 1:
        error = (
            "False, Rule 2b: 'ontop' site requires surface_binding_atoms length of 1, "
            f"but got {len(surf_atoms)}."
        )
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "bridge" and len(surf_atoms) not in [1, 2]:
        error = (
            "False, Rule 2b: 'bridge' site requires surface_binding_atoms length of 1 or 2, "
            f"but got {len(surf_atoms)}."
        )
        print(f"--- Validation Failed: {error} ---")
        return {"validation_error": error}
    if site_type == "hollow" and len(surf_atoms) < 3:
        error = (
            "False, Rule 2b: 'hollow' site requires surface_binding_atoms to have at least 3 elements, "
            f"but got {len(surf_atoms)}."
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
                "False, This plan has already been attempted in the (site_type, surface_binding_atoms, adsorbate_binding_indices) space. "
                "Please plan a different combination."
            )
            print(f"--- Validation Failed: {error} ---")
            return {"validation_error": error}
    except Exception as e_dup:
        print(f"--- ⚠️ Exception during Duplicate-check: {e_dup} ---")

    print("--- Validation Succeeded ---")
    return {"validation_error": None}

def tool_executor_node(state: AgentState) -> dict:
    """ Node 4: Tool Executor """
    print("--- 🛠️ Calling Tool Executor Node ---")
    
    plan_json = state.get("plan", {})
    plan_solution = plan_json.get("solution", {})

    if not plan_solution:
        error_message = "Tool Executor Failed: 'plan' missing 'solution' dictionary."
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
        # 1. Read original Slab
        raw_slab_atoms = read_atoms_object(slab_path)
        tool_logs.append(f"Success: Read slab atoms from {slab_path}.")

        # 2. Prepare Slab before any energy calculation
        final_slab_atoms, is_expanded = prepare_slab(raw_slab_atoms)
        if is_expanded:
            tool_logs.append("Note: Slab was automatically expanded (2x2) for physical accuracy.")
        
        # 3. Initialize Calculator
        try:
            # Define relaxation parameters uniformly
            has_cuda = torch.cuda.is_available()

            if not has_cuda:
                opt_fmax = 0.10
                opt_steps = 200
                md_steps = 0
                md_temp = 150.0
                mace_model = "small"
                mace_device = "cpu"
                mace_precision = "float32" if platform.system() == "Darwin" else "float64"
                use_dispersion = False
            else:
                opt_fmax = 0.05
                opt_steps = 500
                md_steps = 20
                md_temp = 150.0
                mace_model = "large"
                mace_device = "cuda"
                mace_precision = "float64"
                use_dispersion = True

            # temp_calc here is mainly used for E_adsorbate and E_surface
            temp_calc = mace_mp(
                model=mace_model,
                device=mace_device,
                default_dtype=mace_precision,
                dispersion=use_dispersion,
            )

        except Exception as e_calc:
            raise ValueError(f"Failed to initialize MACE calculator: {e_calc}")

        # 4. Calculate E_surface
        try:
            e_surf_atoms = final_slab_atoms.copy()
            e_surf_atoms.calc = temp_calc

            # *** Apply constraints EXACTLY as in relax_atoms ***
            # tools.py::relax_atoms fixes *ALL* surface atoms.
            constraint = FixAtoms(indices=list(range(len(e_surf_atoms))))
            e_surf_atoms.set_constraint(constraint)

            print(f"--- 🛠️ Calculating single point energy of bare surface (all atoms fixed)... ---")

            E_surface = e_surf_atoms.get_potential_energy() # This is now a single point energy
            tool_logs.append(f"Success: E_surface = {E_surface:.4f} eV.")
            
        except Exception as e_surf_err:
            raise ValueError(f"Failed to calculate E_surface: {e_surf_err}")

        # 5. Create Fragment
        fragment_object = create_fragment_from_plan(
            original_smiles=state["smiles"],
            binding_atom_indices=plan_solution.get("adsorbate_binding_indices"),
            plan_dict=plan_json,
            to_initialize=plan_solution.get("conformers_per_site_cap", 4)
        )
        tool_logs.append(f"Success: Created fragment object from plan (SMILES: {state['smiles']}).")

        # 6. Calculate E_adsorbate
        try:
            adsorbate_only_atoms = fragment_object.conformers[0].copy()
            
            # Remove markers
            if adsorbate_only_atoms.info["smiles"] == "Cl":
                del adsorbate_only_atoms[0]
            elif adsorbate_only_atoms.info["smiles"] == "S1S":
                del adsorbate_only_atoms[:2]
                
            adsorbate_only_atoms.calc = temp_calc
            adsorbate_only_atoms.set_cell([20, 20, 20]) 
            adsorbate_only_atoms.center()
            
            print(f"--- 🛠️ Relaxing isolated {state['smiles']} molecule... ---")

            # Detect single atom molecule.
            if len(adsorbate_only_atoms) > 1:
                # Protocol 1: MD Warmup (Consistent with relax_atoms)
                if md_steps > 0:
                    MaxwellBoltzmannDistribution(adsorbate_only_atoms, temperature_K=md_temp)
                    dyn_md_ads = Langevin(adsorbate_only_atoms, 1 * units.fs, temperature_K=md_temp, friction=0.01)
                    dyn_md_ads.run(md_steps)
                    
                # Protocol 2: BFGS Optimization (Consistent with relax_atoms)
                BFGS(adsorbate_only_atoms, trajectory=None, logfile=None).run(fmax=opt_fmax, steps=opt_steps)
            else:
                print(f"--- 🛠️ Single atom adsorbate detected ({len(adsorbate_only_atoms)} atom), skipping vacuum relaxation. ---")
            
            E_adsorbate = adsorbate_only_atoms.get_potential_energy()
            tool_logs.append(f"Success: E_adsorbate = {E_adsorbate:.4f} eV.")
            
        except Exception as e_ads_err:
            raise ValueError(f"Failed to calculate E_adsorbate: {e_ads_err}")

        # 7. Place adsorbate
        generated_traj_file = populate_surface_with_fragment(
            slab_atoms=final_slab_atoms,
            fragment_object=fragment_object,
            plan_solution=plan_solution
        )
        tool_logs.append(f"Success: Fragment placed on slab. Configs saved to: {generated_traj_file}")

        initial_conformers = read(generated_traj_file, index=":")
        if not initial_conformers or len(initial_conformers) == 0:
            raise ValueError(f"populate_surface_with_fragment failed to generate any configs (empty trajectory: {generated_traj_file}).")
        
        # 8. Structure Relaxation
        print("--- ⏳ Starting structure relaxation... ---")
        slab_indices = list(range(len(final_slab_atoms)))
        relax_n = plan_solution.get("relax_top_n", 1)
        print(f"--- 🛠️ MACE using device: {mace_device} ---")

        final_traj_file = relax_atoms(
            atoms_list=list(initial_conformers),
            slab_indices=slab_indices,
            relax_top_n=relax_n,
            fmax=opt_fmax,
            steps=opt_steps,
            md_steps=md_steps,
            md_temp=md_temp,
            mace_model=mace_model,
            mace_device=mace_device,
            mace_precision=mace_precision,
            use_dispersion=use_dispersion
        )
        tool_logs.append(f"Success: Structure relaxation complete (Relaxed Top {relax_n}). Trajectory saved to '{final_traj_file}'.")
        
        # 9. Analyze Results
        print("--- 🔬 Calling Analysis Tool... ---")
        analysis_json_str = analyze_relaxation_results(
            relaxed_trajectory_file=final_traj_file,
            slab_atoms=final_slab_atoms,
            original_smiles=state["smiles"],
            plan_dict=plan_json,
            e_surface_ref=E_surface,
            e_adsorbate_ref=E_adsorbate
        )
        tool_logs.append(f"Success: Analysis tool executed.")
        print(f"--- 🔬 Analysis Result: {analysis_json_str} ---")
        analysis_json = json.loads(analysis_json_str)

        if analysis_json.get("status") == "success":
            e_new = analysis_json.get("most_stable_energy_eV")
            is_dissociated = analysis_json.get("is_dissociated")

            # Logic Branch A: Molecular State
            if not is_dissociated:
                e_old_mol = new_best_molecular.get("most_stable_energy_eV", float('inf')) if new_best_molecular else float('inf')
                if isinstance(e_new, (int, float)) and e_new < e_old_mol:
                    print(f"--- 🌟 New Best Found [Molecular]: {e_new:.4f} eV ---")
                    new_best_molecular = {
                        "most_stable_energy_eV": e_new,
                        "analysis_json": analysis_json,
                        "plan": state.get("plan"),
                        "result_type": "Perfect" if analysis_json.get("bond_change_count")==0 else "Isomerized"
                    }

            # Logic Branch B: Dissociated State
            else:
                e_old_diss = new_best_dissociated.get("most_stable_energy_eV", float('inf')) if new_best_dissociated else float('inf')
                if isinstance(e_new, (int, float)) and e_new < e_old_diss:
                    print(f"--- ⚠️ More stable [Dissociated] state found: {e_new:.4f} eV (will serve as thermodynamic reference) ---")
                    new_best_dissociated = {
                        "most_stable_energy_eV": e_new,
                        "analysis_json": analysis_json,
                        "plan": state.get("plan"),
                        "result_type": "Dissociated"
                    }

    except Exception as e:
        error_message = str(e)
        print(f"--- 🛑 Tool Execution Failed: {error_message} ---")
        tool_logs.append(f"Error during tool execution: {error_message}")
        analysis_json = {"status": "error", "message": f"Tool execution failed: {error_message}"}
        
    return {
        "messages": [ToolMessage(content="\n".join(tool_logs), tool_call_id="tool_executor")],
        "analysis_json": json.dumps(analysis_json),
        "best_result": new_best_molecular,
        "best_dissociated_result": new_best_dissociated
    }

def final_analyzer_node(state: AgentState) -> dict:
    """ 
    Node 5: Final Analyzer
    Function: Generate report based on global best results, distinguishing between perfect adsorption and intramolecular rearrangement.
    """
    print("--- ✍️ Calling Final Analyzer Node ---")
    llm = get_llm()
    
    # 1. Extract Data Sources
    best_result = state.get("best_result")
    best_dissociated = state.get("best_dissociated_result")
    last_analysis_json_str = state.get("analysis_json", "{}")
    
    try:
        last_analysis = json.loads(last_analysis_json_str)
    except:
        last_analysis = {}

    # 2. Decision: Which data to report?
    target_data = None
    plan_used = None
    source_type = "failure"
    result_label = "Unknown" # Used to prompt LLM for result type

    # Priority 1: History Best
    if best_result and isinstance(best_result, dict):
        print(f"--- ✍️ Final Analyzer: Locked global best plan (E={best_result.get('most_stable_energy_eV')} eV) ---")
        target_data = best_result.get("analysis_json")
        plan_used = best_result.get("plan")
        # If route_after_analysis saved result_type, read it
        result_label = best_result.get("result_type", "Best History")
        source_type = "success"
    
    # Priority 2: Last attempt success
    elif last_analysis.get("status") == "success" and last_analysis.get("is_covalently_bound"):
        print("--- ✍️ Final Analyzer: No history best, using success result from last step ---")
        target_data = last_analysis
        plan_used = state.get("plan")
        result_label = "Last Attempt"
        source_type = "success"
    
    else:
        print("--- ✍️ Final Analyzer: All attempts failed ---")
        source_type = "failure"

    # 3. Construct Prompt
    if source_type == "success":
        data_str = json.dumps(target_data, indent=2, ensure_ascii=False)
        plan_str = json.dumps(plan_used, indent=2, ensure_ascii=False)
        
        # [New] Prepare dissociated state comparison data
        diss_warning_context = ""
        if best_dissociated:
            e_mol = target_data.get("most_stable_energy_eV", 999)
            e_diss = best_dissociated.get("most_stable_energy_eV", 999)
            if e_diss < e_mol:
                delta_E = e_diss - e_mol
                diss_warning_context = (
                    f"\n*** SEVERE THERMODYNAMIC WARNING ***\n"
                    f"Although the user requested molecular adsorption, the system found a lower energy dissociated state in history.\n"
                    f"- Molecular State Energy: {e_mol:.3f} eV\n"
                    f"- Dissociated State Energy: {e_diss:.3f} eV (More stable by {abs(delta_E):.3f} eV)\n"
                    f"This means the reported molecular state is thermodynamically metastable and prone to spontaneous dissociation."
                )

        final_prompt = f"""
        You are a rigorous computational chemist. Your task is to write a final experimental report based on the provided [OBJECTIVE FACTS].

        !!! SEVERE WARNING & SCIENTIFIC STANDARDS !!!
        1. **Precision Judgment**: Due to hardware limits, calculations use float32 precision. Energy differences < 0.05 eV may be due to **"numerical noise"** or **"energy degeneracy"**. If you find sub-optimal sites with energy differences within this range, you MUST declare in the report that they are competitive at room temperature. **DO NOT** arbitrarily claim one is the unique absolute best.
        2. **Label Correction**: Tools might incorrectly label high-coordination adsorption (Hollow) as "desorbed" based on geometric distance.
        3. **Heterogeneity Judgment**: For alloy surfaces (e.g., Ru3Mo), the same type of site (e.g., Bridge Ru-Ru) may exist in multiple environments. If history shows different results for two attempts at Bridge sites, point out in the discussion that this is due to **"surface heterogeneity"**.
        4. **No Fabrication**: Strictly base on JSON data.

        **User Request:** {state['user_request']}

        **Best Adsorption Configuration Data:**
        ```json
        {data_str}
        ```

        {diss_warning_context}

        **Initial Plan:**
        ```json
        {plan_str}
        ```

        **Writing Requirements:**
        1.  **Conclusion:** Directly answer the user request. If energy degeneracy (<0.05 eV) exists, explicitly state that multiple competitive configurations exist.
        2.  **Data Support:** List `most_stable_energy_eV` (3 decimal places) and `final_bond_distance_A`.
        3.  **Geometric Details:** Describe `bonded_surface_atoms`, and explicitly mention specific atom indices (e.g., Ru #41) to reflect site uniqueness.
        4.  **Site Correction & Slip:** Describe if a slip occurred from `planned_site_type` to `actual_site_type`.
        5.  **Chemical State Judgment:** - **Perfect Adsorption**: `bond_change_count == 0`
            - **Isomerization/Rearrangement**: `bond_change_count > 0` but not dissociated
            - **Dissociation**: `is_dissociated == True`
        """
    else:
        fail_reason = last_analysis.get("message", "No stable configuration found.")
        final_prompt = f"""
        You are an error reporting assistant.
        Task: Politely inform the user that after multiple attempts, no stable adsorption configuration meeting the requirements was found.
        Error Log: "{fail_reason}"
        Please suggest the user check the SMILES or change the surface model. Do not fabricate results.
        """

    # 4. Call LLM
    response = llm.invoke([HumanMessage(content=final_prompt)])
    
    print("--- 🏁 Final Report Generated ---")
    return {"messages": [AIMessage(content=response.content)]}

# --- 4. Define Graph Logic Flow (Edges) ---
def route_after_validation(state: AgentState) -> str:
    print("--- 🤔 Python Decision Branch 1 (Validator) ---")
    if state.get("validation_error"):
        print(f"--- Decision: Plan failed, returning to Planner ---")
        return "planner"
    
    # Routing logic
    plan_json = state.get("plan", {})
    solution = plan_json.get("solution", {})
    if solution.get("action") == "terminate":
        print(f"--- Decision: Planner requested termination, going to Final Analyzer ---")
        return "final_analyzer"  # Skip Tool Executor, go directly to report
    
    else:
        print(f"--- Decision: Plan passed, going to Tool Executor ---")
        return "tool_executor"

def route_after_analysis(state: AgentState) -> str:
    """
    Simplified Router: Generates rich history and decides next step.
    """
    print("--- 🤔 Python Decision Branch 3 (Analyzer) ---")

    # 1. Priority Check: If previous Planner decided to terminate, and we just finished Final Analyzer,
    #    then we must end the process now.
    plan_solution = state.get("plan", {}).get("solution", {})
    if plan_solution.get("action") == "terminate":
        print("--- Decision: Termination signal detected (Terminate Action), process ending normally. ---")
        return "end"

    current_history = state.get("history", [])
    
    try:
        analysis_data = json.loads(state.get("analysis_json", "{}"))
        status = analysis_data.get("status")
        
        # Extract plan description
        plan = state.get("plan", {}).get("solution", {})
        plan_desc = f"{plan.get('site_type')} @ {plan.get('surface_binding_atoms')} (Index {plan.get('adsorbate_binding_indices')})"
        
        if status == "fatal_error":
            state["history"].append(f"【FATAL ERROR】 Plan: {plan_desc} -> {analysis_data.get('message')}")
            return "end"

        # 1. Extract Key Metrics
        energy = analysis_data.get("most_stable_energy_eV", "N/A")
        bond_change = analysis_data.get("bond_change_count", 0)
        is_dissociated = analysis_data.get("is_dissociated", False)
        
        # 2. Extract Site Slip Information
        site_info = analysis_data.get("site_analysis", {})
        actual_site = site_info.get("actual_site_type", "unknown")
        planned_site = site_info.get("planned_site_type", "unknown")
        
        # Handle Chemical Slip
        is_chem_slip = site_info.get("is_chemical_slip", False)
        planned_syms = site_info.get("planned_symbols", [])
        actual_syms = site_info.get("actual_symbols", [])

        # --- Define base site_msg ---
        site_msg = f"Site: {actual_site} ({','.join(actual_syms)})"

        # Reinforce negative feedback for slips
        if is_chem_slip:
            planned_str = "-".join(planned_syms)
            actual_str = "-".join(actual_syms)
            site_msg = (
                f"⚠️【Unstable Site Warning】⚠️: "
                f"Planned {planned_site} ({planned_str}) is unstable, adsorbate spontaneously slipped to {actual_site} ({actual_str})."
                f"This means {planned_str} has insufficient affinity for this adsorbate. Please **FORBID** testing {planned_str} type sites again!"
            )
        elif actual_site != "unknown" and planned_site != "unknown" and actual_site != planned_site:
            site_msg = f"⚠️ Geometric Slip: {planned_site} -> {actual_site}"

        # --- 3. [Fix Logic] Intelligently distinguish "New Best" from "Duplicate Convergence" ---
        tag = ""
        best_res = state.get("best_result")
        
        if best_res and isinstance(energy, (int, float)):
            best_e = best_res.get("most_stable_energy_eV", float('inf'))
            
            # Core Fix: Check if current run is the one that created Best Result
            # We compare plan objects. best_result stores the plan that produced it.
            current_plan_obj = state.get("plan")
            best_plan_obj = best_res.get("plan")
            
            # If current Plan is Best Plan, it's a "New Record", not "Duplicate"
            is_new_record = (current_plan_obj == best_plan_obj)
            
            if is_new_record:
                tag = " [🌟 New Best]"
            elif abs(energy - best_e) < 0.05: # Within 0.05 eV error
                # Not new record, and energy is same -> Duplicate path
                tag = " [🔄 Converged to known best]"
        
        # Append Tag
        site_msg = f"{site_msg}{tag}"

        # 4. Build History Entry
        if status == "success":
            if is_dissociated:
                res_str = "❌ Dissociated"
            elif bond_change > 0:
                res_str = f"⚠️ Rearrangement(BC={bond_change})"
            else:
                res_str = "✅ Perfect Adsorption"
                
            # Format: [Result] Plan -> Actual Site | Energy
            history_entry = (
                f"【{res_str}】 {plan_desc} "
                f"-> {site_msg} | "
                f"E={energy:.3f} eV"
            )
        else:
            history_entry = f"【Calculation Failed】 {plan_desc} -> Reason: {analysis_data.get('message')}"
            
        current_history.append(history_entry)

    except Exception as e:
        current_history.append(f"History generation exception: {e}")

    # Update History
    state["history"] = current_history

    # 5. Decision Logic
    if len(current_history) >= MAX_RETRIES:
        print(f"--- Decision: Reached {len(current_history)} attempts limit. Process ending. ---")
        return "end"
    
    return "planner"

# --- 5. Build and Compile Graph ---
def get_agent_executor():
    """ Build and compile the Adsorb-Agent state machine graph. """
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
        {"tool_executor": "tool_executor", "planner": "planner", "final_analyzer": "final_analyzer"}
    )
    workflow.add_conditional_edges(
        "final_analyzer",
        route_after_analysis,
        {"planner": "planner", "end": END}
    )
    return workflow.compile()

# --- 6. Run Program ---
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
    print("\n--- 🚀 Adsorb-Agent Started ---\n")
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
    print("\n--- 🏁 Adsorb-Agent Task Completed ---\n")
    print("Final Analysis Report:")
    if final_state and "messages" in final_state:
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage):
                print(msg.content)
                break
        else:
             print("No final AI message found.")

if __name__ == '__main__':
    exec_globals = builtins.__dict__.copy()
    exec_globals.update({
        "np": np, "pd": pd, "scipy": scipy, "sklearn": sklearn, "math": math,
        "ase": ase, "autoadsorbate": autoadsorbate, "torch": torch, "mace": mace,
    })
    
    main_cli()