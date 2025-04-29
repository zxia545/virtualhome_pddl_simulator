# evaluate_llm_virtualhome.py (v2 - Cleaned global handling)

import os
import argparse
import concurrent.futures
import time
import json
import re
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Optional
import traceback # Import traceback for better error printing

# Assuming these custom modules are in the same directory or Python path
from utils import (
    start_vllm_server_with_gpus,
    stop_vllm_server,
    chat_completion,
    create_output_directory,
    # allocate_gpus, # Not used if running single server instance
    write_jsonl
)
# Import necessary functions from load_pddl
# Removed generate_init_description and generate_state_description if defined elsewhere
# or assuming they are part of run_and_act_virtualhome now based on original code structure
from load_pddl import (
    load_pddl_problem_line_by_line,
    # Add generate_init_description, generate_state_description back here if they are in load_pddl.py
)
# Import necessary functions from virtualhome_pddl_to_text
from virtualhome_pddl_to_text import (
    generate_diff_description,
    generate_diff_trajectory_description,
    # generate_init_description,  # Make sure these are imported from the correct place
    # generate_state_description
)

# Import necessary functions and the 'actions' dictionary from run_and_act_virtualhome
# Updated to remove direct reliance on globals from here
try:
    from run_and_act_virtualhome import (
        check_preconditions,
        apply_action,
        compute_state_diff,
        check_goal,
        actions, # Make sure this 'actions' dictionary is accessible
        generate_init_description, # Assuming these are defined here now
        generate_state_description # Assuming these are defined here now
    )
    ACTIONS_DICT = actions # Store the imported actions
except ImportError as e:
    print(f"Error importing from run_and_act_virtualhome: {e}")
    print("Please ensure run_and_act_virtualhome.py exists and defines necessary functions/variables.")
    exit(1)
except NameError as e:
    print(f"Error accessing variables in run_and_act_virtualhome: {e}")
    print("Please ensure 'actions', 'generate_init_description', 'generate_state_description' etc. are defined correctly.")
    exit(1)


# --- Action Parsing and Formatting (Unchanged) ---

def format_action_execution(action_name: str, args: List[str]) -> str:
    capitalized_action = action_name.upper()
    joined_args = ", ".join(args)
    return f"{capitalized_action}({joined_args})"

def parse_gpt_action(response_text: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Parses the GPT response to extract the action name and arguments.
    Returns (action_name, [args]) in lowercase for the action name, or (None, None) if parsing fails.
    """
    pattern_with_wrapper = r"^(?:ACTION:?\s*\()?([A-Z_]+)\((.*?)\)\)?\s*$"
    pattern_simple = r"^(?:ACTION:?\s*)?([A-Z_]+)\((.*?)\)\s*$"

    for line in response_text.splitlines():
        line = line.strip()
        match = re.match(pattern_with_wrapper, line)
        if match:
            action_name = match.group(1).lower()
            args_str = match.group(2).strip()
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()] if args_str else []
            return action_name, args

    for line in response_text.splitlines():
        line = line.strip()
        match = re.match(pattern_simple, line)
        if match:
            action_name = match.group(1).lower()
            args_str = match.group(2).strip()
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()] if args_str else []
            return action_name, args

    return None, None

# --- Environment Interaction Logic (UPDATED) ---

def execute_action(
    action_name: str,
    args: List[str],
    state: Dict[str, Any],
    local_actions: Dict[str, Any],
    all_objects: List[str],         # Now passed directly
    characters: List[str]          # Now passed directly
) -> Tuple[Dict[str, Any], Optional[List[str]], Optional[List[str]], Optional[str]]:
    """
    Executes a single action if valid.
    Assumes check_preconditions and apply_action can accept all_objects and characters if needed.
    """
    if action_name not in local_actions:
        return state, None, None, f"Invalid action: {action_name.upper()}."

    action_def = local_actions[action_name]

    # --- Check Preconditions ---
    # Pass the locally loaded all_objects and characters.
    # **** IMPORTANT ****: The 'check_preconditions' function in
    # 'run_and_act_virtualhome.py' MUST be modified to accept
    # 'all_objects' and 'characters' as parameters if it needs them.
    try:
        preconditions_met = check_preconditions(state, action_def, args)
    except TypeError as e:
        if "positional argument" in str(e) or "unexpected keyword argument" in str(e):
             print("\n\n*** ERROR ***")
             print("The `check_preconditions` function in `run_and_act_virtualhome.py`")
             print("likely needs to be updated to accept `all_objects` and `characters` as parameters.")
             print(f"Original error: {e}")
             print("Stopping execution.")
             # Re-raise or handle differently if needed, e.g., return specific error tuple
             raise e # Stop execution for this worker
        else:
            raise e # Re-raise other TypeErrors

    if not preconditions_met:
        return state, None, None, (f"Preconditions not met for "
                                   f"{format_action_execution(action_name, args)}.")

    # --- Apply the action ---
    old_state = deepcopy(state)
    # **** IMPORTANT ****: Similarly, if 'apply_action' in 'run_and_act_virtualhome.py'
    # needs access to all_objects or characters, it MUST be modified to accept them
    # as parameters, and they should be passed here.
    # Assuming apply_action does NOT need them based on original code structure:
    try:
         new_state = apply_action(state, action_def, args)
    except TypeError as e:
         # Add similar error checking/reporting for apply_action if needed
         raise e


    added, removed = compute_state_diff(old_state, new_state)
    return new_state, added, removed, None


# --- Worker Function for a Single PDDL Task (UPDATED Imports/Calls) ---

def run_single_pddl_task(
    pddl_file_path: str,
    sas_plan_path: Optional[str], # Can be None if not found
    output_dir: str,
    vllm_api_base: str,
    model_name: str,
    max_steps: int,
    task_id: int # For logging prefix
) -> Dict[str, Any]:
    """
    Runs the interactive simulation for a single PDDL file using the vLLM server.
    Returns a dictionary with results.
    """
    pddl_name = os.path.basename(pddl_file_path).replace(".pddl", "")
    print(f"[Task {task_id}-{pddl_name}] Starting...")

    task_output_dir = os.path.join(output_dir, pddl_name)
    os.makedirs(task_output_dir, exist_ok=True)
    messages_log_path = os.path.join(task_output_dir, f"{pddl_name}_messages.log")
    trajectory_log_path = os.path.join(task_output_dir, f"{pddl_name}_trajectory.log")

    results = {
        "pddl_name": pddl_name,
        "status": "failed",
        "reason": "Initialization error", # Default reason
        "steps_taken": 0,
        "final_progress_rate": 0.0,
        "messages_log": messages_log_path,
        "trajectory_log": trajectory_log_path,
        "error_details": None
    }

    try:
        # 1. Load PDDL Problem
        # These are the correct, task-specific objects and characters
        initial_state, goal_conditions, current_all_objects, current_characters = load_pddl_problem_line_by_line(pddl_file_path)

        # Use generate_init_description (ensure it's imported/defined correctly)
        # Pass pddl_file_path or initial_state as needed by the function definition
        init_description = generate_init_description(pddl_file_path) # Or (initial_state)
        state = deepcopy(initial_state)

        # Use the globally imported actions dictionary
        local_actions_dict = ACTIONS_DICT

        # Demonstration example to show GPT how we format steps and actions:
        demonstration_example = (
            "Below is an example demonstrating the desired action format and reporting style:\n\n"
            "**Step 1:** WALK_TOWARDS(character, coffe_maker)\n\n"
            "Action executed: WALK_TOWARDS(character, coffe_maker)\n\n"
            "**Added:**\n"
            "- character is next to coffee_filter.\n"
            "- character is next to water.\n"
            "- character is next to ground_coffee.\n"
            "- character is next to coffe_maker.\n\n"
            "**Removed:**\n"
            "- None\n"
            "Current Progress rate is: 40.0%\n\n"
            "**Step 2:** OPEN(character, coffe_maker)\n\n"
            "Action executed: OPEN(character, coffe_maker)\n\n"
            "**Added:**\n"
            "- coffe_maker is open.\n\n"
            "**Removed:**\n"
            "- coffe_maker is closed.\n"
            "Current Progress rate is: 20.0%\n\n"
            "**Step 3:** GRAB(character, ground_coffee)\n\n"
            "Action executed: GRAB(character, ground_coffee)\n\n"
            "**Added:**\n"
            "- character is holding ground_coffee with right hand.\n\n"
            "**Removed:**\n"
            "- None\n"
            "Current Progress rate is: 20.0%\n\n"
            "**Step 4:** PUT_ON(character, ground_coffee, coffe_maker)\n\n"
            "Action executed: PUT_ON(character, ground_coffee, coffe_maker)\n\n"
            "**Added:**\n"
            "- ground_coffee is on top of coffe_maker.\n\n"
            "**Removed:**\n"
            "- character is holding ground_coffee with right hand.\n"
            "Current Progress rate is: 40.0%\n\n"
            "**Step 5:** GRAB(character, coffee_filter)\n\n"
            "Action executed: GRAB(character, coffee_filter)\n\n"
            "**Added:**\n"
            "- character is holding coffee_filter with right hand.\n\n"
            "**Removed:**\n"
            "- None\n"
            "Current Progress rate is: 40.0%\n\n"
            "**Step 6:** CLOSE(character, coffe_maker)\n\n"
            "Action executed: CLOSE(character, coffe_maker)\n\n"
            "**Added:**\n"
            "- coffe_maker is closed.\n\n"
            "**Removed:**\n"
            "- None\n"
            "Current Progress rate is: 60.0%\n\n"
            "**Step 7:** SWITCH_ON(character, coffe_maker)\n\n"
            "Action executed: SWITCH_ON(character, coffe_maker)\n\n"
            "**Added:**\n"
            "- coffe_maker is on top of coffe_maker.\n\n"
            "**Removed:**\n"
            "- coffe_maker is off.\n"
            "Current Progress rate is: 80.0%\n\n"
            "**Step 8:** PUT_ON(character, coffee_filter, coffe_maker)\n\n"
            "Action executed: PUT_ON(character, coffee_filter, coffe_maker)\n\n"
            "**Added:**\n"
            "- coffee_filter is on top of coffe_maker.\n\n"
            "**Removed:**\n"
            "- character is holding coffee_filter with right hand.\n\n"
            "Goal reached after step: 8\n\n"
            "End of example.\n\n"
        )

        # System prompt for GPT with instructions about CURRENT_STATE
        system_prompt = (
            "You are an assistant controlling a character in a virtual home environment.\n\n"
            "You have a list of actions you can perform (in uppercase) and a goal state.\n\n"
            "At the start, I will provide the initial state and goal.\n"
            "After that, I will only provide observations of the last action or errors.\n"
            "If at any point you need to see the current state again, you can call the special action:\n"
            "CURRENT_STATE()\n\n"
            "Upon calling CURRENT_STATE(), I will provide the current state. "
            "After seeing it, you must propose your next action.\n\n"
            "You must respond with exactly one action in the form: ACTION(ARG1, ARG2, ...)\n"
            "The ACTION name must be uppercase and arguments separated by commas.\n\n"
            "If the last action was invalid or preconditions failed, I will tell you why.\n"
            "Then propose another valid action.\n"
            "Continue until the goal is reached.\n\n"
            "Please follow the formatting style shown in the example below:\n\n"
            f"{demonstration_example}"
            "Start by proposing the first action for the current problem now."
        )

        

        # Initial user prompt includes the initial state and goals.
        user_prompt = (
            f"Initial State:\n{init_description}\n\n"
            "Please propose the first action in the required format (e.g., OPEN(character, coffe_maker))."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        final_trajectory = []
        goal_reached = False
        results["reason"] = "Unknown error during run" # Update default reason

        # 3. Interaction Loop
        for step in range(max_steps):
            results["steps_taken"] = step + 1
            current_step_trajectory = {"step": step + 1, "action_requested": "", "observation": "", "progress_rate": 0.0}

            # Query LLM (Unchanged)
            try:
                gpt_response = chat_completion(
                    api_base=vllm_api_base, model_name=model_name, messages=messages,
                    max_tokens=150, temperature=0.7 # Adjusted temperature slightly, 0 is often too rigid
                )
            except Exception as e:
                results["status"], results["reason"] = "failed", "LLM API Error"
                results["error_details"] = str(e)
                print(f"[Task {task_id}-{pddl_name}] Error querying LLM: {e}")
                break # Exit loop

            # Parse Action (Unchanged)
            action_name, args = parse_gpt_action(gpt_response)
            messages.append({"role": "assistant", "content": gpt_response}) # Log LLM response regardless of format

            if action_name is None:
                error_msg = ("I could not parse your action. Please reply with a single action "
                             "in uppercase format: ACTION(ARG1, ARG2, ...)")
                print(f"[Task {task_id}-{pddl_name}] Step {step+1}: Invalid action format from LLM.")
                messages.append({"role": "user", "content": error_msg})
                current_step_trajectory["action_requested"] = f"INVALID_FORMAT: {gpt_response}"
                current_step_trajectory["observation"] = error_msg
                # Check progress even on error
                is_goal_met, progress_rate = check_goal(state, goal_conditions)
                current_step_trajectory["progress_rate"] = progress_rate
                final_trajectory.append(current_step_trajectory)
                continue

            # Log LLM's validly formatted action request
            formatted_action = format_action_execution(action_name, args)
            current_step_trajectory["action_requested"] = formatted_action

            # Handle CURRENT_STATE (Unchanged, ensure generate_state_description is available)
            if action_name == "current_state":
                print(f"[Task {task_id}-{pddl_name}] Step {step+1}: LLM requested CURRENT_STATE.")
                # Use generate_state_description (ensure it's imported/defined correctly)
                current_state_desc = generate_state_description(state)
                current_step_trajectory["observation"] = f"Current State Provided:\n{current_state_desc}"
                user_update = (f"Current State:\n{current_state_desc}\n\nPlease propose your next action.")
                messages.append({"role": "user", "content": user_update})
                is_goal_met, progress_rate = check_goal(state, goal_conditions)
                current_step_trajectory["progress_rate"] = progress_rate
                final_trajectory.append(current_step_trajectory)
                continue

            # Execute the action (UPDATED CALL)
            # Pass the task-specific objects and characters loaded earlier
            new_state, added, removed, error_msg = execute_action(
                action_name, args, state, local_actions_dict,
                current_all_objects, current_characters # Pass local variables
            )

            # Handle Action Outcome (Logic largely unchanged)
            if error_msg:
                print(f"[Task {task_id}-{pddl_name}] Step {step+1}: Action Error - {error_msg}")
                current_step_trajectory["observation"] = f"Action Failed: {error_msg}\nNothing happens."
                user_update = (f"{error_msg}\nPlease propose another action.")
                messages.append({"role": "user", "content": user_update})
                is_goal_met, progress_rate = check_goal(state, goal_conditions) # Check progress on current state
                current_step_trajectory["progress_rate"] = progress_rate
                final_trajectory.append(current_step_trajectory)
                continue

            # Action Succeeded
            state = new_state # Update state
            action_description = format_action_execution(action_name, args)
            print(f"[Task {task_id}-{pddl_name}] Step {step+1}: Executed {action_description}")

            obs_text = generate_diff_description(added, removed)
            this_step_observation = generate_diff_trajectory_description(added, removed)
            is_goal_met, progress_rate = check_goal(state, goal_conditions)
            current_step_trajectory["progress_rate"] = progress_rate
            current_step_trajectory["observation"] = this_step_observation + f"\nProgress: {progress_rate:.1f}%"

            user_update = (
                 f"Action executed: {action_description}\n\n"
                 f"{obs_text}\n"
                 f"Current Progress rate is: {progress_rate:.1f}%\n\n"
            )

            if is_goal_met:
                print(f"[Task {task_id}-{pddl_name}] Goal reached after step {step+1}!")
                user_update += "Goal reached! No more actions needed."
                messages.append({"role": "user", "content": user_update})
                results["status"], results["reason"] = "success", "Goal reached"
                results["final_progress_rate"] = progress_rate
                current_step_trajectory["observation"] += "\nGoal Reached!"
                final_trajectory.append(current_step_trajectory)
                goal_reached = True
                break # Exit loop
            else:
                 user_update += ("You must respond with exactly one action in the form: ACTION(ARG1, ARG2, ...)\n"
                                 "If you need the current state, call CURRENT_STATE(). Otherwise, propose the next action.")
                 messages.append({"role": "user", "content": user_update})
                 final_trajectory.append(current_step_trajectory)

        # 4. Final Status Update (Unchanged)
        if not goal_reached:
            if results["steps_taken"] >= max_steps:
                 results["status"], results["reason"] = "failed", "Max steps reached"
                 print(f"[Task {task_id}-{pddl_name}] Max steps ({max_steps}) reached.")
                 is_goal_met, progress_rate = check_goal(state, goal_conditions)
                 results["final_progress_rate"] = progress_rate
                 messages.append({"role": "user", "content": f"Max steps reached. Final progress: {progress_rate:.1f}%"})
            # Check if loop exited for other reasons (like API error handled above)
            elif results["status"] == "failed" and results["reason"] == "Unknown error during run":
                 results["reason"] = "Exited early (check logs)"
                 print(f"[Task {task_id}-{pddl_name}] Exited loop unexpectedly.")
                 is_goal_met, progress_rate = check_goal(state, goal_conditions)
                 results["final_progress_rate"] = progress_rate


    except FileNotFoundError as e:
        results["status"], results["reason"] = "failed", "PDDL File Not Found"
        results["error_details"] = str(e)
        print(f"[Task {task_id}-{pddl_name}] Error: {e}")
    except Exception as e:
        # Catch errors from execute_action (like the TypeError if functions aren't updated)
        # or other runtime errors within the task.
        results["status"], results["reason"] = "failed", f"Runtime Error: {type(e).__name__}"
        results["error_details"] = str(e)
        print(f"[Task {task_id}-{pddl_name}] Runtime Error: {e}\n{traceback.format_exc()}")


    # 5. Save Logs (Unchanged)
    try:
        with open(messages_log_path, "w", encoding='utf-8') as f:
            for message in messages:
                f.write(f"[{message['role'].upper()}]\n{message['content']}\n\n")
        with open(trajectory_log_path, "w", encoding='utf-8') as f:
            for step_data in final_trajectory:
                 f.write(f"--- Step {step_data['step']} ---\n")
                 f.write(f"Action Requested: {step_data['action_requested']}\n")
                 f.write(f"Observation & Progress:\n{step_data['observation']}\n")
                 f.write(f"Progress Rate: {step_data['progress_rate']:.1f}%\n\n")
    except Exception as e:
        print(f"[Task {task_id}-{pddl_name}] Error writing log files: {e}")
        if results["status"] != "failed":
             results["status"] = "warning"
             results["reason"] += " | Log writing error"
        results["error_details"] = (results.get("error_details", "") or "") + f" | Log Error: {str(e)}"


    print(f"[Task {task_id}-{pddl_name}] Finished. Status: {results['status']}, Reason: {results['reason']}, Steps: {results['steps_taken']}, Progress: {results['final_progress_rate']:.1f}%")
    return results


# --- Main Execution (Unchanged) ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on VirtualHome PDDL tasks using vLLM.")
    # Arguments (same as before)
    parser.add_argument("--model_path", type=str, required=True, help="Path to the HuggingFace model/weights.")
    parser.add_argument("--model_name", type=str, required=True, help="Name to assign the model in vLLM server.")
    parser.add_argument("--pddl_dir", type=str, required=True, help="Directory containing PDDL problem files.")
    parser.add_argument("--sas_dir", type=str, required=True, help="Directory containing corresponding SAS plan files (required structure, but not used in simulation).")
    parser.add_argument("--output_dir", type=str, default="eval_outputs", help="Directory to save evaluation results and logs.")
    parser.add_argument("--vllm_port", type=int, default=8000, help="Port for the vLLM OpenAI API server.")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU indices to use (e.g., '0,1').")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent PDDL tasks to run.")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum simulation steps per PDDL task.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for waiting for vLLM server start.")
    args = parser.parse_args()

    # --- Validate Inputs (same as before) ---
    if not os.path.isdir(args.pddl_dir): # ... (validation logic)
        print(f"Error: PDDL directory not found: {args.pddl_dir}")
        return
    if not os.path.isdir(args.sas_dir):
        print(f"Warning: SAS plan directory not found: {args.sas_dir}.")

    # Create output directory (same as before)
    model_output_dir = create_output_directory(os.path.join(args.output_dir, args.model_name))
    print(f"Output will be saved in: {model_output_dir}")

    # Parse GPU list (same as before)
    try: # ... (GPU parsing)
        gpu_indices = [int(g.strip()) for g in args.gpus.split(',')]
    except ValueError:
        print(f"Error: Invalid GPU list format: {args.gpus}.")
        return

    vllm_process = None
    vllm_api_base = f"http://localhost:{args.vllm_port}"

    try:
        # --- Start vLLM Server (same as before) ---
        print(f"Starting vLLM server...") # ... (server start logic)
        vllm_process = start_vllm_server_with_gpus(
            model_path=args.model_path, model_name=args.model_name,
            port=args.vllm_port, gpus=gpu_indices
        )
        print("vLLM Server started successfully.")

        # --- Discover PDDL Tasks (same as before) ---
        pddl_files = sorted([f for f in os.listdir(args.pddl_dir) if f.endswith(".pddl")])
        tasks = []
        for i, pddl_file in enumerate(pddl_files):
            pddl_path = os.path.join(args.pddl_dir, pddl_file)
            sas_file = pddl_file.replace(".pddl", ".sas_plan")
            sas_path = os.path.join(args.sas_dir, sas_file)
            if not os.path.exists(sas_path):
                 print(f"Warning: SAS plan file not found for {pddl_file} at {sas_path}. Proceeding without it.")
                 sas_path = None # Set to None if missing

            tasks.append({ # ... (task definition)
                "pddl_file_path": pddl_path,
                "sas_plan_path": sas_path, # Pass None if not found
                "output_dir": model_output_dir,
                "vllm_api_base": vllm_api_base,
                "model_name": args.model_name,
                "max_steps": args.max_steps,
                "task_id": i + 1
            })
        # ... (check if tasks exist)
        if not tasks:
            print("No valid PDDL tasks found. Exiting.")
            return
        print(f"Found {len(tasks)} PDDL tasks to process.")

        # --- Run Tasks Concurrently (same as before) ---
        all_results = []
        start_time = time.time()
        print(f"Running tasks with {args.workers} worker(s)...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            # ... (executor submission and result gathering)
             future_to_task = {executor.submit(run_single_pddl_task, **task): task for task in tasks}
             for future in concurrent.futures.as_completed(future_to_task):
                task_info = future_to_task[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc: # Catch errors raised from worker process
                    print(f"[Main] Task {task_info.get('task_id', 'N/A')}-{os.path.basename(task_info.get('pddl_file_path','N/A'))} generated an exception: {exc}")
                    print(traceback.format_exc()) # Print traceback from main process
                    all_results.append({
                        "pddl_name": os.path.basename(task_info.get('pddl_file_path','N/A')).replace(".pddl", ""),
                        "status": "failed",
                        "reason": f"Worker process error: {type(exc).__name__}",
                        "steps_taken": 0, "final_progress_rate": 0.0,
                        "messages_log": None, "trajectory_log": None,
                        "error_details": str(exc) + "\n" + traceback.format_exc() # Include traceback
                    })

        # --- Aggregate and Report Results (same as before) ---
        end_time = time.time()
        print(f"\n--- Evaluation Summary ---") # ... (reporting logic)
        success_count = sum(1 for r in all_results if r.get("status") == "success")
        failed_count = len(all_results) - success_count
        success_rate = (success_count / len(all_results)) * 100 if all_results else 0

        print(f"\nSuccess Rate: {success_rate:.2f}% ({success_count} / {len(all_results)})")
        print(f"Failed Tasks: {failed_count}")
        print("\nIndividual Task Results:")
        all_results.sort(key=lambda x: x.get('pddl_name', ''))
        for result in all_results: # ... (print individual results)
             print(f"- {result.get('pddl_name', 'N/A'):<30} "
                   # ... (rest of the print format)
                   f"Progress: {result.get('final_progress_rate', 0.0):.1f}%")
             if result.get("error_details") and result.get("status") != "success": # Show errors for non-success
                 # Indent error details for clarity
                 error_lines = str(result["error_details"]).splitlines()
                 print(f"  Error Details: {error_lines[0]}")
                 for line in error_lines[1:]:
                      print(f"                 {line}") # Align subsequent lines


        # Save summary results (same as before)
        summary_file = os.path.join(model_output_dir, "summary_results.jsonl")
        try: # ... (saving logic)
            write_jsonl(summary_file, all_results)
            print(f"\nDetailed results saved to: {summary_file}")
        except Exception as e:
            print(f"\nError saving summary file: {e}")


    except Exception as e:
        # Catch errors in the main setup/orchestration part
        print(f"\nAn error occurred during the main execution: {e}")
        print(traceback.format_exc())
    finally:
        # --- Stop vLLM Server (same as before) ---
        if vllm_process:
            print("\nStopping vLLM server...")
            stop_vllm_server(vllm_process)
            print("vLLM server stopped.")
        else:
             print("\nNo vLLM server process was found to stop.")


if __name__ == "__main__":
    main()