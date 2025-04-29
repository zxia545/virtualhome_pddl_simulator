# evaluate_llm_virtualhome.py

import os
import argparse
import concurrent.futures
import time
import json
import re
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Optional

# Assuming these custom modules are in the same directory or Python path
from utils import (
    start_vllm_server_with_gpus,
    stop_vllm_server,
    chat_completion,
    create_output_directory,
    allocate_gpus,
    write_jsonl # Added for potential summary saving
)
from load_pddl import (
    load_pddl_problem_line_by_line,
)
from virtualhome_pddl_to_text import (
    generate_diff_description,
    generate_diff_trajectory_description
)
from run_and_act_virtualhome import *
# Import necessary functions and the 'actions' dictionary from run_and_act_virtualhome
# IMPORTANT: Ensure 'actions' is defined appropriately in this module,
# possibly loaded from a file or defined directly.
# If it's dynamically generated, it might need to be passed explicitly.
try:
    from run_and_act_virtualhome import (
        check_preconditions,
        apply_action,
        compute_state_diff,
        check_goal,
        actions, # Make sure this 'actions' dictionary is accessible
        all_objects as global_all_objects, # Handle potential global state if needed
        characters as global_characters   # Handle potential global state if needed
    )
    ACTIONS_DICT = actions # Store the imported actions
except ImportError as e:
    print(f"Error importing from run_and_act_virtualhome: {e}")
    print("Please ensure run_and_act_virtualhome.py exists and defines necessary functions/variables.")
    exit(1)
except NameError as e:
    print(f"Error accessing variables in run_and_act_virtualhome: {e}")
    print("Please ensure 'actions', 'all_objects', 'characters' are defined correctly.")
    exit(1)


# --- Action Parsing and Formatting (Copied from original script) ---

def format_action_execution(action_name: str, args: List[str]) -> str:
    capitalized_action = action_name.upper()
    joined_args = ", ".join(args)
    return f"{capitalized_action}({joined_args})"

def parse_gpt_action(response_text: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Parses the GPT response to extract the action name and arguments.
    Returns (action_name, [args]) in lowercase for the action name, or (None, None) if parsing fails.
    """
    # First try a pattern that handles if the entire action call might be inside ACTION(...)
    pattern_with_wrapper = r"^(?:ACTION:?\s*\()?([A-Z_]+)\((.*?)\)\)?\s*$"
    # If that fails, try a simpler pattern
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

# --- Environment Interaction Logic (Adapted from original script) ---

def execute_action(
    action_name: str,
    args: List[str],
    state: Dict[str, Any],
    local_actions: Dict[str, Any], # Pass actions explicitly
    all_objects: List[str],
    characters: List[str]
) -> Tuple[Dict[str, Any], Optional[List[str]], Optional[List[str]], Optional[str]]:
    """Executes a single action if valid."""
    if action_name not in local_actions:
        return state, None, None, f"Invalid action: {action_name.upper()}."

    action_def = local_actions[action_name]

    # --- Handle potential global state from run_and_act_virtualhome ---
    # This is a workaround if the imported functions rely on global state.
    # It's better if these functions take objects/characters as parameters.
    original_globals = {}
    if 'global_all_objects' in globals() and 'global_characters' in globals():
         original_globals = {
             'all_objects': global_all_objects.copy() if global_all_objects else None,
             'characters': global_characters.copy() if global_characters else None
         }
         global_all_objects.clear()
         global_all_objects.extend(all_objects)
         global_characters.clear()
         global_characters.extend(characters)
    # --- End Handle global state ---

    preconditions_met = check_preconditions(state, action_def, args)

    # --- Restore potential global state ---
    if original_globals:
        if original_globals['all_objects'] is not None:
            global_all_objects.clear()
            global_all_objects.extend(original_globals['all_objects'])
        if original_globals['characters'] is not None:
            global_characters.clear()
            global_characters.extend(original_globals['characters'])
    # --- End Restore global state ---

    if not preconditions_met:
        return state, None, None, (f"Preconditions not met for "
                                   f"{format_action_execution(action_name, args)}.")

    # Apply the action
    old_state = deepcopy(state)
    # Apply action might also depend on globals, handle similarly if needed
    new_state = apply_action(state, action_def, args)
    added, removed = compute_state_diff(old_state, new_state)
    return new_state, added, removed, None


# --- Worker Function for a Single PDDL Task ---

def run_single_pddl_task(
    pddl_file_path: str,
    sas_plan_path: str, # Included as requested, but not used in core logic
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
        "reason": "Unknown error",
        "steps_taken": 0,
        "final_progress_rate": 0.0,
        "messages_log": messages_log_path,
        "trajectory_log": trajectory_log_path,
        "error_details": None
    }

    try:
        # 1. Load PDDL Problem
        initial_state, goal_conditions, all_objects, characters = load_pddl_problem_line_by_line(pddl_file_path)
        init_description = generate_init_description(pddl_file_path) # Or generate from initial_state
        state = deepcopy(initial_state)

        # Use the globally imported actions dictionary for this task
        local_actions_dict = ACTIONS_DICT

        # 2. Prepare LLM Interaction
        # Demonstration example (same as original)
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

        # 3. Interaction Loop
        for step in range(max_steps):
            results["steps_taken"] = step + 1
            current_step_trajectory = {"step": step + 1, "action_requested": "", "observation": "", "progress_rate": 0.0}

            # Query LLM using chat_completion from utils
            try:
                # Note: Using a generic api_key="xxx" as vLLM doesn't typically require one
                gpt_response = chat_completion(
                    api_base=vllm_api_base,
                    model_name=model_name,
                    messages=messages,
                    max_tokens=150, # Adjust as needed
                    temperature=0.7 # Low temperature for deterministic planning
                )
            except Exception as e:
                results["status"] = "failed"
                results["reason"] = "LLM API Error"
                results["error_details"] = str(e)
                print(f"[Task {task_id}-{pddl_name}] Error querying LLM: {e}")
                break # Exit loop on LLM failure

            action_name, args = parse_gpt_action(gpt_response)
            current_gpt_response_log = f"LLM Response (Step {step+1}):\n{gpt_response}\n"

            if action_name is None:
                error_msg = ("I could not parse your action. Please reply with a single action "
                             "in uppercase format: ACTION(ARG1, ARG2, ...)")
                print(f"[Task {task_id}-{pddl_name}] Step {step+1}: Invalid action format from LLM.")
                messages.append({"role": "assistant", "content": gpt_response}) # Log LLM response
                messages.append({"role": "user", "content": error_msg})
                current_step_trajectory["action_requested"] = f"INVALID_FORMAT: {gpt_response}"
                current_step_trajectory["observation"] = error_msg
                final_trajectory.append(current_step_trajectory)
                continue # Ask LLM again

            # Log LLM's validly formatted action request
            messages.append({"role": "assistant", "content": gpt_response})
            formatted_action = format_action_execution(action_name, args)
            current_step_trajectory["action_requested"] = formatted_action

            if action_name == "current_state":
                print(f"[Task {task_id}-{pddl_name}] Step {step+1}: LLM requested CURRENT_STATE.")
                current_state_desc = generate_state_description(state) # Assumes this func exists
                current_step_trajectory["observation"] = f"Current State Provided:\n{current_state_desc}"
                user_update = (f"Current State:\n{current_state_desc}\n\n"
                               "Please propose your next action.")
                messages.append({"role": "user", "content": user_update})
                # No state change, just provide info
                is_goal_met, progress_rate = check_goal(state, goal_conditions)
                current_step_trajectory["progress_rate"] = progress_rate
                final_trajectory.append(current_step_trajectory)
                continue

            # Execute the action
            new_state, added, removed, error_msg = execute_action(
                action_name, args, state, local_actions_dict, all_objects, characters
            )

            if error_msg:
                print(f"[Task {task_id}-{pddl_name}] Step {step+1}: Action Error - {error_msg}")
                current_step_trajectory["observation"] = f"Action Failed: {error_msg}\nNothing happens."
                user_update = (f"{error_msg}\nPlease propose another action.")
                messages.append({"role": "user", "content": user_update})
                # State doesn't change, check progress on current state
                is_goal_met, progress_rate = check_goal(state, goal_conditions)
                current_step_trajectory["progress_rate"] = progress_rate
                final_trajectory.append(current_step_trajectory)
                continue

            # Action Succeeded
            state = new_state
            action_description = format_action_execution(action_name, args)
            print(f"[Task {task_id}-{pddl_name}] Step {step+1}: Executed {action_description}")

            obs_text = generate_diff_description(added, removed) # For user message
            this_step_observation = generate_diff_trajectory_description(added, removed) # For trajectory log
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
                results["status"] = "success"
                results["reason"] = "Goal reached"
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

        # 4. Final Status Update
        if not goal_reached:
            if results["steps_taken"] >= max_steps:
                 results["status"] = "failed"
                 results["reason"] = "Max steps reached"
                 print(f"[Task {task_id}-{pddl_name}] Max steps ({max_steps}) reached.")
                 # Final progress check
                 is_goal_met, progress_rate = check_goal(state, goal_conditions)
                 results["final_progress_rate"] = progress_rate
                 messages.append({"role": "user", "content": f"Max steps reached. Final progress: {progress_rate:.1f}%"})
            elif results["status"] == "failed" and results["reason"] == "Unknown error":
                 # If loop exited early due to non-LLM, non-max-steps error
                 results["reason"] = "Exited early (check logs)"
                 print(f"[Task {task_id}-{pddl_name}] Exited loop unexpectedly.")
                 is_goal_met, progress_rate = check_goal(state, goal_conditions)
                 results["final_progress_rate"] = progress_rate


    except FileNotFoundError as e:
        results["status"] = "failed"
        results["reason"] = "PDDL File Not Found"
        results["error_details"] = str(e)
        print(f"[Task {task_id}-{pddl_name}] Error: {e}")
    except Exception as e:
        results["status"] = "failed"
        results["reason":] = f"Runtime Error during task execution: {type(e).__name__}"
        results["error_details"] = str(e)
        import traceback
        print(f"[Task {task_id}-{pddl_name}] Runtime Error: {e}\n{traceback.format_exc()}")


    # 5. Save Logs
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
        # Update results if logs failed to save
        if results["status"] != "failed": # Don't overwrite existing failure reason
             results["status"] = "warning"
             results["reason"] = results.get("reason", "") + " | Log writing error"
        results["error_details"] = results.get("error_details", "") + f" | Log Error: {str(e)}"


    print(f"[Task {task_id}-{pddl_name}] Finished. Status: {results['status']}, Reason: {results['reason']}, Steps: {results['steps_taken']}, Progress: {results['final_progress_rate']:.1f}%")
    return results


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on VirtualHome PDDL tasks using vLLM.")
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

    # --- Validate Inputs ---
    if not os.path.isdir(args.pddl_dir):
        print(f"Error: PDDL directory not found: {args.pddl_dir}")
        return
    if not os.path.isdir(args.sas_dir):
        print(f"Warning: SAS plan directory not found: {args.sas_dir}. Ensure structure is correct if needed later.")
        # Allow continuing as SAS isn't used in core logic right now.

    # Create specific output directory for this model
    model_output_dir = create_output_directory(os.path.join(args.output_dir, args.model_name))
    print(f"Output will be saved in: {model_output_dir}")

    # Parse GPU list
    try:
        gpu_indices = [int(g.strip()) for g in args.gpus.split(',')]
    except ValueError:
        print(f"Error: Invalid GPU list format: {args.gpus}. Use comma-separated integers (e.g., '0,1').")
        return

    vllm_process = None
    vllm_api_base = f"http://localhost:{args.vllm_port}"

    try:
        # --- Start vLLM Server ---
        print(f"Starting vLLM server for model '{args.model_name}' ({args.model_path}) on port {args.vllm_port} using GPUs: {gpu_indices}...")
        vllm_process = start_vllm_server_with_gpus(
            model_path=args.model_path,
            model_name=args.model_name,
            port=args.vllm_port,
            gpus=gpu_indices
        )
        print("vLLM Server started successfully.")

        # --- Discover PDDL Tasks ---
        pddl_files = sorted([f for f in os.listdir(args.pddl_dir) if f.endswith(".pddl")])
        tasks = []
        for i, pddl_file in enumerate(pddl_files):
            pddl_path = os.path.join(args.pddl_dir, pddl_file)
            sas_file = pddl_file.replace(".pddl", ".sas_plan")
            sas_path = os.path.join(args.sas_dir, sas_file)
            if not os.path.exists(sas_path):
                 print(f"Warning: SAS plan file not found for {pddl_file} at {sas_path}. Skipping this task.")
                 # Or handle differently if SAS plans become mandatory
                 # continue
                 sas_path = None # Allow task to run without SAS path if it's optional

            tasks.append({
                "pddl_file_path": pddl_path,
                "sas_plan_path": sas_path,
                "output_dir": model_output_dir,
                "vllm_api_base": vllm_api_base,
                "model_name": args.model_name,
                "max_steps": args.max_steps,
                "task_id": i + 1
            })

        if not tasks:
            print("No valid PDDL tasks found. Exiting.")
            return

        print(f"Found {len(tasks)} PDDL tasks to process.")

        # --- Run Tasks Concurrently ---
        all_results = []
        start_time = time.time()
        print(f"Running tasks with {args.workers} worker(s)...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Use map to submit all tasks and get results in order
            future_to_task = {executor.submit(run_single_pddl_task, **task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task_info = future_to_task[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    print(f"[Main] Task {task_info['task_id']}-{os.path.basename(task_info['pddl_file_path'])} generated an exception: {exc}")
                    # Log this failure centrally too
                    all_results.append({
                        "pddl_name": os.path.basename(task_info['pddl_file_path']).replace(".pddl", ""),
                        "status": "failed",
                        "reason": f"Worker process error: {type(exc).__name__}",
                        "steps_taken": 0,
                        "final_progress_rate": 0.0,
                        "messages_log": None,
                        "trajectory_log": None,
                        "error_details": str(exc)
                    })

        end_time = time.time()
        print(f"\n--- Evaluation Summary ---")
        print(f"Total tasks: {len(tasks)}")
        print(f"Concurrency: {args.workers} workers")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        # --- Aggregate and Report Results ---
        success_count = sum(1 for r in all_results if r["status"] == "success")
        failed_count = len(all_results) - success_count
        success_rate = (success_count / len(all_results)) * 100 if all_results else 0

        print(f"\nSuccess Rate: {success_rate:.2f}% ({success_count} / {len(all_results)})")
        print(f"Failed Tasks: {failed_count}")

        # Detailed results per task
        print("\nIndividual Task Results:")
        all_results.sort(key=lambda x: x.get('pddl_name', '')) # Sort by name for consistent output
        for result in all_results:
            print(f"- {result.get('pddl_name', 'N/A'):<30} "
                  f"Status: {result.get('status', 'N/A'):<10} "
                  f"Reason: {result.get('reason', 'N/A'):<25} "
                  f"Steps: {result.get('steps_taken', 'N/A'):<4} "
                  f"Progress: {result.get('final_progress_rate', 0.0):.1f}%")
            if result.get("error_details"):
                print(f"  Error Details: {result['error_details']}")

        # Save summary results
        summary_file = os.path.join(model_output_dir, "summary_results.jsonl")
        try:
            write_jsonl(summary_file, all_results)
            print(f"\nDetailed results saved to: {summary_file}")
        except Exception as e:
            print(f"\nError saving summary file: {e}")


    except Exception as e:
        import traceback
        print(f"\nAn error occurred during the main execution: {e}")
        print(traceback.format_exc())
    finally:
        # --- Stop vLLM Server ---
        if vllm_process:
            print("\nStopping vLLM server...")
            stop_vllm_server(vllm_process)
            print("vLLM server stopped.")
        else:
             print("\nNo vLLM server process was found to stop.")


if __name__ == "__main__":
    main()