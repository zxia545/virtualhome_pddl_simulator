from openai import OpenAI
from run_and_act_virtualhome import *
from load_pddl import *
from copy import deepcopy
import re

client = OpenAI(api_key = "xxx")

def query_gpt(messages, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content

def format_action_execution(action_name, args):
    capitalized_action = action_name.upper()
    joined_args = ", ".join(args)
    return f"{capitalized_action}({joined_args})"

import re

def parse_gpt_action(response_text):
    """
    Parses the GPT response to extract the action name and arguments.
    
    We consider these possible formats:
    - "ACTION: WALK_INTO(character, dining_room)"
    - "ACTION(WALK_INTO(character, dining_room))"
    - "WALK_INTO(character, dining_room)"

    Returns (action_name, [args]) in lowercase for the action name, or (None, None) if parsing fails.
    """

    # First try a pattern that handles if the entire action call might be inside ACTION(...)
    # e.g. ACTION(WALK_INTO(character, dining_room))
    # Explanation:
    # - Optional "ACTION" with optional ":" and a "(" right after
    # - Then capture the uppercase action name
    # - Then capture the arguments inside parentheses
    # - Optional closing ")" for ACTION(...)
    pattern_with_wrapper = r"^(?:ACTION:?\s*\()?([A-Z_]+)\((.*?)\)\)?\s*$"

    # If that fails, try a simpler pattern that doesn't expect wrapping parentheses around the whole action
    # e.g. ACTION: WALK_INTO(character, dining_room) or WALK_INTO(character, dining_room)
    pattern_simple = r"^(?:ACTION:?\s*)?([A-Z_]+)\((.*?)\)\s*$"

    # Try the first pattern
    for line in response_text.splitlines():
        line = line.strip()
        match = re.match(pattern_with_wrapper, line)
        if match:
            action_name = match.group(1).lower()
            args_str = match.group(2).strip()
            args = [arg.strip() for arg in args_str.split(',')] if args_str else []
            return action_name, args

    # If not matched, try the simpler pattern
    for line in response_text.splitlines():
        line = line.strip()
        match = re.match(pattern_simple, line)
        if match:
            action_name = match.group(1).lower()
            args_str = match.group(2).strip()
            args = [arg.strip() for arg in args_str.split(',')] if args_str else []
            return action_name, args

    # If no pattern matched, return None
    return None, None


def execute_action(action_name, args, state, actions, all_objects, characters):
    if action_name not in actions:
        return state, None, None, f"Invalid action: {action_name.upper()}."

    action_def = actions[action_name]

    # Check preconditions
    if not check_preconditions(state, action_def, args):
        return state, None, None, (f"Preconditions not met for "
                                   f"{format_action_execution(action_name, args)}.")

    # Apply the action
    old_state = deepcopy(state)
    new_state = apply_action(state, action_def, args)
    added, removed = compute_state_diff(old_state, new_state)
    return new_state, added, removed, None

def run_interactive_session(initial_state, goal_conditions, all_objects, characters, actions, max_steps=30):
    state = deepcopy(initial_state)

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

    init_description = generate_init_description("virtual_pddls/183_2.pddl")

    # Initial user prompt includes the initial state and goals.
    user_prompt = (
        f"Initial State:\n{init_description}\n\n"
        "Please propose the first action in the required format (e.g., OPEN(character, coffe_maker))."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    step = 0
    for step in range(max_steps):
        gpt_response = query_gpt(messages)
        action_name, args = parse_gpt_action(gpt_response)
        
        current_gpt_response = f"Current GPT response:\n{gpt_response}\n\n"

        if action_name is None:
            # GPT didn't provide a valid action format
            error_msg = ("I could not parse your action. Please reply with a single action "
                         "in uppercase format: ACTION(ARG1, ARG2, ...)")
            print(error_msg)
            messages.append({"role": "user", "content": current_gpt_response + error_msg})
            continue

        if action_name == "current_state":
            # GPT requests current state
            current_state_desc = generate_state_description(state)
            # Provide current state and ask for next action without executing anything
            print("GPT requested CURRENT_STATE. Providing current state.")
            messages.append({"role": "user", "content": f"{current_gpt_response} Current State:\n{current_state_desc}\n\nPlease propose your next action."})
            continue

        # Execute the action normally
        new_state, added, removed, error_msg = execute_action(action_name, args, state, actions, all_objects, characters)

        if error_msg:
            # Action failed
            print(f"Error: {error_msg}")
            messages.append({"role": "user", "content": f"{current_gpt_response} {error_msg}\nPlease propose another action."})
            continue

        # Action succeeded
        state = new_state
        action_description = format_action_execution(action_name, args)
        print(f"Action executed: {action_description}")

        # Generate observation text
        obs_text = generate_diff_description(added, removed)

        # Check goal progress
        is_goal_met, progress_rate = check_goal(state, goal_conditions)
        if is_goal_met:
            print(f"Goal reached after step {step+1}!")
            messages.append({"role": "user", "content": "Goal reached! No more actions needed."})
            break
        else:
            # After a normal action, only provide observations and progress,
            # no updated state unless GPT calls CURRENT_STATE().
            user_update = (
                f'{current_gpt_response}'
                f"Action executed: {action_description}\n\n"
                f"{obs_text}\n"
                f"Current Progress rate is: {progress_rate:.1f}%\n\n"
                "If you need the current state, call CURRENT_STATE(). Otherwise, propose the next action."
            )
            messages.append({"role": "user", "content": user_update})

    else:
        print("Max steps reached without reaching the goal.")
        # final prgress rate
        is_goal_met, progress_rate = check_goal(state, goal_conditions)
        print(f"Final Progress rate is: {progress_rate:.1f}%")
    
    
    # save messages to a file
    with open("gpt_messages.txt", "w") as f:
        for message in messages:
            f.write(f"{message['role']}: {message['content']}\n\n")



if __name__ == "__main__":
    import run_and_act_virtualhome
    pddl_name = "310_2"
    file_path = f"virtual_pddls/{pddl_name}.pddl"
    initial_state, goal_conditions, all_objects, characters = load_pddl_problem_line_by_line(file_path)
    # assign the global variables in run_and_act_virtualhome
    run_and_act_virtualhome.all_objects = all_objects
    run_and_act_virtualhome.characters = characters
    
    run_interactive_session(initial_state, goal_conditions, all_objects, characters, actions, max_steps=30)
