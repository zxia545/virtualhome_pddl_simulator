import re
from collections import defaultdict
import json
from load_pddl import load_pddl_problem_line_by_line


virtualhome_init_description= """
Your goal is to perform various household tasks in a virtual home environment. Tasks include browsing the internet, cooking food, making coffee, putting groceries in the fridge, setting up the table, washing clothes, washing hands, working, brushing teeth, drinking, going to the toilet, petting the cat, reading a book, taking a shower, washing dishes by hand or with a dishwasher, washing teeth, writing an email, changing TV channels, getting some water, listening to music, picking up the phone, relaxing on the sofa, turning on lights, watching TV, and more.

The environment consists of various rooms and objects such as the kitchen, living room, bathroom, bedroom, fridge, stove, coffee maker, sink, dishwasher, washing machine, toilet, shower, sofa, table, chairs, TV, computer, phone, books, clothes, dishes, and personal items like toothbrushes, cups, etc.
"""

ACTION_DESCRIPTIONS = """
ACTION: WALK_TOWARDS(?char: character, ?obj: object)
PRECONDITIONS:
- The character is not sitting.
- The character is not lying down.
EFFECTS:
- The character becomes next to the specified object.
- For all objects (?far_obj: object):
  - If ?far_obj is not next to ?obj, the character is not next to ?far_obj.
  - If ?close_obj is next to ?obj, the character becomes next to ?close_obj.
---

ACTION: WALK_INTO(?char: character, ?room: object)
PRECONDITIONS:
- The character is not sitting.
- The character is not lying down.
EFFECTS:
- The character becomes inside the specified room.
- For all objects (?far_obj: object):
  - If ?far_obj is not inside ?room, the character is not next to ?far_obj.
---

ACTION: FIND(?char: character, ?obj: object)
PRECONDITIONS:
- The character is next to the specified object.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: SIT(?char: character, ?obj: object)
PRECONDITIONS:
- The character is next to the specified object.
- The specified object is sittable.
- The character is not already sitting.
EFFECTS:
- The character becomes sitting.
- The character is now on top of the specified object.
---

ACTION: STANDUP(?char: character)
PRECONDITIONS:
- The character is either sitting or lying down.
EFFECTS:
- The character stops sitting.
- The character stops lying down.
---

ACTION: GRAB(?char: character, ?obj: object)
PRECONDITIONS:
- The object is grabbable.
- The character is next to the object.
- Not:
  - Exists another object (?obj2: object) such that:
    - ?obj is inside ?obj2.
    - ?obj2 is closed.
- Not:
  - The character is holding any object with both the left and right hands.
EFFECTS:
- When the character is holding any object with the left hand:
  - The character begins holding the specified object with the right hand.
- When the character is holding any object with the right hand:
  - The character begins holding the specified object with the left hand.
- When the character is not holding any objects with both hands:
  - The character begins holding the specified object with the right hand.
---

ACTION: OPEN(?char: character, ?obj: object)
PRECONDITIONS:
- The object can be opened.
- The object is closed.
- The character is next to the object.
- The object is not on another object.
EFFECTS:
- The object becomes open.
- The object is no longer closed.
---

ACTION: CLOSE(?char: character, ?obj: object)
PRECONDITIONS:
- The object can be opened.
- The object is open.
- The character is next to the object.
EFFECTS:
- The object becomes closed.
- The object is no longer on another object.
---

ACTION: PUT_ON(?char: character, ?obj1: object, ?obj2: object)
PRECONDITIONS:
- Either:
  - The character is next to ?obj2 and is holding ?obj1 with the left hand.
  - The character is next to ?obj2 and is holding ?obj1 with the right hand.
EFFECTS:
- ?obj1 becomes next to ?obj2.
- ?obj1 is now on top of ?obj2.
- The character releases ?obj1 from the left hand.
- The character releases ?obj1 from the right hand.
---

ACTION: PUT_ON_CHARACTER(?char: character, ?obj: object)
PRECONDITIONS:
- Either:
  - The character is holding ?obj with the left hand.
  - The character is holding ?obj with the right hand.
EFFECTS:
- ?obj is now on the character.
- The character releases ?obj from the left hand.
- The character releases ?obj from the right hand.
---

ACTION: PUT_INSIDE(?char: character, ?obj1: object, ?obj2: object)
PRECONDITIONS:
- Either:
  - The character is next to ?obj2, is holding ?obj1 with the left hand, and ?obj2 cannot be opened.
  - The character is next to ?obj2, is holding ?obj1 with the left hand, and ?obj2 is open.
  - The character is next to ?obj2, is holding ?obj1 with the right hand, and ?obj2 cannot be opened.
  - The character is next to ?obj2, is holding ?obj1 with the right hand, and ?obj2 is open.
EFFECTS:
- ?obj1 is now inside ?obj2.
- The character releases ?obj1 from the left hand.
- The character releases ?obj1 from the right hand.
---

ACTION: SWITCH_ON(?char: character, ?obj: object)
PRECONDITIONS:
- The object has a switch.
- The object is off.
- The object is plugged in.
- The character is next to the object.
EFFECTS:
- The object becomes on.
- The object is no longer off.
---

ACTION: SWITCH_OFF(?char: character, ?obj: object)
PRECONDITIONS:
- The object has a switch.
- The object is on.
- The character is next to the object.
EFFECTS:
- The object becomes off.
- The object is no longer on.
---

ACTION: DRINK(?char: character, ?obj: object)
PRECONDITIONS:
- Either:
  - The object is drinkable and the character is holding it with the left hand.
  - The object is drinkable and the character is holding it with the right hand.
  - The object is a recipient and the character is holding it with the left hand.
  - The object is a recipient and the character is holding it with the right hand.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: TURN_TO(?char: character, ?obj: object)
PRECONDITIONS:
- *(No preconditions; always executable.)*
EFFECTS:
- The character becomes facing the specified object.
---

ACTION: LOOK_AT(?char: character, ?obj: object)
PRECONDITIONS:
- The character is facing the specified object.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: WIPE(?char: character, ?obj1: object, ?obj2: object)
PRECONDITIONS:
- Either:
  - The character is next to ?obj1 and is holding ?obj2 with the left hand.
  - The character is next to ?obj1 and is holding ?obj2 with the right hand.
EFFECTS:
- ?obj1 becomes clean.
- ?obj1 is no longer dirty.
---

ACTION: DROP(?char: character, ?obj: object, ?room: object)
PRECONDITIONS:
- Either:
  - The character is holding ?obj with the left hand and ?obj is inside ?room.
  - The character is holding ?obj with the right hand and ?obj is inside ?room.
EFFECTS:
- The character releases ?obj from the left hand.
- The character releases ?obj from the right hand.
---

ACTION: READ(?char: character, ?obj: object)
PRECONDITIONS:
- Either:
  - The object is readable and the character is holding it with the left hand.
  - The object is readable and the character is holding it with the right hand.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: TOUCH(?char: character, ?obj: object)
PRECONDITIONS:
- Either:
  - The object is readable, the character is holding it with the left hand, and the object is not inside any closed object.
  - The object is readable, the character is holding it with the right hand, and the object is not inside any closed object.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: LIE(?char: character, ?obj: object)
PRECONDITIONS:
- The object is lieable.
- The character is next to the object.
- The character is not already lying down.
EFFECTS:
- The character becomes lying down.
- The character is now on top of the specified object.
- The character is no longer sitting.
---

ACTION: POUR(?char: character, ?obj1: object, ?obj2: object)
PRECONDITIONS:
- Either:
  - ?obj1 is pourable, the character is holding it with the left hand, ?obj2 is a recipient, and the character is next to ?obj2.
  - ?obj1 is pourable, the character is holding it with the right hand, ?obj2 is a recipient, and the character is next to ?obj2.
  - ?obj1 is drinkable, the character is holding it with the left hand, ?obj2 is a recipient, and the character is next to ?obj2.
  - ?obj1 is drinkable, the character is holding it with the right hand, ?obj2 is a recipient, and the character is next to ?obj2.
EFFECTS:
- ?obj1 is now inside ?obj2.
---

ACTION: TYPE(?char: character, ?obj: object)
PRECONDITIONS:
- The object has a switch.
- The character is next to the object.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: WATCH(?char: character, ?obj: object)
PRECONDITIONS:
- The object is lookable.
- The character is facing the object.
- Not:
  - Exists another object (?obj2: object) such that:
    - ?obj is inside ?obj2.
    - ?obj2 is closed.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: MOVE(?char: character, ?obj: object)
PRECONDITIONS:
- The object is movable.
- The character is next to the object.
- Not:
  - Exists another object (?obj2: object) such that:
    - ?obj is inside ?obj2.
    - ?obj2 is closed.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: WASH(?char: character, ?obj: object)
PRECONDITIONS:
- The character is next to the object.
EFFECTS:
- The object becomes clean.
- The object is no longer dirty.
---

ACTION: SQUEEZE(?char: character, ?obj: object)
PRECONDITIONS:
- The character is next to the object.
- The object is clothes.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: PLUG_IN(?char: character, ?obj: object)
PRECONDITIONS:
- Either:
  - The character is next to ?obj, ?obj has a plug, and ?obj is unplugged.
  - The character is next to ?obj, ?obj has a switch, and ?obj is unplugged.
EFFECTS:
- ?obj becomes plugged in.
- ?obj is no longer unplugged.
---

ACTION: PLUG_OUT(?char: character, ?obj: object)
PRECONDITIONS:
- The character is next to ?obj.
- ?obj has a plug.
- ?obj is plugged in.
- ?obj is not on another object.
EFFECTS:
- ?obj becomes unplugged.
- ?obj is no longer plugged in.
---

ACTION: CUT(?char: character, ?obj: object)
PRECONDITIONS:
- The character is next to the object.
- The object is eatable.
- The object is cuttable.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: EAT(?char: character, ?obj: object)
PRECONDITIONS:
- The character is next to the object.
- The object is eatable.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: SLEEP(?char: character, ?obj: object)
PRECONDITIONS:
- Either:
  - The character is lying down.
  - The character is sitting.
EFFECTS:
- *(No direct effects specified.)*
---

ACTION: WAKE_UP(?char: character, ?obj: object)
PRECONDITIONS:
- Either:
  - The character is lying down.
  - The character is sitting.
EFFECTS:
- *(No direct effects specified.)*
---
"""
# Define a mapping from predicates to their English descriptions.
# You should expand this mapping based on your domain's predicates.
# Define a mapping from predicates to their English descriptions.
# Each template uses `{}` as placeholders for predicate arguments.
PREDICATE_MAPPING = {
    "closed": "{} is closed.",
    "open": "{} is open.",
    "on": "{} is on {}.",
    "off": "{} is off.",
    "plugged_in": "{} is plugged in.",
    "plugged_out": "{} is unplugged.",
    "sitting": "{} is sitting.",
    "lying": "{} is lying down.",
    "clean": "{} is clean.",
    "dirty": "{} is dirty.",
    "obj_ontop": "{} is on top of {}.",
    "ontop": "{} is on top of {}.",
    "on_char": "{} is on {}.",
    "inside_room": "{} is inside the {}.",
    "obj_inside": "{} is inside {}.",
    "inside": "{} is inside {}.",
    "obj_next_to": "{} is next to {}.",
    "next_to": "{} is next to {}.",
    "between": "{} is between {} and {}.",
    "facing": "{} is facing {}.",
    "holds_rh": "{} is holding {} with right hand.",
    "holds_lh": "{} is holding {} with left hand.",
    "grabbable": "{} is grabbable.",
    "cuttable": "{} is cuttable.",
    "can_open": "{} can be opened.",
    "readable": "{} is readable.",
    "has_paper": "{} has paper.",
    "movable": "{} is movable.",
    "pourable": "{} is pourable.",
    "cream": "{} is cream.",
    "body_part": "{} is a body part.",
    "cover_object": "{} is a cover object.",
    "surfaces": "{} has surfaces.",
    "person": "{} is a person.",
    "hangable": "{} is hangable.",
    "clothes": "{} are clothes.",
    "lookable": "{} can be looked at.",
    "has_switch": "{} has a switch.",
    "has_plug": "{} has a plug.",
    "drinkable": "{} is drinkable.",
    "recipient": "{} is a recipient.",
    "containers": "{} is a container.",
    "sittable": "{} is sittable.",
    "lieable": "{} is lieable.",
    "eatable": "{} is eatable."
}



# Function to parse the PDDL problem string
def parse_pddl_problem(pddl_text, initial_state, goals):
    
    
    
    task_name = re.search(r'\(define\s*\(problem\s+([^\s)]+)', pddl_text).group(1)
    domain = re.search(r'\(:domain\s+([^\s)]+)', pddl_text).group(1)
    
    # Parse objects
    objects_section = re.search(r'\(:objects\s+(.*?)\)', pddl_text, re.DOTALL).group(1)
    objects = parse_objects(objects_section)
    
    # Parse initial state
    # init_section = re.search(r'\(:init\s+(.*?)\)', pddl_text, re.DOTALL).group(1)
    # initial_state = parse_predicates(init_section)
    
    # # Parse goal
    # goal_section = re.search(r'\(:goal\s+(.*?)\)', pddl_text, re.DOTALL).group(1)
    # goals = parse_predicates(goal_section)
    
    return {
        "task_name": task_name,
        "domain": domain,
        "objects": objects,
        "initial_state": initial_state,
        "goals": goals
    }


def generate_diff_trajectory_description(added, removed):
	final_traj = ""

	added_sentences = predicates_to_sentences(added)
	removed_sentences = predicates_to_sentences(removed)

	num_added = len(added_sentences)
	num_removed = len(removed_sentences)

	for i in range(num_added):
		if i == num_added - 1:
			final_traj += f"{added_sentences[i][:-1]}."
		else:
			final_traj += f"{added_sentences[i][:-1]}, "

	for i in range(num_removed):
		if i == num_removed - 1:	
			final_traj += f"{removed_sentences[i][:-1]}."	
		else:
			final_traj += f"{removed_sentences[i][:-1]}, "

	if final_traj == "":
		final_traj = "Nothing happens."
	return final_traj

# Function to parse the PDDL domain string
def parse_pddl_domain(pddl_text):
    actions = {}
    
    # Find all actions
    action_blocks = re.findall(r'\(:action\s+([^\s]+)\s+(:parameters\s+\([^\)]+\))\s+(:precondition\s+\([^\)]+\))\s+(:effect\s+\([^\)]+\))\)', pddl_text, re.DOTALL)
    
    # Improved regex to capture multiple actions
    action_blocks = re.findall(r'\(:action\s+([^\s]+)\s+.*?\)', pddl_text, re.DOTALL)
    
    # Alternative approach: iterate through the text and extract actions
    action_iter = re.finditer(r'\(:action\s+([^\s]+)\s+(:parameters\s+\([^\)]+\))\s+(:precondition\s+\([^\)]+\))\s+(:effect\s+\([^\)]+\))\)', pddl_text, re.DOTALL)
    
    for match in action_iter:
        action_name = match.group(1)
        parameters = match.group(2)
        precondition = match.group(3)
        effect = match.group(4)
        
        # Parse parameters
        params = parse_parameters(parameters)
        
        # Parse preconditions
        preconds = parse_condition(precondition)
        
        # Parse effects
        effects = parse_condition(effect)
        
        actions[action_name] = {
            "parameters": params,
            "preconditions": preconds,
            "effects": effects
        }
    
    return actions

# Function to parse objects
def parse_objects(objects_text):
    objects = defaultdict(list)
    # Split by spaces and handle multiple types
    tokens = objects_text.replace('\n', ' ').split()
    current_type = None
    object_list = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == '-':
            i += 1
            current_type = tokens[i]
            objects[current_type].extend(object_list)
            object_list = []
        else:
            object_list.append(token)
        i += 1
    return objects

# Function to parse predicates
def parse_predicates(predicates_text):
    # Find all predicates enclosed in parentheses
    predicates = re.findall(r'\(([^()]+)\)', predicates_text)
    parsed_predicates = []
    for pred in predicates:
        parts = pred.split()
        predicate = parts[0]
        args = parts[1:]
        parsed_predicates.append((predicate, args))
    return parsed_predicates

# Function to parse parameters
def parse_parameters(parameters_text):
    # Extract parameters inside parentheses
    params = re.findall(r'\?([^\s)]+)', parameters_text)
    return params

# Function to parse conditions (preconditions and effects)
def parse_condition(condition_text):
    # This function can be enhanced to parse nested conditions
    # For simplicity, we'll return the raw condition string
    return condition_text

# Function to convert predicates to English sentences
def predicates_to_sentences(predicates):
    sentences = []
    for predicate in predicates:
        if predicate in PREDICATE_MAPPING:
            template = PREDICATE_MAPPING[predicate]
            for args in predicates[predicate]:
                if '{}' in template:
                    try:
                        sentence = template.format(*args)
                    except IndexError:
                        sentence = template.format(*args, *args)  # Fallback if not enough args
                else:
                    sentence = template
                sentences.append(sentence)
        else:
            # If predicate not in mapping, provide a generic description
            raise(f'Current predicate {predicate} not in mapping')
    return sentences

# Function to generate the English description
def generate_description(parsed_pddl, actions_description):
    description = []
    
    # Task Type
    description.append(f"### Task: {parsed_pddl['task_name'].replace('_', ' ').capitalize()}\n")
    
    # Object Summary
    description.append("**Objects:**")
    for obj_type, objs in parsed_pddl['objects'].items():
        count = len(objs)
        obj_list = ', '.join(objs)
        obj_type_formatted = obj_type.replace('_', ' ')
        plural = "s" if count != 1 else ""
        description.append(f"- {obj_type_formatted}{plural}: {obj_list}.")
    description.append("")  # Add a newline
    
    # Initial State
    description.append("**Initial State:**")
    initial_sentences = predicates_to_sentences(parsed_pddl['initial_state'])
    for sentence in initial_sentences:
        description.append(f"- {sentence}")
    description.append("")  # Add a newline
    
    # Goal State
    description.append("**Goal State:**")
    goal_sentences = predicates_to_sentences(parsed_pddl['goals'])
    for sentence in goal_sentences:
        description.append(f"- {sentence}")
    description.append("")  # Add a newline
    
    # Actions
    description.append("**Actions:**")
    description.append(actions_description)
    
    return '\n'.join(description)

# Helper function to format conditions into readable text
def format_condition(condition):
    if isinstance(condition, str):
        # Simple condition
        predicates = parse_predicates(condition)
        return ', '.join([f"{pred} {' '.join(args)}" for pred, args in predicates])
    elif isinstance(condition, tuple):
        # Handle logical operators
        return format_logical_condition(condition)
    else:
        return str(condition)

def format_logical_condition(condition):
    if not condition:
        return ""
    op = condition[0]
    if op == "and":
        return " and ".join([format_logical_condition(c) for c in condition[1:]])
    elif op == "or":
        return " or ".join([format_logical_condition(c) for c in condition[1:]])
    elif op == "not":
        return f"not ({format_logical_condition(condition[1])})"
    elif op == "exists":
        var, var_type, sub_cond = condition[1], condition[2], condition[3]
        return f"exists ({var} - {var_type}) {format_logical_condition(sub_cond)}"
    else:
        # Simple predicate
        pred = op
        args = condition[1:]
        return f"{pred.replace('_', ' ').capitalize()} {' '.join(args)}"

def generate_state_description(current_state):
    description = []
    description.append("**Current State:**")
    initial_sentences = predicates_to_sentences(current_state)
    for sentence in initial_sentences:
        description.append(f"- {sentence}")
    return '\n'.join(description)

def generate_diff_description(added, removed):
	description = []

	# Added Predicates
	description.append("\n**Added:**")
	added_sentences = predicates_to_sentences(added)
	if len(added_sentences) != 0:
		for sentence in added_sentences:
			description.append(f"- {sentence}")
	else:
		description.append(f'- None')
        
	# Removed Predicates
	description.append("\n**Removed:**")
	removed_sentences = predicates_to_sentences(removed)
	if len(removed_sentences) != 0:
		for sentence in removed_sentences:
			description.append(f"- {sentence}")
	else:
		description.append(f'- None')

	return '\n'.join(description)





def generate_init_description(problem_file_path):
	# Read problem PDDL
	with open(problem_file_path, 'r') as f:
		problem_pddl = f.read()

	# Parse problem PDDL
	initial_state, goals, all_objects, characters = load_pddl_problem_line_by_line(problem_file_path)
	parsed_problem = parse_pddl_problem(problem_pddl, initial_state, goals)

	# Generate English description
	description = generate_description(parsed_problem, ACTION_DESCRIPTIONS)

	full_description = virtualhome_init_description + '\n' + description
	return full_description


# def generate_action_description(action, action_args*, action_map):
     

# Example usage
if __name__ == "__main__":
    # Paths to your PDDL files
    domain_file_path = "virtual_pddls/virtualhome.pddl"  # Replace with actual domain PDDL file path
    problem_file_path = "virtual_pddls/183_2.pddl"  # Replace with your specific PDDL problem file path
    
    # Read domain PDDL
    with open(domain_file_path, 'r') as f:
        domain_pddl = f.read()
    
    # Read problem PDDL
    with open(problem_file_path, 'r') as f:
        problem_pddl = f.read()
    
    # Parse problem PDDL
    initial_state, goals, all_objects, characters = load_pddl_problem_line_by_line(problem_file_path)
    parsed_problem = parse_pddl_problem(problem_pddl, initial_state, goals)
    
    # Generate English description
    description = generate_description(parsed_problem, ACTION_DESCRIPTIONS)
    print(description)



