# load_pddl_behavior.py

def load_pddl_problem_line_by_line(file_path):
    """
    Load the initial state, goal conditions, and objects from a PDDL problem file by parsing line-by-line for the behavior domain.
    
    Args:
        file_path (str): Path to the PDDL problem file.
    Returns:
        tuple: (initial_state, goal_conditions, all_objects, characters)
            initial_state (dict): A dict where keys are predicates (str),
                                  and values are sets of tuples representing arguments.
            goal_conditions (dict): A dict where keys are predicates (str),
                                  and values are sets of tuples representing arguments.
            all_objects (set): A set of all objects.
            characters (set): A set of agents/characters.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Clean lines: strip whitespace
    lines = [l.strip() for l in lines]

    initial_state = {}
    goal_conditions = {}
    all_objects = set()
    characters = set()

    # Flags to track where we are
    in_objects = False
    in_init = False
    in_goal = False

    init_lines = []
    goal_lines = []
    objects_lines = []

    for line in lines:
        # Detect the start and end of :objects section
        if '(:objects' in line:
            in_objects = True
            after_objects = line.split('(:objects', 1)[-1].strip()
            if after_objects and not after_objects.startswith(')'):
                objects_lines.append(after_objects)
            continue
        if in_objects:
            if ')' in line:
                # The objects section ends here
                part = line.split(')')[0].strip()
                if part:
                    objects_lines.append(part)
                in_objects = False
            else:
                objects_lines.append(line)
            continue

        # Detect the start and end of :init section
        if '(:init' in line:
            in_init = True
            after_init = line.split('(:init', 1)[-1].strip()
            if after_init.startswith("("):
                after_init = after_init[1:].strip()
            if after_init and not after_init.startswith(')'):
                init_lines.append(after_init)
            continue
        if in_init:
            if ')' == line:
                part = line.split(')')[0].strip()
                if part:
                    init_lines.append(part)
                in_init = False
            else:
                init_lines.append(line)
            continue

        # Detect the start and end of :goal section
        if '(:goal' in line:
            in_goal = True
            after_goal = line.split('(:goal', 1)[-1].strip()
            if after_goal.startswith("("):
                after_goal = after_goal[1:].strip()
            if after_goal and not after_goal.startswith(')'):
                goal_lines.append(after_goal)
            continue
        # skip '(and' lines inside goal
        if line == "(and":
            continue
        if in_goal:
            if ')' in line:
                part = line.split(')')[0].strip()
                if part:
                    goal_lines.append(part)
                in_goal = False
            else:
                goal_lines.append(line)

    # Parse :objects section
    # Objects lines might look like:
    # agent_n_01_1 - agent basket_n_01_1 basket_n_01_2 - basket_n_01 ...
    # We'll parse them token by token:
    object_tokens = " ".join(objects_lines).split()
    # The pattern is generally: obj1 obj2 ... - type obj3 ... - type ...
    # We'll group them accordingly
    temp_objs = []
    current_type = None

    i = 0
    while i < len(object_tokens):
        token = object_tokens[i]
        if token == '-':
            # Next token is a type
            i += 1
            current_type = object_tokens[i]
            # Assign the objects collected so far to this type
            if current_type == 'agent':
                characters.update(temp_objs)
            else:
                all_objects.update(temp_objs)
            temp_objs = []
        else:
            # It's an object name
            temp_objs.append(token)
        i += 1

    # If there's a trailing list of objects without a '- type', assume object type
    if temp_objs:
        all_objects.update(temp_objs)

    # Parse :init section
    init_block_str = " ".join(init_lines)
    init_predicates = extract_predicates(init_block_str)
    for pred, args in init_predicates:
        if pred not in initial_state:
            initial_state[pred] = set()
        initial_state[pred].add(tuple(args))

    # Parse :goal section
    goal_block_str = " ".join(goal_lines)
    goal_predicates = extract_predicates(goal_block_str)
    for pred, args in goal_predicates:
        if pred != 'and' and pred not in goal_conditions:
            goal_conditions[pred] = set()
        if pred != 'and':
            goal_conditions[pred].add(tuple(args))

    return initial_state, goal_conditions, all_objects, characters


def extract_predicates(block_str):
    """
    Extract all predicates of the form (pred arg1 arg2 ...)
    from a string block. Returns a list of (pred, [args]) tuples.
    """
    predicates = []
    depth = 0
    current_expr = ""
    for ch in block_str:
        if ch == '(':
            depth += 1
            if depth == 1:
                current_expr = ""
        elif ch == ')':
            depth -= 1
            if depth == 0 and current_expr.strip():
                parts = current_expr.strip().split()
                pred = parts[0]
                args = parts[1:]
                predicates.append((pred, args))
        else:
            if depth > 0:
                current_expr += ch
    return predicates
