def load_pddl_problem_line_by_line(file_path):
    """
    Load the initial state, goal conditions, and objects from a PDDL problem file by parsing line-by-line.
    Args:
        file_path (str): Path to the PDDL problem file.
    Returns:
        tuple: (initial_state, goal_conditions, all_objects, characters)
            initial_state (dict): A dict where keys are predicates (str),
                                  and values are sets of tuples representing arguments.
            goal_conditions (dict): A dict where keys are predicates (str),
                                  and values are sets of tuples representing arguments.
            all_objects (set): A set of all objects.
            characters (set): A set of characters.
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
            if ')' == line:
                append_line = line.split(')')[0].strip()
                if append_line != '':
                    objects_lines.append(append_line)
                in_objects = False
            else:
                objects_lines.append(line)
            continue

        # Detect the start and end of :init section
        if '(:init' == line:
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

        # Detect the start and end of :goal section
        if '(:goal' == line:
            in_goal = True
            after_goal = line.split('(:goal', 1)[-1].strip()
            if after_goal.startswith("("):
                after_goal = after_goal[1:].strip()
            if after_goal and not after_goal.startswith(')'):
                goal_lines.append(after_goal)
            continue
        # skip the '(and' line
        if line == "(and":
            continue
        if in_goal:
            if ')' == line:
                part = line.split(')')[0].strip()
                if part:
                    goal_lines.append(part)
                in_goal = False
            else:
                goal_lines.append(line)

    # Parse :objects section
    if objects_lines:
        for objects_str in objects_lines:
            obj_tokens = objects_str.split()
            current_objects = []
            token_type = obj_tokens[-1]
            current_objects = []
            for token in obj_tokens:
                if token == "-":
                    if token_type == "character":
                        characters.update(current_objects)
                    elif token_type == "object":
                        all_objects.update(current_objects)
                    break
                elif token_type in {"character", "object"}:
                    current_objects.append(token)

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