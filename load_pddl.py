import re


def load_pddl_problem_line_by_line(file_path):
    """
    Load the initial state and goal conditions from a PDDL problem file by parsing line-by-line.
    Args:
        file_path (str): Path to the PDDL problem file.
    Returns:
        tuple: (initial_state, goal_conditions)
            initial_state (dict): A dict where keys are predicates (str),
                                  and values are sets of tuples representing arguments.
            goal_conditions (list): A list of tuples representing the goal predicates.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Clean lines: strip whitespace
    lines = [l.strip() for l in lines]

    initial_state = {}
    goal_conditions = {}

    # Flags to track where we are
    in_init = False
    in_goal = False

    init_lines = []
    goal_lines = []

    for line in lines:
        # Detect the start and end of :init section
        if '(:init' == line:
            in_init = True
            # If something follows :init on the same line, extract it
            after_init = line.split('(:init', 1)[-1].strip()
            # just in case it's something like "(:init" on one line
            if after_init.startswith("("):
                after_init = after_init[1:].strip()
            if after_init and not after_init.startswith(')'):
                init_lines.append(after_init)
            continue
        if in_init:
            # If we find a closing parenthesis that corresponds to the end of init block
            if ')' == line:
                # Extract the part before the closing parenthesis
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

    # Now parse init_lines to extract predicates
    # init_lines contain something like:
    # (inside_room water dining_room)
    # (obj_next_to coffe_maker coffee_filter)
    # etc.
    init_block_str = " ".join(init_lines)
    # Find all (pred arg...) patterns
    init_predicates = extract_predicates(init_block_str)

    for pred, args in init_predicates:
        if pred not in initial_state:
            initial_state[pred] = set()
        initial_state[pred].add(tuple(args))

    # Parse goal_lines
    goal_block_str = " ".join(goal_lines)
    # The goal often is in (and (pred ...) (pred ...))
    # We'll extract predicates inside that
    goal_predicates = extract_predicates(goal_block_str)

    # Filter out 'and' from goals
    for pred, args in goal_predicates:
        if pred != 'and':
            goal_conditions[pred] = set()
        goal_conditions[pred].add(tuple(args))

    return initial_state, goal_conditions


def extract_predicates(block_str):
    """
    Extract all predicates of the form (pred arg1 arg2 ...)
    from a string block. Returns a list of (pred, [args]) tuples.
    """
    predicates = []
    # We'll scan character-by-character and extract contents within parentheses
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
                # Parse current_expr
                parts = current_expr.strip().split()
                pred = parts[0]
                args = parts[1:]
                predicates.append((pred, args))
        else:
            if depth > 0:
                current_expr += ch
    return predicates