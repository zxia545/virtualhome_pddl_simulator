def load_pddl_problem_line_by_line(file_path):
    """
    Load the initial state, goal conditions, and objects from a PDDL problem file by parsing a combined string.
    This approach is more robust if the entire PDDL is mostly on one line.
    
    Returns:
        (initial_state, goal_conditions, all_objects, characters, object_types)
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Remove excessive whitespace
    content = " ".join(content.split())

    # Helper function to extract section content between parentheses after a given keyword
    def extract_section(keyword):
        start_idx = content.find(keyword)
        if start_idx == -1:
            return ""
        # Find the matching parenthesis after keyword
        # We'll start searching from the position of keyword
        depth = 0
        section_str = ""
        # Start scanning from the first '(' after keyword
        start_paren = content.find("(", start_idx)
        if start_paren == -1:
            return ""
        for i in range(start_paren, len(content)):
            ch = content[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            section_str += ch
            if depth == 0:
                break
        return section_str.strip()

    objects_str = extract_section("(:objects")
    init_str = extract_section("(:init")
    goal_str = extract_section("(:goal")

    object_types = {}
    initial_state = {}
    goal_conditions = {}
    all_objects = set()
    characters = set()

    # Parse objects
    # objects_str might look like: "(obj1 obj2 - type obj3 - type ...)"
    # Remove outer parentheses
    if objects_str.startswith('(:objects') and objects_str.endswith(')'):
        objects_str = objects_str[9:-1].strip()

    object_tokens = objects_str.split()
    temp_objs = []
    current_type = None
    i = 0
    while i < len(object_tokens):
        token = object_tokens[i]
        if token == '-':
            i += 1
            current_type = object_tokens[i]
            # Assign the objects collected so far to this type
            if current_type == 'agent':
                for obj_name in temp_objs:
                    characters.add(obj_name)
                    object_types[obj_name] = 'agent'
            else:
                for obj_name in temp_objs:
                    all_objects.add(obj_name)
                    object_types[obj_name] = current_type
            temp_objs = []
        else:
            temp_objs.append(token)
        i += 1

    # # If trailing objects without a type:
    # for obj_name in temp_objs:
    #     if obj_name not in object_types:
    #         object_types[obj_name] = 'object'
    #     all_objects.add(obj_name)

    # Parse :init
    # init_str might look like: "(pred arg1 arg2)(pred2 arg1) ..."
    # Remove outer parentheses if any
    init_str = init_str.strip()
    if init_str.startswith('(') and init_str.endswith(')'):
        init_str = init_str[1:-1].strip()
    init_predicates = extract_predicates(init_str)
    for pred, args in init_predicates:
        if pred not in initial_state:
            initial_state[pred] = set()
        initial_state[pred].add(tuple(args))

    # Parse :goal
    # goal_str might look like: "(and (pred arg1 arg2) (pred arg1))"
    # We'll remove the leading 'and' if present
    goal_str = goal_str.strip()
    if goal_str.startswith('(:goal (and'):
        # Remove '(and' and the corresponding closing ')'
        # Find the first '(' after '(and'
        inner = goal_str[len('(:goal (and'):].strip()
        # if inner.startswith('(') and inner.endswith(')'):
        #     inner = inner[1:-1].strip()
        goal_str = inner
    else:
        # Remove outer parentheses if any
        if goal_str.startswith('(') and goal_str.endswith(')'):
            goal_str = goal_str[1:-1].strip()

    goal_predicates = extract_predicates(goal_str)
    for pred, args in goal_predicates:
        if pred != 'and':
            if pred not in goal_conditions:
                goal_conditions[pred] = set()
            goal_conditions[pred].add(tuple(args))

    return initial_state, goal_conditions, all_objects, characters, object_types


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
                if parts:
                    pred = parts[0]
                    args = parts[1:]
                    predicates.append((pred, args))
        else:
            if depth > 0:
                current_expr += ch
    return predicates
