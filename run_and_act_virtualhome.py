
actions = {
    "walk_towards": {
        "parameters": ["?char", "?obj"],
        "preconditions": ("and",
            ("not", ("sitting", "?char")),
            ("not", ("lying", "?char"))
        ),
        "effects": {
            "add": [("next_to", "?char", "?obj")],
            "del": [],
            "when": [],
            "forall_remove": {
                # (forall (?far_obj - object) 
                #   (when (not (obj_next_to ?far_obj ?obj)) (not (next_to ?char ?far_obj))))
                "vars": [("?far_obj", "object")],
                "condition_negated_pred": ("obj_next_to", "?far_obj", "?obj"),
                "remove_pred": ("next_to", "?char", "?far_obj")
            },
            "forall_add": {
                # (forall (?close_obj - object)
                #   (when (obj_next_to ?close_obj ?obj) (next_to ?char ?close_obj)))
                "vars": [("?close_obj", "object")],
                "condition_pred": ("obj_next_to", "?close_obj", "?obj"),
                "add_pred": ("next_to", "?char", "?close_obj")
            }
        }
    },

    "walk_into": {
        "parameters": ["?char", "?room"],
        "preconditions": ("and",
            ("not", ("sitting", "?char")),
            ("not", ("lying", "?char"))
        ),
        "effects": {
            "add": [("inside", "?char", "?room")],
            "del": [],
            "when": [],
            "forall_remove": {
                # (forall (?far_obj - object)
                #   (when (not (inside_room ?far_obj ?room)) (not (next_to ?char ?far_obj))))
                "vars": [("?far_obj", "object")],
                "condition_negated_pred": ("inside_room", "?far_obj", "?room"),
                "remove_pred": ("next_to", "?char", "?far_obj")
            }
        }
    },

    "find": {
        "parameters": ["?char", "?obj"],
        # Precondition: (next_to ?char ?obj)
        # This is a single condition, so we can just use it directly:
        "preconditions": ("next_to", "?char", "?obj"),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "sit": {
        "parameters": ["?char", "?obj"],
        "preconditions": ("and",
            ("next_to", "?char", "?obj"),
            ("sittable", "?obj"),
            ("not", ("sitting", "?char"))
        ),
        "effects": {
            "add": [("sitting", "?char"), ("ontop", "?char", "?obj")],
            "del": [],
            "when": []
        }
    },

    "standup": {
        "parameters": ["?char"],
        "preconditions": ("or",
            ("sitting", "?char"),
            ("lying", "?char")
        ),
        "effects": {
            "add": [],
            "del": [("sitting", "?char"), ("lying", "?char")],
            "when": []
        }
    },

    "grab": {
        "parameters": ["?char", "?obj"],
        "preconditions": ("and",
            ("grabbable", "?obj"),
            ("next_to", "?char", "?obj"),
            ("not",
                ("exists", ("?obj2", "object"),
                    ("and",
                        ("obj_inside", "?obj", "?obj2"),
                        ("closed", "?obj2")
                    )
                )
            ),
            ("not",
                ("and",
                    ("exists", ("?obj3", "object"), ("holds_lh", "?char", "?obj3")),
                    ("exists", ("?obj4", "object"), ("holds_rh", "?char", "?obj4"))
                )
            )
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": [
                # (when (exists (?obj3 - object) (holds_lh ?char ?obj3)) (holds_rh ?char ?obj))
                (
                    ("exists", ("?obj3", "object"), ("holds_lh", "?char", "?obj3")),
                    {
                        "add": [("holds_rh", "?char", "?obj")],
                        "del": []
                    }
                ),
                # (when (exists (?obj4 - object) (holds_rh ?char ?obj4)) (holds_lh ?char ?obj))
                (
                    ("exists", ("?obj4", "object"), ("holds_rh", "?char", "?obj4")),
                    {
                        "add": [("holds_lh", "?char", "?obj")],
                        "del": []
                    }
                ),
                # (when (not (and (exists (?obj3 - object) (holds_lh ?char ?obj3)) (exists (?obj4 - object) (holds_rh ?char ?obj4))))
                #  (holds_rh ?char ?obj))
                (
                    ("not",
                        ("and",
                            ("exists", ("?obj3", "object"), ("holds_lh", "?char", "?obj3")),
                            ("exists", ("?obj4", "object"), ("holds_rh", "?char", "?obj4"))
                        )
                    ),
                    {
                        "add": [("holds_rh", "?char", "?obj")],
                        "del": []
                    }
                )
            ]
        }
    },

    "open": {
        "parameters": ["?char", "?obj"],
        "preconditions": ("and",
            ("can_open","?obj"),
            ("closed","?obj"),
            ("next_to","?char","?obj"),
            ("not",("on","?obj"))
        ),
        "effects": {
            "add": [("open","?obj")],
            "del": [("closed","?obj")],
            "when": []
        }
    },

    "close": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("can_open","?obj"),
            ("open","?obj"),
            ("next_to","?char","?obj")
        ),
        "effects": {
            "add": [("closed","?obj")],
            "del": [("on","?obj")], # Remove (on ?obj) if it exists
            "when": []
        }
    },

    "put_on": {
        "parameters": ["?char","?obj1","?obj2"],
        "preconditions": ("or",
            ("and",("next_to","?char","?obj2"),("holds_lh","?char","?obj1")),
            ("and",("next_to","?char","?obj2"),("holds_rh","?char","?obj1"))
        ),
        "effects": {
            "add": [("obj_next_to","?obj1","?obj2"),("obj_ontop","?obj1","?obj2")],
            "del": [("holds_lh","?char","?obj1"),("holds_rh","?char","?obj1")],
            "when": []
        }
    },

    "put_on_character": {
        "parameters": ["?char","?obj"],
        "preconditions": ("or",
            ("holds_lh","?char","?obj"),
            ("holds_rh","?char","?obj")
        ),
        "effects": {
            "add": [("on_char","?obj","?char")],
            "del": [("holds_lh","?char","?obj"),("holds_rh","?char","?obj")],
            "when": []
        }
    },

    "put_inside": {
        "parameters": ["?char","?obj1","?obj2"],
        "preconditions": ("or",
            ("and",("next_to","?char","?obj2"),("holds_lh","?char","?obj1"),("not",("can_open","?obj2"))),
            ("and",("next_to","?char","?obj2"),("holds_lh","?char","?obj1"),("open","?obj2")),
            ("and",("next_to","?char","?obj2"),("holds_rh","?char","?obj1"),("not",("can_open","?obj2"))),
            ("and",("next_to","?char","?obj2"),("holds_rh","?char","?obj1"),("open","?obj2"))
        ),
        "effects": {
            "add": [("obj_inside","?obj1","?obj2")],
            "del": [("holds_lh","?char","?obj1"),("holds_rh","?char","?obj1")],
            "when": []
        }
    },

    "switch_on": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("has_switch","?obj"),
            ("off","?obj"),
            ("plugged_in","?obj"),
            ("next_to","?char","?obj")
        ),
        "effects": {
            "add": [("on","?obj")],
            "del": [("off","?obj")],
            "when": []
        }
    },

    "switch_off": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("has_switch","?obj"),
            ("on","?obj"),
            ("next_to","?char","?obj")
        ),
        "effects": {
            "add": [("off","?obj")],
            "del": [("on","?obj")],
            "when": []
        }
    },

    "drink": {
        "parameters": ["?char","?obj"],
        "preconditions": ("or",
            ("and",("drinkable","?obj"),("holds_lh","?char","?obj")),
            ("and",("drinkable","?obj"),("holds_rh","?char","?obj")),
            ("and",("recipient","?obj"),("holds_lh","?char","?obj")),
            ("and",("recipient","?obj"),("holds_rh","?char","?obj"))
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "turn_to": {
        "parameters": ["?char","?obj"],
        "preconditions": (),  # empty preconditions = always true
        "effects": {
            "add": [("facing","?char","?obj")],
            "del": [],
            "when": []
        }
    },

    "look_at": {
        "parameters": ["?char","?obj"],
        "preconditions": ("facing","?char","?obj"),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "wipe": {
        "parameters": ["?char","?obj1","?obj2"],
        "preconditions": ("or",
            ("and",("next_to","?char","?obj1"),("holds_lh","?char","?obj2")),
            ("and",("next_to","?char","?obj1"),("holds_rh","?char","?obj2"))
        ),
        "effects": {
            "add": [("clean","?obj1")],
            "del": [("dirty","?obj1")],
            "when": []
        }
    },

    "drop": {
        "parameters": ["?char","?obj","?room"],
        "preconditions": ("or",
            ("and",("holds_lh","?char","?obj"),("obj_inside","?obj","?room")),
            ("and",("holds_rh","?char","?obj"),("obj_inside","?obj","?room"))
        ),
        "effects": {
            "add": [],
            "del": [("holds_lh","?char","?obj"),("holds_rh","?char","?obj")],
            "when": []
        }
    },

    "read": {
        "parameters": ["?char","?obj"],
        "preconditions": ("or",
            ("and",("readable","?obj"),("holds_lh","?char","?obj")),
            ("and",("readable","?obj"),("holds_rh","?char","?obj"))
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "touch": {
        "parameters": ["?char","?obj"],
        "preconditions": ("or",
            ("and",
                ("readable","?obj"),
                ("holds_lh","?char","?obj"),
                ("not",
                    ("exists",("?obj2","object"),
                        ("and",("obj_inside","?obj","?obj2"),("closed","?obj2"))
                    )
                )
            ),
            ("and",
                ("readable","?obj"),
                ("holds_rh","?char","?obj"),
                ("not",
                    ("exists",("?obj2","object"),
                        ("and",("obj_inside","?obj","?obj2"),("closed","?obj2"))
                    )
                )
            )
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "lie": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("lieable","?obj"),
            ("next_to","?char","?obj"),
            ("not",("lying","?char"))
        ),
        "effects": {
            "add": [("lying","?char"),("ontop","?char","?obj")],
            "del": [("sitting","?char")],
            "when": []
        }
    },

    "pour": {
        "parameters": ["?char","?obj1","?obj2"],
        "preconditions": ("or",
            ("and",("pourable","?obj1"),("holds_lh","?char","?obj1"),("recipient","?obj2"),("next_to","?char","?obj2")),
            ("and",("pourable","?obj1"),("holds_rh","?char","?obj1"),("recipient","?obj2"),("next_to","?char","?obj2")),
            ("and",("drinkable","?obj1"),("holds_lh","?char","?obj1"),("recipient","?obj2"),("next_to","?char","?obj2")),
            ("and",("drinkable","?obj1"),("holds_rh","?char","?obj1"),("recipient","?obj2"),("next_to","?char","?obj2"))
        ),
        "effects": {
            "add": [("obj_inside","?obj1","?obj2")],
            "del": [],
            "when": []
        }
    },

    "type": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("has_switch","?obj"),
            ("next_to","?char","?obj")
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "watch": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("lookable","?obj"),
            ("facing","?char","?obj"),
            ("not",
                ("exists",("?obj2","object"),
                    ("and",("obj_inside","?obj","?obj2"),("closed","?obj2"))
                )
            )
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "move": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("movable","?obj"),
            ("next_to","?char","?obj"),
            ("not",
                ("exists",("?obj2","object"),
                    ("and",("obj_inside","?obj","?obj2"),("closed","?obj2"))
                )
            )
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "wash": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("next_to","?char","?obj")
        ),
        "effects": {
            "add": [("clean","?obj")],
            "del": [("dirty","?obj")],
            "when": []
        }
    },

    "squeeze": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("next_to","?char","?obj"),
            ("clothes","?obj")
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "plug_in": {
        "parameters": ["?char","?obj"],
        "preconditions": ("or",
            ("and",("next_to","?char","?obj"),("has_plug","?obj"),("plugged_out","?obj")),
            ("and",("next_to","?char","?obj"),("has_switch","?obj"),("plugged_out","?obj"))
        ),
        "effects": {
            "add": [("plugged_in","?obj")],
            "del": [("plugged_out","?obj")],
            "when": []
        }
    },

    "plug_out": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("next_to","?char","?obj"),
            ("has_plug","?obj"),
            ("plugged_in","?obj"),
            ("not",("on","?obj"))
        ),
        "effects": {
            "add": [("plugged_out","?obj")],
            "del": [("plugged_in","?obj")],
            "when": []
        }
    },

    "cut": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("next_to","?char","?obj"),
            ("eatable","?obj"),
            ("cuttable","?obj")
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "eat": {
        "parameters": ["?char","?obj"],
        "preconditions": ("and",
            ("next_to","?char","?obj"),
            ("eatable","?obj")
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "sleep": {
        "parameters": ["?char","?obj"],
        "preconditions": ("or",
            ("lying","?char"),
            ("sitting","?char")
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    },

    "wake_up": {
        "parameters": ["?char","?obj"],
        "preconditions": ("or",
            ("lying","?char"),
            ("sitting","?char")
        ),
        "effects": {
            "add": [],
            "del": [],
            "when": []
        }
    }


}


def apply_action(state, action_def, args):
    """
    Applies an action to the current state based on its definition.
    Handles unconditional effects, conditional 'when' effects, and forall conditions.
    """
    new_state = deepcopy(state)
    param_map = {p: a for p, a in zip(action_def["parameters"], args)}
    eff = action_def["effects"]

    # Handle unconditional add/del effects
    if "add" in eff:
        apply_effects(new_state, eff["add"], param_map)
    if "del" in eff:
        remove_effects(new_state, eff["del"], param_map)

    # Handle effects with `exists` in the conditions
    if "when" in eff:
        for (cond, weff) in eff["when"]:
            if cond[0] == "exists":
                # Handle `exists` conditions
                var_decl = cond[1]  # Variable declaration (e.g., (?obj3 - object))
                sub_cond = cond[2]  # Sub-condition to evaluate
                var_name, var_type = var_decl

                # Iterate over all objects of the specified type
                for obj in all_objects:
                    # Skip objects that do not match the declared type
                    if var_type == "object" and obj in characters:
                        continue
                    if var_type == "character" and obj not in characters:
                        continue

                    # Extend param_map with this object
                    new_param_map = {**param_map, var_name: obj}

                    # Evaluate the sub-condition for this object
                    if _check_condition_list(new_state, sub_cond, new_param_map):
                        # Apply the effects if the condition is met
                        if "add" in weff:
                            apply_effects(new_state, weff["add"], param_map)
                        if "del" in weff:
                            remove_effects(new_state, weff["del"], param_map)

            elif cond[0] == "not":
                # Handle `not` conditions
                sub_cond = cond[1]  # Sub-condition under `not`
                if not _check_condition_list(new_state, sub_cond, param_map):
                    # Apply the effects if the `not` condition is satisfied
                    if "add" in weff:
                        apply_effects(new_state, weff["add"], param_map)
                    if "del" in weff:
                        remove_effects(new_state, weff["del"], param_map)

            elif cond[0] == "and":
                # Handle `and` conditions, including nested `exists` and `not`
                if all(_check_condition_list(new_state, sub_cond, param_map) for sub_cond in cond[1:]):
                    # Apply the effects if the `and` condition is satisfied
                    if "add" in weff:
                        apply_effects(new_state, weff["add"], param_map)
                    if "del" in weff:
                        remove_effects(new_state, weff["del"], param_map)


    # Handle forall_remove
    if "forall_remove" in eff:
        fr = eff["forall_remove"]
        var_name, var_type = fr["vars"][0]
        for o in all_objects:
            if var_type == "object" and o in characters:
                continue
            if var_type == "character" and o not in characters:
                continue

            cond_pred = fr["condition_negated_pred"][0]
            cond_args = substitute(fr["condition_negated_pred"][1:], {**param_map, var_name: o})
            condition_met = True

            # if cond args are the same then not process
            if cond_args[0] == cond_args[1]:
                continue

            if cond_pred in new_state and cond_args in new_state[cond_pred]:
                condition_met = False
            if condition_met:
                rm_pred = fr["remove_pred"][0]
                rm_args = substitute(fr["remove_pred"][1:], {**param_map, var_name: o})
                if rm_pred in new_state and rm_args in new_state[rm_pred]:
                    new_state[rm_pred].remove(rm_args)
                    if not new_state[rm_pred]:
                        del new_state[rm_pred]

    # Handle forall_add
    if "forall_add" in eff:
        fa = eff["forall_add"]
        var_name, var_type = fa["vars"][0]
        for o in all_objects:
            if var_type == "object" and o in characters:
                continue
            if var_type == "character" and o not in characters:
                continue

            cond_pred = fa["condition_pred"][0]
            cond_args = substitute(fa["condition_pred"][1:], {**param_map, var_name: o})

            # if cond args are the same then not process
            if cond_args[0] == cond_args[1]:
                continue

            if cond_pred in new_state and cond_args in new_state[cond_pred]:
                add_pred = fa["add_pred"][0]
                add_args = substitute(fa["add_pred"][1:], {**param_map, var_name: o})
                if add_pred not in new_state:
                    new_state[add_pred] = set()
                new_state[add_pred].add(add_args)

    return new_state


def compute_state_diff(old_state, new_state):
    """
    Compute the difference between old_state and new_state.
    Returns:
        added: dict of predicate -> set of tuples that were added
        removed: dict of predicate -> set of tuples that were removed
    """
    added = {}
    removed = {}

    all_preds = set(old_state.keys()).union(new_state.keys())
    for pred in all_preds:
        old_tuples = old_state.get(pred, set())
        new_tuples = new_state.get(pred, set())
        added_t = new_tuples - old_tuples
        removed_t = old_tuples - new_tuples
        if added_t:
            added[pred] = added_t
        if removed_t:
            removed[pred] = removed_t

    return added, removed


def check_goal(state, goal_conditions):
    """
    Check if the current state meets all goal conditions.
    goal_conditions is a dict of pred -> set of arg tuples.
    We must ensure for each pred and each arg tuple in goal_conditions,
    that they appear in state.
    """
    for pred, tuples in goal_conditions.items():
        if pred not in state:
            return False
        for t in tuples:
            if t not in state[pred]:
                return False
    return True

def substitute(args, param_map):
    return tuple(param_map.get(a, a) for a in args)

def _check_condition_list(state, cond, param_map):
    """Evaluate conditions (and/or/not/exists) and simple predicates."""
    if not cond:
        return True
    op = cond[0]
    if op == "and":
        return all(_check_condition_list(state, c, param_map) for c in cond[1:])
    elif op == "or":
        return any(_check_condition_list(state, c, param_map) for c in cond[1:])
    elif op == "not":
        return not _check_condition_list(state, cond[1], param_map)
    elif op == "exists":
        var_decl = cond[1]  # Variable declaration (e.g., (?obj2 - object))
        sub_cond = cond[2]  # Sub-condition to evaluate
        var_name, var_type = var_decl
        for obj in all_objects:
            if var_type == "object" and obj in characters:
                continue
            if var_type == "character" and obj not in characters:
                continue

            # Prevent self-referencing (e.g., ?obj2 == ?obj)
            if obj == param_map.get(var_name):
                continue

            # Extend param_map for this iteration and check the sub-condition
            new_param_map = {**param_map, var_name: obj}
            if _check_condition_list(state, sub_cond, new_param_map):
                return True
        return False
    else:
        # Simple predicate condition
        pred = op
        pred_args = substitute(cond[1:], param_map)
        return pred in state and pred_args in state[pred]

def check_preconditions(state, action_def, args):
    param_map = {p: a for p, a in zip(action_def["parameters"], args)}
    precond = action_def["preconditions"]
    return _check_condition_list(state, precond, param_map)

def apply_effects(new_state, effects_list, param_map):
    """Apply 'add' effects."""
    for eff in effects_list:
        pred = eff[0]
        aargs = substitute(eff[1:], param_map)
        if pred not in new_state:
            new_state[pred] = set()
        new_state[pred].add(aargs)

def remove_effects(new_state, effects_list, param_map):
    """Apply 'del' effects."""
    for eff in effects_list:
        pred = eff[0]
        dargs = substitute(eff[1:], param_map)
        if pred in new_state and dargs in new_state[pred]:
            new_state[pred].remove(dargs)
            if not new_state[pred]:
                del new_state[pred]


########################################################################################################################################

# Example sets of domain objects

########################################################################################################################################

from load_pddl import *
from copy import deepcopy



# Note: Available pddl: 101_2, 183_2, 310_2, 729_2
pddl_name = "310_2"
# Load the PDDL problem file
file_path = f"virtual_pddls/{pddl_name}.pddl"  # Replace with your PDDL problem file path
initial_state, goal_conditions, all_objects, characters = load_pddl_problem_line_by_line(file_path)


# Load the SAS plan file
sas_plan_path = f"sas_plans/{pddl_name}"  # Replace with your SAS plan file path
with open(sas_plan_path, "r") as plan_file:
    sas_plan = [line.strip() for line in plan_file.readlines() if line.strip() and not line.startswith(";")]

# Parse the SAS plan into action names and arguments
parsed_plan = []
for action_line in sas_plan:
    action_line = action_line.strip("()")  # Remove parentheses
    parts = action_line.split()
    action_name = parts[0]
    args = parts[1:]
    parsed_plan.append((action_name, args))

# Initialize the state
state = deepcopy(initial_state)

# Execute the plan
for step, (action_name, args) in enumerate(parsed_plan):
    print(f"Step {step + 1}: {action_name} {args}")
    if check_preconditions(state, actions[action_name], args):
        old_state = deepcopy(state)
        state = apply_action(state, actions[action_name], args)

        # Compute and print state differences
        added, removed = compute_state_diff(old_state, state)
        print("\nAction executed:", action_name, args)
        print("Added:", added)
        print("Removed:", removed)
    else:
        print(f"Preconditions not met for {action_name} {args}. Execution halted.")
        break

    # Check if the goal is reached
    if check_goal(state, goal_conditions):
        print("\nGoal reached after step:", step + 1)
        break
else:
    print("\nPlan executed but goal not reached.")
