import sys
from copy import deepcopy
import argparse

# You would also have a loader function similar to `load_pddl_problem_line_by_line` adapted for behavior domain
from load_pddl_behavior import load_pddl_problem_line_by_line

#########################################################
# Actions definition for behavior domain
#########################################################

actions = {
    "navigate_to": {
        "parameters": ["?objto", "?agent"],
        "param_types": ["object", "agent"],
        "preconditions": ("not", ("in_reach_of_agent", "?objto")),
        "effects": {
            "add": [("in_reach_of_agent", "?objto")],
            # The domain states:
            # (forall (?objfrom - object) 
            #   (when (and (in_reach_of_agent ?objfrom) (not (same_obj ?objfrom ?objto))) 
            #         (not (in_reach_of_agent ?objfrom))))
            #
            # We'll represent this as a forall_remove with a compound condition.
            "forall_remove": {
                "vars": [("?objfrom", "object")],
                "condition_and": [
                    ("in_reach_of_agent", "?objfrom"),
                    ("not", ("same_obj", "?objfrom", "?objto"))
                ],
                "remove_pred": ("in_reach_of_agent", "?objfrom")
            }
        }
    },

    "grasp": {
        "parameters": ["?obj", "?agent"],
        "param_types": ["object", "agent"],
        "preconditions": ("and",
            ("not", ("holding", "?obj")),
            ("not", ("handsfull", "?agent")),
            ("in_reach_of_agent", "?obj"),
            ("not",
                ("exists", ("?obj2", "object"),
                    ("and",
                        ("inside", "?obj", "?obj2"),
                        ("not", ("open", "?obj2"))
                    )
                )
            )
        ),
        "effects": {
            "add": [("holding", "?obj"), ("handsfull", "?agent")],
            # The action has a forall with multiple removals:
            # (forall (?other_obj - object)
            #   (and 
            #       (not (inside ?obj ?other_obj))
            #       (not (ontop ?obj ?other_obj))
            #       (not (under ?obj ?other_obj))
            #       (not (under ?other_obj ?obj))
            #       (not (nextto ?obj ?other_obj))
            #       (not (nextto ?other_obj ?obj))
            #       (not (onfloor ?obj ?other_obj))
            #   )
            # )
            #
            # We'll store these as multiple predicates to remove under a single forall_remove.
            "forall_remove_multiple": {
                "vars": [("?other_obj", "object")],
                "remove_preds": [
                    ("inside", "?obj", "?other_obj"),
                    ("ontop", "?obj", "?other_obj"),
                    ("under", "?obj", "?other_obj"),
                    ("under", "?other_obj", "?obj"),
                    ("nextto", "?obj", "?other_obj"),
                    ("nextto", "?other_obj", "?obj"),
                    ("onfloor", "?obj", "?other_obj")
                ]
            }
        }
    },

    "release": {
        "parameters": ["?obj", "?agent"],
        "param_types": ["object", "agent"],
        "preconditions": ("holding", "?obj"),
        "effects": {
            "del": [("holding", "?obj"), ("handsfull", "?agent")]
        }
    },

    "place_ontop": {
        "parameters": ["?obj_in_hand", "?obj", "?agent"],
        "param_types": ["object", "object", "agent"],
        "preconditions": ("and",
            ("holding", "?obj_in_hand"),
            ("in_reach_of_agent", "?obj")
        ),
        "effects": {
            "add": [("ontop", "?obj_in_hand", "?obj")],
            "del": [("holding", "?obj_in_hand"), ("handsfull", "?agent")]
        }
    },

    "place_inside": {
        "parameters": ["?obj_in_hand", "?obj", "?agent"],
        "param_types": ["object", "object", "agent"],
        "preconditions": ("and",
            ("holding", "?obj_in_hand"),
            ("in_reach_of_agent", "?obj"),
            ("open", "?obj")
        ),
        "effects": {
            "add": [("inside", "?obj_in_hand", "?obj")],
            "del": [("holding", "?obj_in_hand"), ("handsfull", "?agent")]
        }
    },
    "open": {
        "parameters": ["?obj", "?agent"],
        "param_types": ["object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("not", ("open", "?obj")),
            ("not", ("handsfull", "?agent"))
        ),
        "effects": {
            "add": [("open", "?obj")]
        }
    },

    "close": {
        "param_types": ["object", "agent"],
        "parameters": ["?obj", "?agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("open", "?obj"),
            ("not", ("handsfull", "?agent"))
        ),
        "effects": {
            "del": [("open", "?obj")]
        }
    },

    "slice": {
        "parameters": ["?obj", "?knife", "?agent"],
        "param_types": ["object", "knife_n_01", "agent"],
        "preconditions": ("and",
            ("holding", "?knife"),
            ("in_reach_of_agent", "?obj")
        ),
        "effects": {
            "add": [("sliced", "?obj")]
        }
    },

    "slice-carvingknife": {
        "parameters": ["?obj", "?knife", "?board", "?agent"],
        "param_types": ["object", "carving_knife_n_01", "countertop_n_01", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("holding", "?knife"),
            ("ontop", "?obj", "?board"),
            ("not", ("sliced", "?obj"))
        ),
        "effects": {
            "add": [("sliced", "?obj")]
        }
    },

    "place_onfloor": {
        "parameters": ["?obj_in_hand", "?floor", "?agent"],
        "param_types": ["object", "floor_n_01", "agent"],
        "preconditions": ("and",
            ("holding", "?obj_in_hand"),
            ("in_reach_of_agent", "?floor")
        ),
        "effects": {
            "add": [("onfloor", "?obj_in_hand", "?floor")],
            "del": [("holding", "?obj_in_hand"), ("handsfull", "?agent")]
        }
    },

    "place_nextto": {
        "parameters": ["?obj_in_hand", "?obj", "?agent"],
        "param_types": ["object", "object", "agent"],
        "preconditions": ("and",
            ("holding", "?obj_in_hand"),
            ("in_reach_of_agent", "?obj")
        ),
        "effects": {
            "add": [("nextto", "?obj_in_hand", "?obj"),
                    ("nextto", "?obj", "?obj_in_hand")],
            "del": [("holding", "?obj_in_hand"), ("handsfull", "?agent")]
        }
    },

    "place_nextto_ontop": {
        "parameters": ["?obj_in_hand", "?obj1", "?obj2", "?agent"],
        "param_types": ["object", "object", "object", "agent"],
        "preconditions": ("and",
            ("holding", "?obj_in_hand"),
            ("in_reach_of_agent", "?obj1")
        ),
        "effects": {
            "add": [("nextto", "?obj_in_hand", "?obj1"),
                    ("nextto", "?obj1", "?obj_in_hand"),
                    ("ontop", "?obj_in_hand", "?obj2")],
            "del": [("holding", "?obj_in_hand"), ("handsfull", "?agent")]
        }
    },

    "clean_stained_brush": {
        "parameters": ["?scrub_brush", "?obj", "?agent"],
        "param_types": ["scrub_brush_n_01", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("stained", "?obj"),
            ("soaked", "?scrub_brush"),
            ("holding", "?scrub_brush")
        ),
        "effects": {
            "del": [("stained", "?obj")]
        }
    },

    "clean_stained_cloth": {
        "parameters": ["?rag", "?obj", "?agent"],
        "param_types": ["piece_of_cloth_n_01", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("stained", "?obj"),
            ("soaked", "?rag"),
            ("holding", "?rag")
        ),
        "effects": {
            "del": [("stained", "?obj")]
        }
    },

    "clean_stained_handowel": {
        "parameters": ["?hand_towel", "?obj", "?agent"],
        "param_types": ["hand_towel_n_01", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("stained", "?obj"),
            ("soaked", "?hand_towel"),
            ("holding", "?hand_towel")
        ),
        "effects": {
            "del": [("stained", "?obj")]
        }
    },

    "clean_stained_towel": {
        "parameters": ["?hand_towel", "?obj", "?agent"],
        "param_types": ["towel_n_01", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("stained", "?obj"),
            ("soaked", "?hand_towel"),
            ("holding", "?hand_towel")
        ),
        "effects": {
            "del": [("stained", "?obj")]
        }
    },

    "clean_stained_dishtowel": {
        "parameters": ["?hand_towel", "?obj", "?agent"],
        "param_types": ["dishtowel_n_01", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("stained", "?obj"),
            ("soaked", "?hand_towel"),
            ("holding", "?hand_towel")
        ),
        "effects": {
            "del": [("stained", "?obj")]
        }
    },

    "clean_stained_dishwasher": {
        "parameters": ["?dishwasher", "?obj", "?agent"],
        "param_types": ["dishwasher_n_01", "object", "agent"],
        "preconditions": ("and",
            ("holding", "?obj"),
            ("in_reach_of_agent", "?dishwasher")
        ),
        "effects": {
            "del": [("stained", "?obj")]
        }
    },

    "clean_stained_rag": {
        "parameters": ["?rag", "?obj", "?agent"],
        "param_types": ["rag_n_01", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("stained", "?obj"),
            ("soaked", "?rag"),
            ("holding", "?rag")
        ),
        "effects": {
            "del": [("stained", "?obj")]
        }
    },
    
    "soak": {
        "parameters": ["?obj1", "?sink", "?agent"],
        "param_types": ["object", "sink_n_01", "agent"],
        "preconditions": ("and",
            ("holding", "?obj1"),
            ("in_reach_of_agent", "?sink"),
            ("toggled_on", "?sink")
        ),
        "effects": {
            "add": [("soaked", "?obj1")]
        }
    },

    "soak_teapot": {
        "parameters": ["?obj1", "?agent", "?teapot"],
        "param_types": ["object", "agent", "teapot_n_01"],
        "preconditions": ("and",
            ("holding", "?obj1"),
            ("in_reach_of_agent", "?teapot")
        ),
        "effects": {
            "add": [("soaked", "?obj1")]
        }
    },

    "place_under": {
        "parameters": ["?obj_in_hand", "?obj", "?agent"],
        "param_types": ["object", "object", "agent"],
        "preconditions": ("and",
            ("holding", "?obj_in_hand"),
            ("in_reach_of_agent", "?obj")
        ),
        "effects": {
            "add": [("under", "?obj_in_hand", "?obj")],
            "del": [("holding", "?obj_in_hand"), ("handsfull", "?agent")]
        }
    },

    "toggle_on": {
        "parameters": ["?obj", "?agent"],
        "param_types": ["object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("not", ("handsfull", "?agent"))
        ),
        "effects": {
            "add": [("toggled_on", "?obj")]
        }
    },

    "clean_dusty_rag": {
        "param_types": ["rag_n_01", "object", "agent"],
        "parameters": ["?rag", "?obj", "?agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("dusty", "?obj"),
            ("holding", "?rag")
        ),
        "effects": {
            "del": [("dusty", "?obj")]
        }
    },

    "clean_dusty_towel": {
        "parameters": ["?towel", "?obj", "?agent"],
        "param_types": ["towel_n_01", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("dusty", "?obj"),
            ("holding", "?towel")
        ),
        "effects": {
            "del": [("dusty", "?obj")]
        }
    },

    "clean_dusty_cloth": {
        "parameters": ["?rag", "?obj", "?agent"],
        "param_types": ["piece_of_cloth_n_01", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("dusty", "?obj"),
            ("holding", "?rag")
        ),
        "effects": {
            "del": [("dusty", "?obj")]
        }
    },

    "clean_dusty_brush": {
        "parameters": ["?scrub_brush", "?obj", "?agent"],
        "param_types": ["scrub_brush_n_01", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("dusty", "?obj"),
            ("holding", "?scrub_brush")
        ),
        "effects": {
            "del": [("dusty", "?obj")]
        }
    },

    "clean_dusty_vacuum": {
        "parameters": ["?vacuum", "?obj", "?agent"],
        "param_types": ["vacuum_n_04", "object", "agent"],
        "preconditions": ("and",
            ("in_reach_of_agent", "?obj"),
            ("dusty", "?obj"),
            ("holding", "?vacuum")
        ),
        "effects": {
            "del": [("dusty", "?obj")]
        }
    },

    "freeze": {
        "parameters": ["?obj", "?fridge"],
        "param_types": ["object", "electric_refrigerator_n_01"],
        "preconditions": ("and",
            ("inside", "?obj", "?fridge"),
            ("not", ("frozen", "?obj"))
        ),
        "effects": {
            "add": [("frozen", "?obj")]
        }
    },

    "cook": {
        "parameters": ["?obj", "?pan"],
        "param_types": ["object", "pan_n_01"],
        "preconditions": ("and",
            ("ontop", "?obj", "?pan"),
            ("not", ("cooked", "?obj"))
        ),
        "effects": {
            "add": [("cooked", "?obj")]
        }
    }

}


#########################################################
# Core Functions (similar to VirtualHome run_and_act.py)
#########################################################

def substitute(args, param_map):
    return tuple(param_map.get(a, a) for a in args)

def _check_condition_list(state, cond, param_map, all_objects, characters):
    """Evaluate a nested condition structure."""
    if not cond:
        return True
    op = cond[0]
    if op == "and":
        return all(_check_condition_list(state, c, param_map, all_objects, characters) for c in cond[1:])
    elif op == "or":
        return any(_check_condition_list(state, c, param_map, all_objects, characters) for c in cond[1:])
    elif op == "not":
        return not _check_condition_list(state, cond[1], param_map, all_objects, characters)
    elif op == "exists":
        var_decl = cond[1]
        sub_cond = cond[2]
        var_name, var_type = var_decl
        for obj in all_objects:
            # Type check if needed, else skip
            # For now assume all are objects except if var_type == agent, handle it if needed
            if var_type == "agent" and obj not in characters:
                continue
            if var_type == "object" and obj in characters:
                continue
            new_param_map = {**param_map, var_name: obj}
            if _check_condition_list(state, sub_cond, new_param_map, all_objects, characters):
                return True
        return False
    else:
        # Simple predicate condition
        pred = op
        pred_args = substitute(cond[1:], param_map)
        return pred in state and pred_args in state[pred]

def check_preconditions(state, action_def, args, all_objects, characters, object_types):
    param_map = {p: a for p, a in zip(action_def["parameters"], args)}
    # First, check types
    if "param_types" in action_def:
        for (param, arg, ptype) in zip(action_def["parameters"], args, action_def["param_types"]):
            # Check if arg matches ptype
            if ptype == "agent":
                # arg must be in characters
                if arg not in characters:
                    return False
            else:
                # arg must be in all_objects or characters, and must match object_types
                if arg not in object_types:
                    return False
                # If ptype == "object", it's okay as long as it's not agent.
                # If ptype is a specific subtype (e.g., knife_n_01), must match exactly
                if ptype != "object" and ptype != "agent":
                    if object_types[arg] != ptype:
                        return False
                # If ptype == "object", any object (not an agent) is allowed
                if ptype == "object":
                    # Ensure it's not an agent
                    if arg in characters:
                        return False

    precond = action_def["preconditions"]
    if not precond:
        return True
    return _check_condition_list(state, precond, param_map, all_objects, characters)


def apply_effects(state, effects_list, param_map):
    """Add positive effects."""
    for eff in effects_list:
        pred = eff[0]
        aargs = substitute(eff[1:], param_map)
        if pred not in state:
            state[pred] = set()
        state[pred].add(aargs)

def remove_effects(state, effects_list, param_map):
    """Remove negative effects."""
    for eff in effects_list:
        pred = eff[0]
        dargs = substitute(eff[1:], param_map)
        if pred in state and dargs in state[pred]:
            state[pred].remove(dargs)
            if not state[pred]:
                del state[pred]

def apply_action(state, action_def, args, all_objects, characters, object_types):
    new_state = deepcopy(state)
    param_map = {p: a for p, a in zip(action_def["parameters"], args)}
    eff = action_def.get("effects", {})

    # Unconditional add/del
    if "add" in eff:
        apply_effects(new_state, eff["add"], param_map)
    if "del" in eff:
        remove_effects(new_state, eff["del"], param_map)

    # Handle `when` effects if any (similar logic can be added as needed)
    # ...

    # Handle forall_remove
    # This is more complex now since we may have "condition_and" or multiple conditions
    if "forall_remove" in eff:
        fr = eff["forall_remove"]
        var_name, var_type = fr["vars"][0]
        for o in all_objects:
            if var_type == "object" and o in characters:
                continue
            if var_type == "agent" and o not in characters:
                continue
            # Check conditions in condition_and or condition_pred
            conditions_met = True
            if "condition_and" in fr:
                for c in fr["condition_and"]:
                    if not _check_condition_list(new_state, c, {**param_map, var_name: o}, all_objects, characters):
                        conditions_met = False
                        break
            elif "condition_pred" in fr:
                if not _check_condition_list(new_state, fr["condition_pred"], {**param_map, var_name: o}, all_objects, characters):
                    conditions_met = False
            if conditions_met:
                rm_pred = fr["remove_pred"][0]
                rm_args = substitute(fr["remove_pred"][1:], {**param_map, var_name: o})
                if rm_pred in new_state and rm_args in new_state[rm_pred]:
                    new_state[rm_pred].remove(rm_args)
                    if not new_state[rm_pred]:
                        del new_state[rm_pred]

    # Handle forall_remove_multiple
    if "forall_remove_multiple" in eff:
        fr = eff["forall_remove_multiple"]
        var_name, var_type = fr["vars"][0]
        # This is unconditional removal of multiple predicates for each object
        # No condition specified, so it's always apply these removals to all objects
        # If you need conditions, adapt similarly to above.
        for o in all_objects:
            if var_type == "object" and o in characters:
                continue
            if var_type == "agent" and o not in characters:
                continue
            # Remove all listed preds
            for pred_tuple in fr["remove_preds"]:
                pred = pred_tuple[0]
                dargs = substitute(pred_tuple[1:], {**param_map, var_name: o})
                if pred in new_state and dargs in new_state[pred]:
                    new_state[pred].remove(dargs)
                    if not new_state[pred]:
                        del new_state[pred]

    return new_state

def compute_state_diff(old_state, new_state):
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
    for pred, tuples in goal_conditions.items():
        if pred not in state:
            return False
        for t in tuples:
            if t not in state[pred]:
                return False
    return True

#########################################################
# Example main execution (similar to VirtualHome)
#########################################################

if __name__ == "__main__":
    # Provide paths to PDDL problem and SAS plan
    pddl_file_name = "boxing_books_up_for_storage_0_Benevolence_1_int_0_2021-09-10_15-35-47"
    
    problem_file = f"behavior_pddls/{pddl_file_name}.pddl"
    plan_file = f"sas_plans/{pddl_file_name}"

    initial_state, goal_conditions, all_objects, characters, object_types = load_pddl_problem_line_by_line(problem_file)

    with open(plan_file, "r") as f:
        sas_plan = [line.strip() for line in f if line.strip() and not line.startswith(";")]

    # Parse plan
    parsed_plan = []
    for action_line in sas_plan:
        action_line = action_line.strip("()")
        parts = action_line.split()
        action_name = parts[0]
        args = parts[1:]
        parsed_plan.append((action_name, args))

    state = deepcopy(initial_state)

    for step, (action_name, args) in enumerate(parsed_plan):
        print(f"Step {step+1}: {action_name} {args}")
        if action_name not in actions:
            print("Action not defined in dictionary.")
            break
        if check_preconditions(state, actions[action_name], args, all_objects, characters, object_types):
            old_state = deepcopy(state)
            state = apply_action(state, actions[action_name], args, all_objects, characters, object_types)

            added, removed = compute_state_diff(old_state, state)
            print("Action executed:", action_name, args)
            print("Added:", added)
            print("Removed:", removed)
            print()
            if check_goal(state, goal_conditions):
                print("Goal reached after step:", step + 1)
                break
        else:
            print(f"Preconditions not met for {action_name} {args}. Execution halted.")
            break
    else:
        print("Plan executed but goal not reached.")
