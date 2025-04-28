import os
import shutil
import subprocess

# Configuration
ROOT_PROBLEMS = "/home/zxia545/_Code/embodied-agent-interface/src/virtualhome_eval/resources/virtualhome/problem_pddl"
SIMULATOR_PDDLS = "/home/zxia545/_Code/tony_debug_folder/1118_pddl_state/virtualhome_pddl_simulator/virtual_pddls"
SAS_PLANS_DIR = "/home/zxia545/_Code/tony_debug_folder/1118_pddl_state/virtualhome_pddl_simulator/sas_plans"
FAILED_DIR = os.path.join(SIMULATOR_PDDLS, "failed_generate_sas")
DOMAIN_PDDL = os.path.join(SIMULATOR_PDDLS, "virtualhome.pddl")
FAST_DOWNWARD = "/home/zxia545/_Code/tony_debug_folder/1118_pddl_state/pddlgym_planners/pddlgym_planners/FD/fast-downward.py"

# Ensure output directories exist
os.makedirs(SAS_PLANS_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)


def generate_plans():
    for task in os.listdir(ROOT_PROBLEMS):
        task_dir = os.path.join(ROOT_PROBLEMS, task)
        if not os.path.isdir(task_dir):
            continue

        for filename in os.listdir(task_dir):
            if not filename.endswith(".pddl"):
                continue

            src_path = os.path.join(task_dir, filename)
            dest_name = f"{task}_{filename}"
            dest_path = os.path.join(SIMULATOR_PDDLS, dest_name)

            # Copy problem PDDL with prefixed name
            shutil.copy(src_path, dest_path)

            try:
                # Run FastDownward planner
                subprocess.run([
                    FAST_DOWNWARD,
                    "--alias", "lama-first",
                    DOMAIN_PDDL,
                    dest_name
                ], cwd=SIMULATOR_PDDLS, check=True)

                # Move generated sas_plan to output folder
                plan_src = os.path.join(SIMULATOR_PDDLS, "sas_plan")
                plan_dst = os.path.join(
                    SAS_PLANS_DIR,
                    f"{os.path.splitext(dest_name)[0]}.sas_plan"
                )
                shutil.move(plan_src, plan_dst)
                print(f"Generated plan for {dest_name}")

            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                # Move failed problem file to failed directory
                failed_dest = os.path.join(FAILED_DIR, dest_name)
                shutil.move(dest_path, failed_dest)
                print(f"Failed to generate plan for {dest_name}: {e}")


def main():
    generate_plans()


if __name__ == "__main__":
    main()
