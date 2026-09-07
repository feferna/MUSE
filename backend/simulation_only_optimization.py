import os
import shutil
import asyncio
from orchestrator import OptimizationOrchestrator

# Directories
CONFIG_DIR = os.path.abspath("./configs")

# Path to the external YAML configuration (used as-is)
EXTERNAL_CONFIG_PATH = os.path.abspath("./sample_configs/chartreader_default.yaml")

# Task identifier (adjust as needed)
task_id = "1"  # for Chartreader Task 1

def get_backup_dest(run_index):
    """Determine the backup destination folder for a given run, using run_index as sim_id."""
    return os.path.abspath(f"../user_study_data/simulator_{run_index}/task_{task_id}/")

def copy_external_config():
    """
    Copy the external configuration file to the orchestrator's expected location.
    No modifications are made.
    """
    dest_config = os.path.join(CONFIG_DIR, "config.yaml")
    shutil.copy(EXTERNAL_CONFIG_PATH, dest_config)
    print(f"Copied external config to {dest_config}")

async def simulate_user_feedback(orchestrator, max_trials=50):
    """
    Simulate user feedback by automatically choosing the first design from a finished batch.
    Stop the optimization once max_trials have been reached.
    """
    while not orchestrator.stop_requested:
        await asyncio.sleep(30)  # Check every second
        batch_info = orchestrator.get_current_batch_info()
        if (not batch_info.get("batch_in_progress", True) and batch_info.get("trials")
            and orchestrator.chosen_trials is None):
            trials = batch_info["trials"]
            # Always select the first design as the preferred one
            orchestrator.chosen_trials = [trials[0]["trial_number"]]
            orchestrator.not_chosen_trials = [t["trial_number"] for t in trials[1:]]
            print(f"Simulated feedback: chosen {orchestrator.chosen_trials}, not chosen {orchestrator.not_chosen_trials}")
        if orchestrator.get_completed_trials_count() >= max_trials:
            print(f"Reached {max_trials} trials. Stopping optimization.")
            await orchestrator.stop_optimization()
            break

async def run_optimization_run(run_index):
    """
    Run one optimization session:
      - Copy the external config.
      - Start the optimization (with simulated feedback).
      - After finishing, backup and clean up result folders.
    """
    print(f"\n--- Starting optimization run {run_index} ---")
    
    orchestrator = OptimizationOrchestrator()

    copy_external_config()
    
    # Run optimization and simulate feedback concurrently
    optimization_task = asyncio.create_task(orchestrator.start_optimization())
    feedback_task = asyncio.create_task(simulate_user_feedback(orchestrator, max_trials=50))
    
    result = await optimization_task
    await feedback_task
    print(f"Run {run_index} completed: {result}")
    
    backup_and_cleanup(run_index)
    print(f"Backup and cleanup for run {run_index} completed.")

def backup_and_cleanup(run_index):
    """
    Copy the "results", "databases", and "logs" directories to a backup folder,
    then clean up the originals to prepare for the next run.
    
    For the "results" directory, all contents except 'README.md' are removed.
    For the other directories, the entire folder is removed.
    """
    dirs_to_copy = ["results", "databases", "logs"]
    dest_path = get_backup_dest(run_index)
    os.makedirs(dest_path, exist_ok=True)
    
    for d in dirs_to_copy:
        if os.path.isdir(d):
            dest_dir = os.path.join(dest_path, d)
            shutil.copytree(d, dest_dir, dirs_exist_ok=True)
            if d == "results":
                # Remove all files and subdirectories in "results" except "README.md"
                for root, dirs, files in os.walk(d):
                    for file in files:
                        if file != "README.md":
                            os.remove(os.path.join(root, file))
                    for subdir in dirs:
                        shutil.rmtree(os.path.join(root, subdir))
            else:
                # Remove the entire directory
                shutil.rmtree(d)
            print(f"Moved {d} to {dest_path}")
        else:
            print(f"Warning: Directory {d} does not exist. Skipped.")
    print(f"Backup and cleanup completed for run {run_index}.")

async def main():
    num_runs = 4
    for run in range(1, num_runs + 1):
        await run_optimization_run(run)
    print("All optimization runs completed.")

if __name__ == "__main__":
    asyncio.run(main())
