from walker import BiomechanicsTools
from walker.OCP_plugin_gait import OCPPluginGait

# --- Options --- #
participant_name = "VIF_04"
trial_names = ["Cond0001"]
data_path = "data"
body_mass = 71  # Kg

kinematic_model_file_path = f"{data_path}/{participant_name}.bioMod"
static_trial = f"{data_path}/{participant_name}_Statique.c3d"
trials = [
    f"{data_path}/{participant_name}_{condition_name}.c3d" for condition_name in trial_names
]

# --------------- #


def main():
    print(kinematic_model_file_path)

    # Generate the personalized kinematic model
    tools = BiomechanicsTools(OCPPluginGait(name=participant_name, body_mass=body_mass, include_upper_body=True))
    tools.personalize_model(static_trial, kinematic_model_file_path)

    # Perform some biomechanical computation
    for trial in trials:
        print(trial)
        tools.process_trial(trial, compute_automatic_events=False)

    # TODO: Bioviz vizual bug with the end of the trial when resizing the window
    # TODO: Record a tutorial


if __name__ == "__main__":
    main()
